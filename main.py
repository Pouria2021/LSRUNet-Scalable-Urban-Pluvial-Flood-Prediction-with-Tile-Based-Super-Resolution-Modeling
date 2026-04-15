"""
Distributed Training Script for Flood Super-Resolution Deep Learning Models.

This script trains super-resolution neural networks to predict high-resolution
flood inundation maps from low-resolution inputs, rainfall data, and static features.

Usage:
    torchrun --nproc_per_node=4 main.py --case_config_path config.json

The training supports:
    - Multiple model architectures: ResUNet, Swin Transformer V2, MaxViT
    - Multi-task learning (depth + mask prediction)
    - Autoregressive and non-autoregressive training modes
    - Distributed training with PyTorch DDP
    - Cosine annealing learning rate with warm restarts
"""

import os
import shutil
import argparse
import json
import queue
import pandas as pd
import torch
import numpy as np
import contextlib

import geopandas as gpd

import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_autoregressive_dataloader, preload_dataV2
from model_unet import ResUNet_aux, ResUNet_aux_MTL
from model_vit import MaxVIT
from model_swinT import SwinV2
from metric import parallel_calculate_metricsV2, FloodPredLoss
from utils import calculate_cosLR_cycles, get_progress_ratio_of_cycle


def read_case_file(file_path: str) -> list:
    """
    Read case identifiers from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing case identifiers.
        The file must have a column named 'case_ID'.

    Returns
    -------
    list
        List of case identifier strings.

    Examples
    --------
    >>> cases = read_case_file('train_cases.csv')
    >>> print(cases)
    ['case_001', 'case_002', 'case_003']
    """
    df = pd.read_csv(file_path)
    return [str(s) for s in df['case_ID'].tolist()]

def unet_train_mgpu_subproc(case_config_path: str) -> None:
    """
    Execute distributed training of flood super-resolution model using multi-GPU DDP.

    This function handles the complete training pipeline including:
    - Distributed training initialization with NCCL backend
    - Model, optimizer, and learning rate scheduler setup
    - Data loading with DistributedSampler
    - Training loop with non-autoregressive (NAR) and autoregressive (AR) phases
    - Validation with multiple evaluation metrics
    - Model checkpointing and logging

    Parameters
    ----------
    case_config_path : str
        Path to the JSON configuration file containing all training parameters.
        See train_config_napJQX_MTL.json for reference structure.

    Configuration Structure
    -----------------------
    The JSON config must contain:
        - case_identifier : str - Unique identifier for this training run
        - model : dict - Model architecture and hyperparameters
        - dataset : dict - Data paths and dataset configuration
        - learning_settings : dict - Training hyperparameters

    Training Phases
    ---------------
    1. Non-Autoregressive (NAR): Direct prediction, faster convergence
    2. Autoregressive (AR): Sequential prediction using previous outputs

    Outputs
    -------
    - modelBest_{identifier}_NAR.pth : Best NAR phase checkpoint
    - modelBest_{identifier}_AR.pth : Best AR phase checkpoint
    - metricRecords_{identifier}.txt : Training/validation metrics log
    - caseConfig_{identifier}.json : Copy of configuration file

    Notes
    -----
    This function should be launched via torchrun for proper DDP initialization:
        torchrun --nproc_per_node=N main.py --case_config_path config.json

    The function uses NCCL backend and expects CUDA-enabled GPUs.
    """
    ######## ======== Initialize Distributed Training ========#############
    # here we use `torchrun` so we don't need to initialize the process group manually 
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # torch.manual_seed(3407)
    # torch.manual_seed(12306)
    # torch.manual_seed(1017)
    # torch.manual_seed(243)
    torch.manual_seed(116)

    #################################################
    ######## Training configuration ##########
    #################################################

    with open(case_config_path, 'r') as f:
        UserParameters = json.load(f)

    case_identifier = UserParameters['case_identifier']
    model_output_dir = UserParameters['model']['save_root_dir']

    num_model_target_channels = UserParameters['model']['num_target_channels']
    num_aux_target_channels = UserParameters['model']['num_aux_target_channels']
    num_model_levels = UserParameters['model']['num_levels']

    # Model parameters
    MTL_FLAG = UserParameters['model']['mtl_flag']
    MODEL_TYPE = UserParameters['model']['type']
    MODEL_INI_STATE = UserParameters['model']['initial_state']

    # Dataset parameters
    tile_keywords = UserParameters['dataset']['tile_keywords']
    tile_polygon_path = UserParameters['dataset']['tile_polygon_path']

    train_dir = UserParameters['dataset']['train_root_dir']
    train_cases_file = UserParameters['dataset']['train_case_list']

    val_dir = UserParameters['dataset']['valid_root_dir']
    val_cases_file = UserParameters['dataset']['valid_case_list']
    
    flood_LR_dir_train = UserParameters['dataset']['flood_LR_train_root_dir']
    flood_LR_dir_valid = UserParameters['dataset']['flood_LR_valid_root_dir']

    aux_file_path_info = {}
    for key, value in UserParameters['dataset']['aux_file_info'].items():
        aux_file_path_info[key] = value

    mask_file_prefix = UserParameters['dataset']['mask_file_prefix']

    NUM_STATIC_CHANNELS = len(aux_file_path_info)
    print(flood_LR_dir_train)
    if len(flood_LR_dir_train) > 0 and os.path.exists(flood_LR_dir_train):
        NUM_STATIC_CHANNELS = NUM_STATIC_CHANNELS + 1  # Add one for the lower resolution channel
        use_flood_LR = True
    else:
        use_flood_LR = False

    # Learning parameters
    BATCH_SIZE = UserParameters['learning_settings']['batch_size']
    LEARNING_NUM_WORKERS = UserParameters['learning_settings']['num_workers']
    SAMPLE_WEIGHT_TABLE = UserParameters['learning_settings']['sample_weight_path']

    # Learning rates with world_size scaling
    LEARNING_RATE_NAR = UserParameters['learning_settings']['learning_rates']['nar_base'] * world_size
    LEARNING_RATE_SCALING_FACTOR_NAR = UserParameters['learning_settings']['learning_rates']['scaling_factor_nar']
    LEARNING_RATE_AR = UserParameters['learning_settings']['learning_rates']['ar_base'] * world_size
    LEARNING_RATE_SCALING_FACTOR_AR = UserParameters['learning_settings']['learning_rates']['scaling_factor_ar']

    if 'weight_decay' not in UserParameters['learning_settings']:
        WEIGHT_DECAY = 1e-3
    else:
        WEIGHT_DECAY = UserParameters['learning_settings']['weight_decay']
    
    # Sequence parameters
    MAX_ERROR_SEQUENCE = UserParameters['learning_settings']['autoregressive']['max_forward_step']
    NUM_PREV_STEP = UserParameters['learning_settings']['autoregressive']['num_prev_step']
    
    # Epoch parameters
    NON_AUTOREGRESSIVE_EPOCHS = UserParameters['learning_settings']['epochs']['num_non_autoregressive']
    AUTOREGRESSIVE_EPOCHS = UserParameters['learning_settings']['epochs']['num_autoregressive']
    EPOCHS = NON_AUTOREGRESSIVE_EPOCHS + AUTOREGRESSIVE_EPOCHS * (MAX_ERROR_SEQUENCE - 1)
    
    MTL_DYNAMIC_WEIGHT_FLAG = UserParameters['learning_settings']['mtl_dynamic_weight']
    MTL_DEPTH_WEIGHT = UserParameters['learning_settings'].get('mtl_depth_weight', 1.0)
    MTL_MASK_WEIGHT = UserParameters['learning_settings'].get('mtl_mask_weight', 1.0)
    
    # Loss parameters
    delta = UserParameters['learning_settings']['loss']['delta']
    thresholds = UserParameters['learning_settings']['loss']['thresholds']

    #################################################
    ######## Model training initialization ##########
    #################################################

    num_tile = len(gpd.read_file(tile_polygon_path))
    print(f"Number of tiles: {num_tile}")

    if not os.path.exists(model_output_dir):
        print(f"Creating output directory: {model_output_dir}")
        os.makedirs(model_output_dir)
    print(f"Model output directory: {model_output_dir}")
    # ------copy the case config file to the model output directory
    shutil.copy(case_config_path, os.path.join(model_output_dir, f'caseConfig_{case_identifier}.json'))

    train_cases = read_case_file(train_cases_file)
    val_cases = read_case_file(val_cases_file)

    T0 = 5
    TMULT = 2
    lr_cycle_list = []
    nar_cycles = calculate_cosLR_cycles(0, T0, TMULT, NON_AUTOREGRESSIVE_EPOCHS)
    lr_cycle_list.extend(nar_cycles)

    T_start = NON_AUTOREGRESSIVE_EPOCHS
    for i in range(0, MAX_ERROR_SEQUENCE - 1):
        ar_cycles = calculate_cosLR_cycles(T_start, T0, TMULT, T_start + AUTOREGRESSIVE_EPOCHS)
        T_start = T_start + AUTOREGRESSIVE_EPOCHS
        lr_cycle_list.extend(ar_cycles)
    print("Learning rate cycles for training:", lr_cycle_list)

    # Open a text file to log the metrics
    metrics_file_path = os.path.join(model_output_dir, f'metricRecords_{case_identifier}.txt')
    with open(metrics_file_path, 'w') as metrics_file:
        header = "Epoch,Train Metric"
        for threshold in thresholds:
            scaled_threshold = int(threshold * 100)
            header += f",RMSE_{scaled_threshold},IoU_{scaled_threshold},CSI_{scaled_threshold},F2-Score_{scaled_threshold},POD_{scaled_threshold},FAR_{scaled_threshold}, Bias_{scaled_threshold}"
        metrics_file.write(header + "\n")

    # Initialize model, criterion, and optimizer
    # model = UNet(n_channels=2 * SEQUENCE_LENGTH, n_classes=1, bilinear=True).to(device)
    if MTL_FLAG:
        print("Using MTL model architecture.")
        model = ResUNet_aux_MTL(num_input_channels=NUM_PREV_STEP + 1, 
                                num_target_channels=num_model_target_channels,
                                num_levels=num_model_levels, 
                                num_aux_target_channels=[num_aux_target_channels for i in range(0, NUM_STATIC_CHANNELS)], 
                                num_aux_levels_list=[num_model_levels for i in range(0, NUM_STATIC_CHANNELS)]).to(device)
        
        if MODEL_INI_STATE is not None and os.path.exists(MODEL_INI_STATE):
            print(f"Loading model initial state from: {MODEL_INI_STATE}")
            # model.load_state_dict(torch.load(MODEL_INI_STATE, map_location=device))
            model.load_pretrained_model(trained_record=MODEL_INI_STATE, map_location=device)
        
        model_ddp = DDP(model, device_ids=[device], find_unused_parameters=False)
        if MTL_DYNAMIC_WEIGHT_FLAG:
            scale_depth = torch.zeros((1,), requires_grad=True, device="cuda")
            scale_mask = torch.zeros((1,), requires_grad=True, device="cuda")
            params = list(model_ddp.parameters()) + [scale_depth, scale_mask]
        else:
            scale_depth = MTL_DEPTH_WEIGHT
            scale_mask = MTL_MASK_WEIGHT
            # scale_depth = 0.0
            # scale_mask = 10.0
            # params = model_ddp.parameters()   # here we can not use `model_ddp.parameters()` (which returns a generator) when we use two optimizers
            params = list(model_ddp.parameters())
    else:
        if MODEL_TYPE == "UNet":
            model = ResUNet_aux(num_input_channels=NUM_PREV_STEP + 1, 
                                num_target_channels=num_model_target_channels, 
                                num_levels=num_model_levels, 
                                num_aux_target_channels=[num_aux_target_channels for i in range(0, NUM_STATIC_CHANNELS)], 
                                num_aux_levels_list=[num_model_levels for i in range(0, NUM_STATIC_CHANNELS)]).to(device)
        elif MODEL_TYPE == "MaxVIT":
            model = MaxVIT(num_input_channels=NUM_PREV_STEP + 1, 
                            num_target_channels=num_model_target_channels, 
                            num_levels=num_model_levels, 
                            num_aux_target_channels=[num_aux_target_channels for i in range(0, NUM_STATIC_CHANNELS)], 
                            num_aux_levels_list=[num_model_levels for i in range(0, NUM_STATIC_CHANNELS)]).to(device)
        elif MODEL_TYPE == "SwinT":
            model = SwinV2(num_input_channels=NUM_PREV_STEP + 1, 
                            num_target_channels=num_model_target_channels, 
                            num_levels=num_model_levels, 
                            num_aux_target_channels=[num_aux_target_channels for i in range(0, NUM_STATIC_CHANNELS)], 
                            num_aux_levels_list=[num_model_levels for i in range(0, NUM_STATIC_CHANNELS)]).to(device)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
        
        if MODEL_INI_STATE is not None and os.path.exists(MODEL_INI_STATE):
            print(f"Loading model initial state from: {MODEL_INI_STATE}")
            model.load_state_dict(torch.load(MODEL_INI_STATE, map_location=device))
            
        print(f"[Rank {dist.get_rank()}] Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model_ddp = DDP(model, device_ids=[device], find_unused_parameters=False)
        params = list(model_ddp.parameters())

    # ------according to some practical experiences,
    # ------we can start with Adam optimizer and then switch to SGD with momentum
    #optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    print("Rank ", rank, "\tNumber of Parameters: ", sum([p.numel() for p in model_ddp.parameters() if p.requires_grad]))
    optimizer_NAR = torch.optim.AdamW(params, lr=LEARNING_RATE_NAR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    optimizer_AR = torch.optim.AdamW(params, lr=LEARNING_RATE_AR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=LEARNING_RATE_MIN)
    lr_scheduler_NAR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_NAR, T_0=T0, T_mult=TMULT, eta_min=LEARNING_RATE_NAR * LEARNING_RATE_SCALING_FACTOR_NAR)  # 5, 15 (5+10), 35 (5+10+20), 75 (5+10+20+40)
    lr_scheduler_AR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_AR, T_0=T0, T_mult=TMULT, eta_min=LEARNING_RATE_AR * LEARNING_RATE_SCALING_FACTOR_AR)

    #torch.autograd.set_detect_anomaly(True)    #####disable when you want to run it

    # Initialize loss function
    loss_object = FloodPredLoss(cuda=True, weight_csv_path=SAMPLE_WEIGHT_TABLE, adaptive_mask_loss_wight=MTL_DYNAMIC_WEIGHT_FLAG)
    # criterion = loss_object.build_loss("weighted_HuberLoss_withBoundaryMask")
    criterion = loss_object.build_loss("weighted_HuberLoss_withBinMask_withBoundaryMask")
    # criterion = nn.SmoothL1Loss(beta=delta)
    # criterion = nn.MSELoss() stop learn

    # Preload data before training
    tile_ID_list = list(range(1, num_tile + 1))
    # [NOTE] `torch.utils.data.distributed.DistributedSampler` will shard the dataset across all processes
    # thus, we do not need to manually assign cases to each rank
    # assigned_cases = train_cases[rank::world_size]
    assigned_cases = train_cases
    num_train_cases = len(assigned_cases)
    print(f"[Rank {rank}] Assigned {num_train_cases} training cases: {assigned_cases}")
    train_data = preload_dataV2(train_cases, train_dir, 
                              #timestep_list,
                              rank, world_size, 
                              tile_ID_list=tile_ID_list,
                              aux_file_info=aux_file_path_info,
                              mask_file_prefix=mask_file_prefix,
                              flood_LR_dir=flood_LR_dir_train,
                              loading_ratio=0.0,
                              tile_kw=tile_keywords,
                              flood_LR_kw="FloodLR", 
                              flood_HR_kw="Flood", 
                              rainfall_kw="Rainfall", 
                              timestep_kw="TS",
                              zero_discard_flag=True,
                              num_cpu=16)
    
    # assigned_cases = val_cases[rank::world_size]
    assigned_cases = val_cases
    num_val_cases = len(assigned_cases)
    print(f"[Rank {rank}] Assigned {num_val_cases} validation cases: {assigned_cases}")
    val_data = preload_dataV2(val_cases, val_dir, 
                            #timestep_list,
                            rank, world_size,
                            tile_ID_list=tile_ID_list,
                            aux_file_info=aux_file_path_info,
                            mask_file_prefix=mask_file_prefix,
                            flood_LR_dir=flood_LR_dir_valid,
                            loading_ratio=0.0,
                            tile_kw=tile_keywords,
                            flood_LR_kw="FloodLR", 
                            flood_HR_kw="Flood", 
                            rainfall_kw="Rainfall", 
                            timestep_kw="TS",
                            zero_discard_flag=True,
                            num_cpu=16)

    dist.barrier()  # Ensure all processes are synchronized
    #################################################
    ######### Training & Validation  ################
    #################################################
    # Initialize previous autoregressive step tracker
    best_rmse_015 = float('inf')
    NAR_passed_flag = False
    training_metrics = []

    prev_autoregressive_step = 0

    torch.autograd.set_detect_anomaly(True)

    # ------valid the model over `MAX_ERROR_SEQUENCE` steps
    val_loader, val_sampler = get_autoregressive_dataloader(flood_dataset_info=val_data,
                                                                num_tile=num_tile,
                                                                num_case=num_val_cases,
                                                                seq_length=NUM_PREV_STEP,
                                                                autoregressive_step=MAX_ERROR_SEQUENCE,
                                                                batch_size=BATCH_SIZE,
                                                                num_workers=LEARNING_NUM_WORKERS,
                                                                rank=rank,
                                                                world_size=world_size,
                                                                use_flood_LR=use_flood_LR,
                                                                loading_ratio=0.0)

    for epoch in range(EPOCHS):
        # Calculate NUM_ERROR_SEQUENCE
        NUM_ERROR_SEQUENCE = 1
        model_save_suffix = "NAR"
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Error Sequence: {NUM_ERROR_SEQUENCE}")

        current_autoregressive_step = NUM_ERROR_SEQUENCE - 1
        
        # Reinitialize datasets if autoregressive step changed
        if current_autoregressive_step != prev_autoregressive_step or epoch == 0:
            print(f"[Rank {rank}] Reinitializing datasets with autoregressive_step={current_autoregressive_step}")
            
            current_autoregressive_step = NUM_ERROR_SEQUENCE - 1
            
            train_loader, train_sampler = get_autoregressive_dataloader(flood_dataset_info=train_data,
                                                                        num_tile=num_tile,
                                                                        num_case=num_train_cases,
                                                                        seq_length=NUM_PREV_STEP,
                                                                        autoregressive_step=current_autoregressive_step,
                                                                        batch_size=BATCH_SIZE,
                                                                        num_workers=LEARNING_NUM_WORKERS,
                                                                        rank=rank,
                                                                        world_size=world_size,
                                                                        use_flood_LR=use_flood_LR,
                                                                        loading_ratio=0.0)

            # Update previous step tracker
            prev_autoregressive_step = current_autoregressive_step
    
        # Update weights for the last 10 epochs of each cycle
        # if is_last_epochs_of_cycle(epoch, nar_cycles, ar_cycles):
        ratio = get_progress_ratio_of_cycle(epoch, lr_cycle_list)
        loss_object.update_weights(ratio=ratio)
        
        train_sampler.set_epoch(epoch)  # Required for shuffling
        model_ddp.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # ************ [1] load data ************
            # if MTL_FLAG:
            #     rainfall_seq, flood_input_seq, flood_out_seq, flood_out_seq_mask, floodLR_seq, static_inputs, boundary_mask = batch
            # else:
            #     rainfall_seq, flood_input_seq, flood_out_seq, floodLR_seq, static_inputs, boundary_mask = batch
            rainfall_seq, flood_input_seq, flood_out_seq, flood_out_seq_mask, floodLR_seq, static_inputs, boundary_mask = batch

            rainfall_seq = rainfall_seq.to(device)      # [batch, (sequence_length + autoregressive_step), height, width]
            flood_input_seq = flood_input_seq.to(device)  # [batch, (sequence_length + autoregressive_step), height, width]
            targets = flood_out_seq.to(device)  # [batch, autoregressive_step + 1, height, width]
            if floodLR_seq is not None:
                floodLR_seq = floodLR_seq.to(device)
            if static_inputs is not None:
                static = static_inputs.to(device)   # [batch, 5, height, width]
            if boundary_mask is not None:
                boundary_mask = boundary_mask.to(device)
            
            if MTL_FLAG:
                targets_mask = flood_out_seq_mask.to(device)

            num_target_sample = targets.shape[1]

            # ---------Zero the optimizers gradients at the start of processing a new file
            optimizer_NAR.zero_grad()
            # ------get the input rainfall
            rainfall_inp_tmp = rainfall_seq[:, 0, :, :].unsqueeze(1)
            floodLR_inp_tmp  = floodLR_seq[:, 0, :, :].unsqueeze(1)

            # ------get the input flood
            # ---------solely use the current coarse flood input
            if flood_input_seq.shape[1] == 0:
                x_tensor = rainfall_inp_tmp
            # ---------use the current coarse flood input and the previous flood input
            else:
                x_tensor = torch.cat([rainfall_inp_tmp, flood_input_seq], dim=1)
            if static is not None and static.shape[1] > 0:
                if use_flood_LR:
                    aux_tensor = torch.cat([floodLR_inp_tmp, static], dim=1)
                    # aux_tensor = [aux_tensor[:, i, :, :].unsqueeze(1) for i in range(aux_tensor.shape[1])]
                else:
                    # aux_tensor = [static]
                    aux_tensor = static
            else:
                if use_flood_LR:
                    aux_tensor = floodLR_inp_tmp
                    # aux_tensor = [aux_tensor[:, i, :, :].unsqueeze(1) for i in range(aux_tensor.shape[1])]
                else:
                    aux_tensor = None

            # ------get the target flood and calculate the loss
            out = model_ddp(x_tensor, aux_tensor)
            if MTL_FLAG:
                flood_out_tmp, flood_out_mask_tmp = out
            else:
                flood_out_tmp = out

            if MTL_FLAG:
                loss = criterion(flood_out_tmp, targets, flood_out_mask_tmp, targets_mask, scale_depth, scale_mask, delta, boundary_mask)
            else:
                loss = criterion(flood_out_tmp, targets, delta, boundary_mask)
            
            loss.backward()
            optimizer_NAR.step()
            optimizer_NAR.zero_grad()

            num_batches += 1

            total_loss += loss.item()

        train_metric = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{EPOCHS}, Average train metric: {train_metric}")
        training_metrics.append(train_metric)
        epoch_metrics = {'epoch': epoch + 1, 'train_metric': train_metric}

        if MTL_FLAG and MTL_DYNAMIC_WEIGHT_FLAG:
            w_depth = 0.5 * torch.exp(-scale_depth)
            w_mask = 0.5 * torch.exp(-scale_mask)
            print(f"weight_depth: {w_depth}, weight_mask: {w_mask}")

        lr_scheduler_NAR.step()
        lr = optimizer_NAR.param_groups[0]['lr']
        print(f"[Epoch {epoch + 1}/{EPOCHS}] Learning rate: {lr}")

        # Validation phase
        model_ddp.eval()
        num_batches = 0
        mae_list = []
        validation_metric_record = {str(int(t*100)): {'RMSE':0.0, 
                                                      'IoU':0.0, 
                                                      'CSI':0.0,
                                                      'F2-Score':0.0,
                                                      'POD':0.0,
                                                      'FAR':0.0,
                                                      'Bias':0.0} for t in thresholds}
        
        with torch.no_grad():
            for batch in val_loader:
                rainfall_seq, flood_input_seq, flood_out_seq, flood_out_seq_mask, floodLR_seq, static_inputs, boundary_mask = batch
                rainfall_seq = rainfall_seq.to(device)
                flood_input_seq = flood_input_seq.to(device)
                targets = flood_out_seq.to(device)
                if floodLR_seq is not None:
                    floodLR_seq = floodLR_seq.to(device)
                if static_inputs is not None:
                    static = static_inputs.to(device)   # [batch, 5, height, width]
                if boundary_mask is not None:
                    boundary_mask = boundary_mask.to(device)
                
                if MTL_FLAG:
                    targets_mask = flood_out_seq_mask.to(device)

                num_target_sample = targets.shape[1]

                # Initialize a FIFO queue like in training
                flood_queue = queue.Queue(maxsize=NUM_PREV_STEP)
                for i in range(NUM_PREV_STEP):
                    flood_queue.put(flood_input_seq[:, i, :, :].unsqueeze(1))

                flood_pred_list = []
                flood_ref_list = []
                for sample_id in range(num_target_sample):
                    # Get rainfall input for current step
                    rainfall_inp_tmp = rainfall_seq[:, sample_id, :, :].unsqueeze(1)
                    floodLR_inp_tmp = floodLR_seq[:, sample_id, :, :].unsqueeze(1)
                            
                    # Get corresponding flood input
                    if flood_input_seq.shape[1] == 0:
                        x_tensor = rainfall_inp_tmp
                    else:
                        flood_inp_list_tmp = [flood_queue.queue[i] for i in range(0, NUM_PREV_STEP)]
                        flood_inp_tmp = torch.cat(flood_inp_list_tmp, dim=1)
                        x_tensor = torch.cat([rainfall_inp_tmp, flood_inp_tmp], dim=1)
                    
                    if static is not None and static.shape[1] > 0:
                        if use_flood_LR:
                            aux_tensor = torch.cat([floodLR_inp_tmp, static], dim=1)
                            # aux_tensor = [aux_tensor[:, i, :, :].unsqueeze(1) for i in range(aux_tensor.shape[1])]
                        else:
                            # aux_tensor = [static]
                            aux_tensor = static
                    else:
                        if use_flood_LR:
                            aux_tensor = floodLR_inp_tmp
                            # aux_tensor = [aux_tensor[:, i, :, :].unsqueeze(1) for i in range(aux_tensor.shape[1])]
                        else:
                            aux_tensor = None

                    # Predict flood output
                    if MTL_FLAG:
                        flood_out_tmp, flood_out_mask_tmp = model_ddp(x_tensor, aux_tensor)
                        # ------set the `flood_out_tmp` to zero where the mask (probability) is smaller than 0.5
                        flood_out_tmp = flood_out_tmp * (flood_out_mask_tmp >= 0.5).float()
                    else:
                        flood_out_tmp = model_ddp(x_tensor, aux_tensor)

                    # Update the flood queue for the next step
                    if flood_input_seq.shape[1] > 0:
                        flood_queue.get()
                        flood_queue.put(flood_out_tmp)

                    flood_pred_list.append(flood_out_tmp)
                    flood_ref_list.append(targets[:, sample_id, :, :].unsqueeze(1))
                
                # Compute metrics
                flood_pred = torch.cat(flood_pred_list, dim=1)
                flood_ref = torch.cat(flood_ref_list, dim=1)
                metrics = parallel_calculate_metricsV2(flood_pred, flood_ref, thresholds, fixed_mask=boundary_mask)
                
                for threshold in thresholds:
                    threshold_key = str(int(threshold * 100))
                    for metric in ["RMSE", "IoU", "CSI", "F2-Score", "POD", "FAR", "Bias"]:
                        validation_metric_record[threshold_key][metric] += metrics[threshold_key][metric]
                
                # Collect MAE values
                for threshold in thresholds:
                    threshold_key = str(int(threshold * 100))
                    mae_list.append(metrics[threshold_key]["MAE"])

                num_batches += 1           
                
        # ==== Synchronize metrics across all GPUs ====
        # Get total number of batches across all processes
        total_num_batches = torch.tensor(num_batches, dtype=torch.float32, device=device)
        dist.all_reduce(total_num_batches, op=dist.ReduceOp.SUM)
        total_num_batches = total_num_batches.item()
        
        for threshold in thresholds:
            threshold_key = str(int(threshold * 100))
            for metric in ["RMSE", "IoU", "CSI", "F2-Score", "POD", "FAR", "Bias"]:
                # Convert to tensor and perform all-reduce
                total = torch.tensor(validation_metric_record[threshold_key][metric], dtype=torch.float32, device=device)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                # Average across all GPUs
                validation_metric_record[threshold_key][metric] = total.item() / total_num_batches

        # ==== Synchronize MAE values ====
        # Gather MAE lists from all processes
        gathered_mae_lists = [None] * world_size
        dist.all_gather_object(gathered_mae_lists, mae_list)
        global_mae_list = np.concatenate(gathered_mae_lists)
        # Calculate 90th percentile MAE
        mae_q95 = np.quantile(global_mae_list, 0.95) if len(global_mae_list) > 0 else 1.0

        # delta = max(mae_q90, 1.0)
        delta = mae_q95

        # Calculate average metrics
        if rank == 0:
            # print(f"Updated delta to: {delta:.4f} (90th percentile MAE: {mae_q90:.4f})")
            print(f"95th percentile MAE: {mae_q95:.4f}")

            if epoch == NON_AUTOREGRESSIVE_EPOCHS - 1:
                NAR_passed_flag = True
                best_rmse_015 = float('inf')  # Reset best RMSE for 0.15 threshold after NAR phase
                print(f"[Rank {rank}] Non-autoregressive phase completed. Starting autoregressive training with {MAX_ERROR_SEQUENCE} steps.")
            
            # Populate epoch_metrics with synchronized values
            for threshold in thresholds:
                threshold_key = str(int(threshold * 100))
                epoch_metrics.update({
                    f'RMSE_{threshold_key}': round(validation_metric_record[threshold_key]['RMSE'], 4),
                    f'IoU_{threshold_key}': round(validation_metric_record[threshold_key]['IoU'], 4),
                    f'CSI_{threshold_key}': round(validation_metric_record[threshold_key]['CSI'], 4),
                    f'F2-Score_{threshold_key}': round(validation_metric_record[threshold_key]['F2-Score'], 4),
                    f'POD_{threshold_key}': round(validation_metric_record[threshold_key]['POD'], 4),
                    f'FAR_{threshold_key}': round(validation_metric_record[threshold_key]['FAR'], 4),
                    f'Bias_{threshold_key}': round(validation_metric_record[threshold_key]['Bias'], 4)
                })

            # Save best model based on 0.15 threshold RMSE
            threshold_015_key = 'RMSE_15'  # Corresponds to 0.15 threshold
            if threshold_015_key in epoch_metrics and epoch_metrics[threshold_015_key] < best_rmse_015:
                best_rmse_015 = epoch_metrics[threshold_015_key]
                torch.save(model_ddp.module.state_dict(), os.path.join(model_output_dir, f'modelBest_{case_identifier}_{model_save_suffix}.pth'))
                print(f"New best model saved for threshold 0.15 with RMSE: {best_rmse_015:.4f}")

            # Log metrics
            with open(metrics_file_path, 'a') as metrics_file:
                line = f"{epoch_metrics['epoch']},{epoch_metrics['train_metric']}"
                for threshold in thresholds:
                    scaled_threshold = int(threshold * 100)
                    line += f",{epoch_metrics[f'RMSE_{scaled_threshold}']},{epoch_metrics[f'IoU_{scaled_threshold}']},{epoch_metrics[f'CSI_{scaled_threshold}']},{epoch_metrics[f'F2-Score_{scaled_threshold}']},{epoch_metrics[f'POD_{scaled_threshold}']},{epoch_metrics[f'FAR_{scaled_threshold}']},{epoch_metrics[f'Bias_{scaled_threshold}']}"
                metrics_file.write(line + "\n")

    # ---save the checkpoint only for rank 0
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state': model_ddp.module.state_dict(),
            'optimizer_NAR': optimizer_NAR.state_dict(),
            'optimizer_AR': optimizer_AR.state_dict()
        }
        checkpoint_path = os.path.join(model_output_dir, f'modelEpoch{epoch}_{case_identifier}.pth')
        torch.save(checkpoint, checkpoint_path)

    # Clean up
    dist.destroy_process_group()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed UNet Training Script")
    parser.add_argument('--case_config_path', type=str, required=True, help="Path to the case configuration JSON file")
    args = parser.parse_args()
    case_config_path = args.case_config_path

    start_time = time.time()

    unet_train_mgpu_subproc(case_config_path)

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Total time taken for the script to execute: {elapsed_time:.2f} seconds")