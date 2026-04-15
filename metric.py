"""Loss functions and evaluation metrics for flood prediction.

Provides:
    - FloodPredLoss: Weighted Huber/MSE loss with depth-bin weighting and
      optional boundary mask. Supports adaptive multi-task weighting.
    - Classification metrics: IoU, CSI, F2-Score, POD, FAR at configurable
      flood depth thresholds.
    - Regression metrics: RMSE, MAE (masked by threshold).
    - parallel_calculate_metricsV2: Batch metric computation with fixed mask support.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


# ****************** weighted loss ******************
class FloodPredLoss(object):
    def __init__(self, cuda=False, weight_csv_path=None, adaptive_mask_loss_wight=True):
        self.cuda = cuda
        self.adaptive_mask_loss_wight = adaptive_mask_loss_wight
        self.bounds = None
        self.target_weight = None
        self.target_weight_cub = None  # New weight column
        self.target_count = None
        self.loss_power_min = 0.05
        self.loss_power_max = 0.75

        if weight_csv_path is not None and os.path.exists(weight_csv_path):
            weight_df = pd.read_csv(weight_csv_path)
            self.bounds = torch.tensor(weight_df["boundary"].values / 100.0, dtype=torch.float32)
            # self.target_weight = torch.tensor(weight_df["weight"].values, dtype=torch.float32)
            # self.target_weight_cub = torch.tensor(weight_df["weight_cub"].values, dtype=torch.float32)  # Load new weight column
            self.target_count = weight_df["Count"].values
            print("Weight boundaries:", self.bounds)
            print("Target count:", self.target_count)

            if self.cuda:
                self.bounds = self.bounds.cuda()
                # self.target_weight = self.target_weight.cuda()
                # self.target_weight_cub = self.target_weight_cub.cuda()
        else:
            self.bounds = None
            self.target_weight = None
            self.target_count = None

    # def update_weights(self, use_cub_weights):
    #     if use_cub_weights:
    #         self.target_weight = self.target_weight_cub
    #     else:
    #         # Reset to the original weights if needed
    #         self.target_weight = self.target_weight
    
    def update_weights(self, ratio):
        power_tmp = self.loss_power_max + ratio * (self.loss_power_min - self.loss_power_max)
        self.target_weight = self.get_weights(power_tmp)
        self.target_weight = self.target_weight.cuda() if self.cuda else self.target_weight

    def get_weights(self, power):
        count_tmp = np.where(self.target_count > 0, self.target_count, 1)
        freq_tmp = 1.0 / count_tmp
        weight_tmp = np.power(freq_tmp, power)  # here if power is larger, the target weight will be more biased (0.5 -> 0.05)
        weight_tmp = weight_tmp / np.sum(weight_tmp)
        print("Target weight:", weight_tmp, "\twith power:", power)
        weight_tmp = torch.tensor(weight_tmp, dtype=torch.float32)
        return weight_tmp

    def build_loss(self, loss_mode="MSE"):
        if loss_mode == "MSE":
            return nn.MSELoss(reduction="mean")
        elif loss_mode == "weighted_MSE":
            return self.weighted_MSELoss
        elif loss_mode == "weighted_MSE_withBinMask":
            return self.weighted_MSELoss_withBinMask
        elif loss_mode == "weighted_MSE_withBoundaryMask":
            return self.weighted_MSELoss_withBoundaryMask
        elif loss_mode == "weighted_HuberLoss_withBoundaryMask":
            return self.weighted_HuberLoss_withBoundaryMask
        elif loss_mode == "weighted_HuberLoss_withBinMask_withBoundaryMask":
            return self.weighted_HuberLoss_withBinMask_withBoundaryMask
        else:
            raise NotImplementedError

    def weighted_MSELoss(self, var_pred, var_ref):
        weight_id = torch.bucketize(var_ref, self.bounds, right=True) - 1
        weight_val = self.target_weight[weight_id]

        loss = (weight_val * (var_pred - var_ref)**2).mean()

        return loss

    def weighted_MSELoss_withBoundaryMask(self, var_pred, var_ref, mask_dta=None):
        weight_id = torch.bucketize(var_ref, self.bounds, right=True) - 1
        weight_val = self.target_weight[weight_id]
        
        if mask_dta is None:
            loss = (weight_val * (var_pred - var_ref)**2).mean()
        else:
            loss = (weight_val * (var_pred - var_ref)**2) * mask_dta
            loss = loss.sum() / mask_dta.sum()

        return loss

    def weighted_HuberLoss_withBoundaryMask(self, var_pred, var_ref, beta, mask_dta=None):
        weight_id = torch.bucketize(var_ref, self.bounds, right=True) - 1
        weight_val = self.target_weight[weight_id]

        loss = F.smooth_l1_loss(var_pred, var_ref, reduction='none', beta=beta)
        
        if mask_dta is None:
            loss = (weight_val * loss).mean()
        else:
            loss = (weight_val * loss) * mask_dta
            loss = loss.sum() / mask_dta.sum()

        return loss
    
    def weighted_HuberLoss_withBinMask_withBoundaryMask(self, var_pred_depth, var_ref_depth, var_pred_mask, var_ref_mask, scale_depth, scale_mask, beta, mask_dta=None):
        weight_id = torch.bucketize(var_ref_depth, self.bounds, right=True) - 1
        weight_val = self.target_weight[weight_id]
        
        loss_depth = F.smooth_l1_loss(var_pred_depth, var_ref_depth, reduction='none', beta=beta)
        loss_mask = F.binary_cross_entropy(var_pred_mask, var_ref_mask, reduction="none")
        
        if mask_dta is None:
            loss_depth = (weight_val * loss_depth).mean()
            loss_mask = (weight_val * loss_mask).mean()
        else:
            loss_depth = (weight_val * loss_depth) * mask_dta
            loss_depth = loss_depth.sum() / (mask_dta.sum() + 1e-8)

            loss_mask = (weight_val * loss_mask) * mask_dta
            loss_mask = loss_mask.sum() / (mask_dta.sum() + 1e-8)

        if self.adaptive_mask_loss_wight:
            weight_depth = 0.5 * torch.exp(-scale_depth)
            res_depth = 0.5 * scale_depth

            weight_mask = 0.5 * torch.exp(-scale_mask)
            res_mask = 0.5 * scale_mask

            loss = weight_depth * loss_depth + res_depth + weight_mask * loss_mask + res_mask
        else:
            loss = scale_depth * loss_depth + scale_mask * loss_mask

        # print("maxWeight: ", torch.max(weight_val).item())
        # print("Loss_depth: ", loss_depth.item(), "Loss_mask: ", loss_mask.item())

        return loss

    def weighted_MSELoss_withBinMask(self, var_pred_depth, var_ref_depth, var_pred_mask, var_ref_mask, scale_depth, scale_mask):
        weight_id = torch.bucketize(var_ref_depth, self.bounds, right=True) - 1
        weight_val = self.target_weight[weight_id]
        
        loss_depth = (weight_val * (var_pred_depth - var_ref_depth)**2).mean()

        loss_mask = F.binary_cross_entropy(var_pred_mask, var_ref_mask, reduction="mean", weight=weight_val)

        if self.adaptive_mask_loss_wight:
            weight_depth = 0.5 * torch.exp(-scale_depth)
            res_depth = 0.5 * scale_depth

            weight_mask = 0.5 * torch.exp(-scale_mask)
            res_mask = 0.5 * scale_mask

            loss = weight_depth * loss_depth + res_depth + weight_mask * loss_mask + res_mask
        else:

            loss = scale_depth * loss_depth + scale_mask * loss_mask

        # print("maxWeight: ", torch.max(weight_val).item())
        # print("Loss_depth: ", loss_depth.item(), "Loss_mask: ", loss_mask.item())

        return loss
# ****************** weighted loss ******************


# Metric calculation functions
def calculate_iou(predictions, targets, threshold):
    predictions_bin = predictions > threshold
    targets_bin = targets > threshold

    intersection = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
    union = (predictions_bin | targets_bin).float().sum(dim=(1, 2, 3))

    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou.mean().item()


def calculate_masked_rmse(predictions, targets, threshold):
    mask = targets > threshold
    if mask.sum() == 0:
        return 0.0
    
    mask = mask.unsqueeze(0).unsqueeze(1)
    masked_diff = (predictions - targets)[mask]
    mse = (masked_diff ** 2).mean()
    rmse = torch.sqrt(mse).item()
    return rmse


def calculate_masked_rmse_mae(predictions, targets, threshold):
    # Assume predictions and targets have shape [batch_size, sequence_length, channels, height, width]
    mask = targets > threshold  # This creates a mask with the same shape as targets

    if mask.sum() == 0:
        return 0.0, 0.0

    # Ensure mask is broadcasted correctly
    masked_diff = torch.masked_select(predictions - targets, mask)
    
    # Compute RMSE and MAE over the masked elements
    mse = (masked_diff ** 2).mean()
    rmse = torch.sqrt(mse).item()
    mae = masked_diff.abs().mean().item()

    return rmse, mae


def calculate_csi(predictions, targets, threshold):
    predictions_bin = predictions > threshold
    targets_bin = targets > threshold

    true_positive = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
    false_negative = (~predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
    false_positive = (predictions_bin & ~targets_bin).float().sum(dim=(1, 2, 3))

    denominator = true_positive + false_negative + false_positive
    csi = true_positive / (denominator + 1e-6)  # Avoid division by zero
    return csi.mean().item()


def calculate_pod(predictions, targets, threshold):
    predictions_bin = predictions > threshold
    targets_bin = targets > threshold

    true_positive = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
    false_negative = (~predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))

    denominator = true_positive + false_negative
    pod = true_positive / (denominator + 1e-6)  # Avoid division by zero
    return pod.mean().item()


def calculate_far(predictions, targets, threshold):
    predictions_bin = predictions > threshold
    targets_bin = targets > threshold

    false_positive = (predictions_bin & ~targets_bin).float().sum(dim=(1, 2, 3))
    true_positive = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))

    denominator = true_positive + false_positive
    far = false_positive / (denominator + 1e-6)  # Avoid division by zero
    return far.mean().item()


def calculate_f2_score(predictions, targets, threshold):
    predictions_bin = predictions > threshold
    targets_bin = targets > threshold

    true_positive = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
    false_positive = (predictions_bin & ~targets_bin).float().sum(dim=(1, 2, 3))
    false_negative = (~predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))

    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)

    f2_score = (5 * precision * recall) / (4 * precision + recall + 1e-6)  # Weighted F2-score
    return f2_score.mean().item()


def calculate_bias(predictions, targets):
    # Bias is calculated as the mean difference between predictions and targets
    bias = (predictions.mean() - targets.mean()).item()
    return bias


# Function to calculate all metrics in parallel
def parallel_calculate_metrics(predictions, targets, thresholds):
    metrics = {}

    def compute_for_threshold(threshold):
        rmse, mae = calculate_masked_rmse_mae(predictions, targets, threshold)
        iou = calculate_iou(predictions, targets, threshold)
        csi = calculate_csi(predictions, targets, threshold)
        f2_score = calculate_f2_score(predictions, targets, threshold)
        pod = calculate_pod(predictions, targets, threshold)
        far = calculate_far(predictions, targets, threshold)
        bias = calculate_bias(predictions, targets)

        return threshold, {
            "RMSE": rmse,
            "MAE": mae,
            "IoU": iou,
            "CSI": csi,
            "F2-Score": f2_score,
            "POD": pod,
            "FAR": far,
            "Bias": bias,
        }

    # Use ThreadPoolExecutor to parallelize the computation across thresholds
    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_for_threshold, thresholds)

    for threshold, metrics_for_threshold in results:
        metrics[str(int(threshold * 100))] = metrics_for_threshold

    return metrics


def parallel_calculate_metricsV2(predictions, targets, thresholds, fixed_mask=None):
    metrics = {}

    # def compute_for_threshold(threshold):
    #     rmse, mae = calculate_masked_rmse_mae(predictions, targets, threshold)
    #     iou = calculate_iou(predictions, targets, threshold)
    #     csi = calculate_csi(predictions, targets, threshold)
    #     f2_score = calculate_f2_score(predictions, targets, threshold)
    #     pod = calculate_pod(predictions, targets, threshold)
    #     far = calculate_far(predictions, targets, threshold)
    #     bias = calculate_bias(predictions, targets)

    #     return threshold, {
    #         "RMSE": rmse,
    #         "MAE": mae,
    #         "IoU": iou,
    #         "CSI": csi,
    #         "F2-Score": f2_score,
    #         "POD": pod,
    #         "FAR": far,
    #         "Bias": bias,
    #     }

    for thres in thresholds:
        # ****** RMSE and MAE ****** #
        # Assume predictions and targets have shape [batch_size, sequence_length, channels, height, width]
        mask = targets > thres  # This creates a mask with the same shape as targets

        # ------get the intersection of the mask and the fixed mask
        if fixed_mask is not None:
            mask = mask & fixed_mask.bool()

        if mask.sum() == 0:
            rmse = 0.0
            mae = 0.0
        else:
            # Ensure mask is broadcasted correctly
            masked_diff = torch.masked_select(predictions - targets, mask)
            
            # Compute RMSE and MAE over the masked elements
            mse = (masked_diff ** 2).mean()
            rmse = torch.sqrt(mse).item()
            mae = masked_diff.abs().mean().item()

        # ****** binary-based metrics ****** #
        predictions_bin = predictions > thres
        targets_bin = targets > thres
        #print("Threshold: ", thres)
        # ------ check the number of unmasked target pixels
        #n_target = targets_bin.sum(dim=(1, 2, 3)).float()
        #print("Number of target pixels: ", n_target)
        # ------ check the maximum value of the target
        #print("Max target value: ", targets.max().item())
        #n_pred = predictions_bin.sum(dim=(1, 2, 3)).float()
        #print("Number of predicted pixels: ", n_pred)
        #print("Max predicted value: ", predictions.max().item())

        true_positive = (predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
        false_negative = (~predictions_bin & targets_bin).float().sum(dim=(1, 2, 3))
        false_positive = (predictions_bin & ~targets_bin).float().sum(dim=(1, 2, 3))

        union = (predictions_bin | targets_bin).float().sum(dim=(1, 2, 3))

        # ------ IoU
        iou = true_positive / (union + 1e-6)  # Avoid division by zero
        iou = iou.mean().item()
        # ------ CSI
        csi = true_positive / (true_positive + false_negative + false_positive + 1e-6)  # Avoid division by zero
        csi = csi.mean().item()
        # ------ F2-Score
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        f2_score = (5 * precision * recall) / (4 * precision + recall + 1e-6)  # Weighted F2-score
        f2_score = f2_score.mean().item()
        # ------ POD
        pod = true_positive / (true_positive + false_negative + 1e-6)  # Avoid division by zero
        pod = pod.mean().item()
        # ------ FAR
        far = false_positive / (true_positive + false_positive + 1e-6)  # Avoid division by zero
        far = far.mean().item()
        # ------ Bias
        bias = (predictions.mean() - targets.mean()).item()

        metrics[str(int(thres * 100))] = {
            "RMSE": rmse,
            "MAE": mae,
            "IoU": round(iou, 5),
            "CSI": round(csi, 5),
            "F2-Score": round(f2_score, 5),
            "POD": round(pod, 5),
            "FAR": round(far, 5),
            "Bias": round(bias, 5),
        }

    return metrics


if __name__ == "__main__":
    pass