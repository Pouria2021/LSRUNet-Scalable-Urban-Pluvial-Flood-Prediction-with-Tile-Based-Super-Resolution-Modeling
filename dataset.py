"""
PyTorch Dataset Classes for Flood Super-Resolution Training.

This module provides dataset classes for loading flood prediction training data,
including:
    - FloodPredictionDataset: Original dataset with tile/case hierarchical structure
    - FloodPredictionDatasetV2: Optimized flat dataset for faster training

Data Structure
--------------
Each training sample consists of:
    - Rainfall: Precipitation at current timestep (input)
    - Flood HR: High-resolution flood depth at target timestep (output)
    - Flood LR: Low-resolution flood depth at target timestep (auxiliary input)
    - Static features: DEM, roughness, runoff coefficient (auxiliary input)
    - Mask: Boundary mask for valid prediction area

Key Functions
-------------
    - preload_data: Load and organize training data by tile/case structure
    - preload_dataV2: Optimized preloader for flat dataset structure
    - get_autoregressive_dataloader: Create distributed dataloader

Notes
-----
    - Data is stored as NumPy arrays (.npy) for fast I/O
    - Original GeoTIFF files are converted by utils_tif_to_npy.py
    - Supports distributed training with DistributedSampler
"""

import os
import math
import gc
from functools import partial
import multiprocessing as mp

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler


# In preload_data, assign cases based on rank/world_size
def load_geotiff_dynamic(filepath: str) -> np.ndarray:
    """
    Load a single-band GeoTIFF as float16 array.

    Used for loading time-series flood depth data where memory efficiency
    is important. NaN values are replaced with zeros.

    Parameters
    ----------
    filepath : str
        Path to the GeoTIFF file.

    Returns
    -------
    np.ndarray
        2D array with shape (H, W), dtype float16.
    """
    """Load all bands (time steps) from a GeoTIFF."""
    with rasterio.open(filepath) as src:
        # data = src.read(1).astype(np.float32)  # Returns (H, W)
        data = src.read(1).astype(np.float16)  # Returns (H, W)
        data = np.nan_to_num(data)  # Replace NaNs with 0
    return data


def load_geotiff_static(filepath: str) -> np.ndarray:
    """
    Load a single-band GeoTIFF as float32 array.

    Used for loading static terrain features (DEM, roughness, etc.)
    where higher precision is preferred.

    Parameters
    ----------
    filepath : str
        Path to the GeoTIFF file.

    Returns
    -------
    np.ndarray
        2D array with shape (H, W), dtype float32.
    """
    """Load only the first band from a single-band GeoTIFF."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)  # Returns (H, W)
    return data


def convert_geotiff_to_numpy(inp_tif_pathlist: str, out_npy_dir: str):
    for inp_tif_path in inp_tif_pathlist:
        with rasterio.open(inp_tif_path) as src:
            data = src.read(1).astype(np.float32)  # Returns (H, W)
            data = np.nan_to_num(data)  # Replace NaNs with 0
        
        inp_tif_basename = os.path.basename(inp_tif_path)
        npy_filename = inp_tif_basename.replace('.tif', '.npy')
        npy_filepath = os.path.join(out_npy_dir, npy_filename)
        np.save(npy_filepath, data)


def batch_convert_geotiff_to_numpy(inp_tif_dir: str, out_npy_dir: str, num_cpu: int = 4):
    inp_tif_list = [os.path.join(inp_tif_dir, f) for f in os.listdir(inp_tif_dir) if f.endswith('.tif') and not f.startswith('.')]
    print(f"Found {len(inp_tif_list)} GeoTIFF files to convert.")

    if not os.path.exists(out_npy_dir):
        os.makedirs(out_npy_dir)
        print(f"Created output directory: {out_npy_dir}")
    
    inp_tif_list = np.array_split(inp_tif_list, num_cpu)

    pool = mp.Pool(processes=num_cpu)
    args_list = [(inp_tif_sublist, out_npy_dir) for inp_tif_sublist in inp_tif_list]
    pool.starmap(convert_geotiff_to_numpy, args_list)

    pool.close()
    pool.join()


class FloodPredictionDataset(Dataset):
    """
    PyTorch Dataset for flood super-resolution with hierarchical tile/case structure.

    This dataset organizes training samples by tiles (spatial regions) and cases
    (simulation scenarios), supporting both autoregressive and non-autoregressive
    training modes.

    Parameters
    ----------
    flood_dataset_info : dict
        Nested dictionary containing data paths organized by tile and case.
        Structure: {TileID: {caseID: {rainfall: [...], flood: [...], flood_LR: [...]}}}
    num_tile : int
        Total number of tiles in the dataset.
    num_case : int
        Total number of cases (simulation scenarios).
    sequence_length : int
        Number of previous flood frames to use as input (typically 0 or 1).
    autoregressive_step : int, optional
        Number of autoregressive forward steps. Default is 0 (non-autoregressive).
    use_flood_LR : bool, optional
        Whether to include low-resolution flood as auxiliary input. Default is True.

    Attributes
    ----------
    num_sample : int
        Total number of training samples.
    num_sample_interval : list
        Cumulative sample counts for indexing.

    Sample Output
    -------------
    __getitem__ returns:
        - rainfall_seq_tensor: (T, H, W) - Rainfall sequence
        - flood_input_seq_tensor: (S, H, W) - Previous flood frames
        - flood_out_seq_tensor: (T, H, W) - Target HR flood frames
        - flood_LR_seq_tensor: (T, H, W) or None - LR flood frames
        - static_input_tensor: (C, H, W) - Static features
        - mask_input_tensor: (1, H, W) - Boundary mask

    Notes
    -----
    - Supports loading data from memory (if preloaded) or disk (numpy files)
    - Static features and masks are preloaded during initialization
    """
    def __init__(self, flood_dataset_info, num_tile, num_case, sequence_length, autoregressive_step=0, use_flood_LR=True):
        self.flood_dataset_info = flood_dataset_info
        self.num_tile = num_tile
        self.num_case = num_case
        
        self.input_flood_sequence_length = sequence_length
        self.input_rainfall_sequence_length = 1
        self.autoregressive_step = autoregressive_step
        self.use_flood_LR = use_flood_LR

        num_sample = 0
        num_sample_interval = []
        for tile_ID in range(self.num_tile):
            tile_info = self.flood_dataset_info[f"Tile{tile_ID}"]
            for case_ID in range(self.num_case):
                cas = f"case{case_ID}"
                case_info = tile_info[cas]
                num_sample_interval.append(num_sample)
                num_sample += len(case_info['rainfall']) - self.autoregressive_step - max(self.input_flood_sequence_length, self.input_rainfall_sequence_length)

        self.num_sample = num_sample
        self.num_sample_interval = num_sample_interval
        print(f"Total number of samples: {self.num_sample}")
        print("Sample intervals:", self.num_sample_interval)

        # ---preload the static inputs (constant for the tile)
        for tile_ID in range(self.num_tile):
            if self.flood_dataset_info[f"Tile{tile_ID}"]["static"] is not None:
                if not isinstance(self.flood_dataset_info[f"Tile{tile_ID}"]["static"], torch.Tensor):
                    self.flood_dataset_info[f"Tile{tile_ID}"]["static"] = torch.tensor(self.flood_dataset_info[f"Tile{tile_ID}"]["static"], dtype=torch.float32)

        # ---preload the mask inputs (constant for the tile)
        for tile_ID in range(self.num_tile):
            if self.flood_dataset_info[f"Tile{tile_ID}"]["mask"] is not None:
                # mask_dta = torch.from_numpy(self.flood_dataset_info[f"Tile{tile_ID}"]["mask"])
                if not isinstance(self.flood_dataset_info[f"Tile{tile_ID}"]["mask"], torch.Tensor):
                    mask_dta = torch.tensor(np.array(self.flood_dataset_info[f"Tile{tile_ID}"]["mask"]))
                    mask_dta = mask_dta.unsqueeze(0)  # [1, height, width]
                    self.flood_dataset_info[f"Tile{tile_ID}"]["mask"] = mask_dta.float()  # Convert to float32

    # ---[1] single-process loading
    def load_dynamic_data_memory(self, loading_ratio, num_cpu):
        # ---preload the dynamic inputs (rainfall, flood, flood_LR)
        if loading_ratio >= 1.0:
            for tile_ID in range(self.num_tile):
                # ------check the dynamic file information
                case_list = [cas for cas in self.flood_dataset_info[f"Tile{tile_ID}"].keys() if "case" in cas]
                for cas in case_list:
                    if not isinstance(self.flood_dataset_info[f"Tile{tile_ID}"][cas]["rainfall"], np.ndarray):
                        num_ts = len(self.flood_dataset_info[f"Tile{tile_ID}"][cas]["rainfall"])
                        self.flood_dataset_info[f"Tile{tile_ID}"][cas]["rainfall"] = np.array([load_geotiff_dynamic(self.flood_dataset_info[f"Tile{tile_ID}"][cas]["rainfall"][tsID]) for tsID in range(0, num_ts)])
                        self.flood_dataset_info[f"Tile{tile_ID}"][cas]["flood"] = np.array([load_geotiff_dynamic(self.flood_dataset_info[f"Tile{tile_ID}"][cas]["flood"][tsID]) for tsID in range(0, num_ts)])

                        if self.use_flood_LR:
                            self.flood_dataset_info[f"Tile{tile_ID}"][cas]["flood_LR"] = np.array([load_geotiff_dynamic(self.flood_dataset_info[f"Tile{tile_ID}"][cas]["flood_LR"][tsID]) for tsID in range(0, num_ts)])
                        print(f"Tile {tile_ID}, {cas} loaded with {num_ts} time steps of dynamic data.")
                   
    def __len__(self):
        return self.num_sample

    def update_autoregressive_step(self, autoregressive_step):
        self.autoregressive_step = autoregressive_step

        # ------update the number of samples based on the new autoregressive step
        num_sample = 0
        num_sample_interval = []
        for tile_ID in range(self.num_tile):
            tile_info = self.flood_dataset_info[f"Tile{tile_ID}"]
            for case_ID in range(self.num_case):
                cas = f"case{case_ID}"
                case_info = tile_info[cas]
                num_sample_interval.append(num_sample)
                num_sample += len(case_info['rainfall']) - self.autoregressive_step - max(self.input_flood_sequence_length, self.input_rainfall_sequence_length)

        self.num_sample = num_sample
        self.num_sample_interval = num_sample_interval
        print(f"Total number of samples: {self.num_sample}")
        print("Sample intervals:", self.num_sample_interval)

    def __getitem__(self, idx):
        """
        Returns a sample where rainfall and flood sequences are temporally consecutive
        according to the autoregressive_step parameter.
        """

        idx_tmp = np.searchsorted(self.num_sample_interval, idx, side='right') - 1
        tile_ID = idx_tmp // self.num_case
        case_ID = idx_tmp % self.num_case

        idx_ini = idx
        idx = idx - self.num_sample_interval[idx_tmp]

        timeID_target_start = idx + max(self.input_flood_sequence_length, self.input_rainfall_sequence_length)   # index of the first target sample
        timeID_target_end = idx + max(self.input_flood_sequence_length, self.input_rainfall_sequence_length) + self.autoregressive_step # index of the last target sample

        timeID_floodHR_prev_start = timeID_target_start - self.input_flood_sequence_length
        timeID_floodHR_prev_end = timeID_target_start - 1 + self.autoregressive_step

        timeID_rainfall_start = timeID_target_start - self.input_rainfall_sequence_length
        timeID_rainfall_end = timeID_target_start - 1 + self.autoregressive_step

        sample_info = self.flood_dataset_info[f"Tile{tile_ID}"][f"case{case_ID}"]

        if timeID_rainfall_start >= len(sample_info["rainfall"]):
            print(idx, idx_ini, idx_tmp)
            print(timeID_floodHR_prev_start, timeID_floodHR_prev_end)
            print(timeID_rainfall_start, timeID_rainfall_end)
            print(self.autoregressive_step)
            print(self.num_sample_interval[idx_tmp])

        if isinstance(sample_info["rainfall"][timeID_target_start], str):
            rainfall_seq = np.array([np.load(sample_info["rainfall"][tsID]) for tsID in range(timeID_rainfall_start, timeID_rainfall_end + 1)])
            flood_input_seq = np.array([np.load(sample_info["flood"][tsID]) for tsID in range(timeID_floodHR_prev_start, timeID_floodHR_prev_end + 1)])
            flood_out_seq = np.array([np.load(sample_info["flood"][tsID]) for tsID in range(timeID_target_start, timeID_target_end + 1)])
        else:
            rainfall_seq = np.array([sample_info["rainfall"][tsID] for tsID in range(timeID_rainfall_start, timeID_rainfall_end + 1)])
            flood_input_seq = np.array([sample_info["flood"][tsID] for tsID in range(timeID_floodHR_prev_start, timeID_floodHR_prev_end + 1)])
            flood_out_seq = np.array([sample_info["flood"][tsID] for tsID in range(timeID_target_start, timeID_target_end + 1)])

        # Convert to tensors
        rainfall_seq_tensor = torch.tensor(rainfall_seq, dtype=torch.float32)
        flood_input_seq_tensor = torch.tensor(flood_input_seq, dtype=torch.float32)
        flood_out_seq_tensor = torch.tensor(flood_out_seq, dtype=torch.float32)  

        if self.use_flood_LR:
            if isinstance(sample_info["flood_LR"][timeID_target_start], str):
                # flood_LR_seq = np.array([load_geotiff_dynamic(sample_info["flood_LR"][tsID]) for tsID in range(timeID_prediction_start, timeID_prediction_end)])
                flood_LR_seq = np.array([np.load(sample_info["flood_LR"][tsID]) for tsID in range(timeID_target_start, timeID_target_end + 1)])
            else:
                flood_LR_seq = np.array([sample_info["flood_LR"][tsID] for tsID in range(timeID_target_start, timeID_target_end + 1)])
            flood_LR_seq_tensor = torch.tensor(flood_LR_seq, dtype=torch.float32)  # Flood100m tensor
        else:
            flood_LR_seq_tensor = None

        # Static inputs (constant for the tile)
        static_input_tensor = self.flood_dataset_info[f"Tile{tile_ID}"]["static"]
        if static_input_tensor is None:
            static_input_tensor = torch.tensor([])
        mask_input_tensor = self.flood_dataset_info[f"Tile{tile_ID}"]["mask"]
        
        return rainfall_seq_tensor, flood_input_seq_tensor, flood_out_seq_tensor, flood_LR_seq_tensor, static_input_tensor, mask_input_tensor


class FloodPredictionDatasetV2(Dataset):
    """
    Flat fast-path dataset (no tile/case decoding at runtime).

    Returns:
      rainfall_seq_tensor:      [1, ...]  (rainfall at t)
      flood_input_seq_tensor:   [0, ...]  (empty, because input_flood_sequence_length=0)
      flood_out_seq_tensor:     [1, ...]  (HR flood at t+1)
      flood_LR_seq_tensor:      [1, ...]  (LR flood at t+1)
      static_input_tensor:      [C, H, W] or empty
      mask_input_tensor:        [1, H, W] or empty
    """

    def __init__(self, flood_dataset_info):
        self.info = flood_dataset_info

        self.rainfall = self.info["rainfall"]
        self.flood_HR = self.info["flood_HR"]
        self.flood_LR = self.info["flood_LR"]

        self.tile_idx = self.info.get("tile_idx", None)
        self.static_by_tile = self.info.get("static_by_tile", None)
        self.mask_by_tile = self.info.get("mask_by_tile", None)

        n = len(self.rainfall)
        if len(self.flood_HR) != n or len(self.flood_LR) != n:
            raise ValueError("rainfall/flood_out/flood_LR must have the same length.")

        if self.tile_idx is not None:
            if len(self.tile_idx) != n:
                raise ValueError("tile_idx must have the same length as rainfall.")
            if self.static_by_tile is None or self.mask_by_tile is None:
                raise ValueError("static_by_tile and mask_by_tile must exist if tile_idx is provided.")

        # Fixed fast-path settings (kept as attributes for clarity/debugging)
        self.input_flood_sequence_length = 0
        self.input_rainfall_sequence_length = 1
        self.autoregressive_step = 0
        self.use_flood_LR = True

        print(f"Total number of samples: {n}")

    def __len__(self):
        return len(self.rainfall)

    @staticmethod
    def _to_float_tensor(x):
        """
        Accepts:
          - numpy array
          - torch tensor
          - filepath (str / PathLike) to .npy
        Returns float32 torch tensor.
        """
        if isinstance(x, torch.Tensor):
            return x.float() if x.dtype != torch.float32 else x

        if isinstance(x, (str, os.PathLike)):
            arr = np.load(x)
        else:
            arr = x

        # Ensure float32 without unnecessary copies
        if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        return torch.as_tensor(arr, dtype=torch.float32)

    def __getitem__(self, idx):
        """
        Retrieve a single training sample by index.

        This method returns data for the flat fast-path dataset structure where each sample
        consists of rainfall input and corresponding flood outputs at the next timestep.

        Parameters
        ----------
        idx : int
            Sample index in the range [0, len(dataset)).

        Returns
        -------
        tuple
            A tuple containing:
            - rainfall_frame: torch.Tensor of shape [1, H, W]
                Rainfall at timestep t (input).
            - flood_input_seq_tensor: torch.Tensor of shape [0, H, W]
                Empty tensor (input_flood_sequence_length=0 for V2).
            - flood_HR_frame: torch.Tensor of shape [1, H, W]
                High-resolution flood depth at timestep t+1 (target).
            - flood_HR_binary_frame: torch.Tensor of shape [1, H, W]
                Binary mask indicating flood presence (depth > 0.15m).
            - flood_LR_frame: torch.Tensor of shape [1, H, W]
                Low-resolution flood depth at timestep t+1 (auxiliary input).
            - static_input_tensor: torch.Tensor of shape [C, H, W] or [0]
                Static features (DEM, roughness, runoff coefficient) for this tile.
                Retrieved from static_by_tile using tile_idx.
            - mask_input_tensor: torch.Tensor of shape [1, H, W] or [0]
                Boundary mask for valid prediction area.
                Retrieved from mask_by_tile using tile_idx.

        Notes
        -----
        The tile_idx value maps each sample to its corresponding tile, which determines
        which static features and boundary mask to use. This is essential for multi-tile
        datasets where different geographic regions have different DEM, roughness,
        and boundary characteristics.

        The sample mapping follows:
        - idx -> tile_idx[idx] -> static_by_tile[tile_idx], mask_by_tile[tile_idx]
        """
        DEPTH_THRESHOLD = 0.15

        # Dynamic inputs/targets
        rainfall_frame = self._to_float_tensor(self.rainfall[idx]).unsqueeze(0)    # [1, ...]
        flood_HR_frame = self._to_float_tensor(self.flood_HR[idx]).unsqueeze(0) # [1, ...]
        flood_LR_frame = self._to_float_tensor(self.flood_LR[idx]).unsqueeze(0)   # [1, ...]

        # Empty flood history input: [0, ...]
        flood_input_seq_tensor = flood_HR_frame.new_empty((0,) + flood_HR_frame.shape[1:])
        flood_HR_binary_frame = (flood_HR_frame > DEPTH_THRESHOLD).float()

        # Static/mask lookup by tile index (or empty tensors if not provided)
        if self.tile_idx is None:
            static_input_tensor = torch.empty(0, dtype=torch.float32)
            mask_input_tensor = torch.empty(0, dtype=torch.float32)
        else:
            tpos = self.tile_idx[idx]
            static_input_tensor = self.static_by_tile[tpos]
            mask_input_tensor = self.mask_by_tile[tpos]

        return rainfall_frame, flood_input_seq_tensor, flood_HR_frame, flood_HR_binary_frame, flood_LR_frame, static_input_tensor, mask_input_tensor


def preload_data(case_namelist, data_dir, rank, world_size, tile_ID_list, aux_file_info=None, mask_file_prefix=None, flood_LR_dir=None, loading_ratio=1.0, tile_kw="tile", flood_LR_kw="FloodLR", flood_HR_kw="Flood", rainfall_kw="Rainfall", timestep_kw="TS"):
    dataset_info = {}

    print("All cases:", case_namelist, "world_size:", world_size, "rank:", rank)
    # assigned_cases = case_namelist[rank::world_size]
    assigned_cases = case_namelist
    print(f"Rank {rank} assigned cases: {assigned_cases}")

    if loading_ratio < 1.0:
        loading_flag = False
    else:
        loading_flag = True
        print("Loading all data into memory.")

    for tile_ID in tile_ID_list:      # for the current implementation, we assume tile_ID starts from 1
        tile_ID_used = int(tile_ID - 1)
        dataset_info[f"Tile{tile_ID_used}"] = {}

        # ------check the auxiliary file information
        if aux_file_info is not None:
            # Load static inputs from GeoTIFFs
            aux_dta_list = []
            for k in aux_file_info:
                aux_dta = load_geotiff_static(f"{aux_file_info[k]}_{tile_kw}{tile_ID}.tif")
                aux_dta_list.append(aux_dta)

            if len(aux_dta_list) > 0:
                static_dta = np.stack(aux_dta_list, axis=0)
            else:
                static_dta = None
        else:
            static_dta = None
        
        dataset_info[f"Tile{tile_ID_used}"]["static"] = static_dta

        # ------check the mask file information
        if mask_file_prefix is not None:
            mask_file = f"{mask_file_prefix}_{tile_kw}{tile_ID}.npy"
            if os.path.exists(mask_file):
                mask_dta = np.load(mask_file).astype(np.float32)
            else:
                mask_dta = None
        else:
            mask_dta = None
        
        dataset_info[f"Tile{tile_ID_used}"]["mask"] = mask_dta
        
        # ------check the dynamic file information
        case_counter = 0
        for cas in assigned_cases:
            cas_used = f"case{case_counter}"
            dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"] = {}

            dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["rainfall"] = []
            dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["flood"] = []
            dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["flood_LR"] = []

            timestep_list = [f for f in os.listdir(data_dir) if f.startswith(f"{cas}_{rainfall_kw}_{timestep_kw}") and f.endswith(f"_{tile_kw}{tile_ID}.npy")]
            timestep_list = sorted([int(f.split(f"{timestep_kw}")[1].split('_')[0]) for f in timestep_list])
            print("Timestep list for case", cas, "tile", tile_ID, ":", timestep_list)

            for ts in timestep_list:
                rf = os.path.join(data_dir, f"{cas}_{rainfall_kw}_{timestep_kw}{int(ts)}_{tile_kw}{tile_ID}.npy")
                ff = os.path.join(data_dir, f"{cas}_{flood_HR_kw}_{timestep_kw}{int(ts)}_{tile_kw}{tile_ID}.npy")

                if not os.path.exists(rf) or not os.path.exists(ff):
                    print(f"Missing files: {rf}, {ff}")
                    continue

                if loading_flag:
                    rf_dta = load_geotiff_dynamic(rf)
                    ff_dta = load_geotiff_dynamic(ff)
                else:
                    ff_dta = ff
                    rf_dta = rf
                
                dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["rainfall"].append(rf_dta)
                dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["flood"].append(ff_dta)

                if flood_LR_dir is not None and os.path.exists(flood_LR_dir):
                    ff_LR = os.path.join(flood_LR_dir, f"{cas}_{flood_LR_kw}_{timestep_kw}{int(ts)}_{tile_kw}{tile_ID}.npy")
                    if loading_flag:
                        ff_LR_dta = load_geotiff_dynamic(ff_LR)
                    else:
                        ff_LR_dta = ff_LR
                    dataset_info[f"Tile{tile_ID_used}"][f"{cas_used}"]["flood_LR"].append(ff_LR_dta)
            case_counter += 1
                           
    return dataset_info


def _scan_rainfall_timesteps_for_tile(
    data_dir,
    tile_ID,
    rainfall_kw="Rainfall",
    timestep_kw="TS",
    tile_kw="tile",
    case_filter=None,
):
    """
    Scan data directory for rainfall files belonging to a specific tile.

    This function scans the data directory (including subdirectories) to find
    all rainfall files matching the pattern {case}_{Rainfall}_TS{ts}_tile{tile_ID}.npy
    and returns a mapping of case names to their sorted timestep lists.

    Parameters
    ----------
    data_dir : str
        Root directory containing tile subdirectories with rainfall files.
        Expected structure: {data_dir}/{tile_ID}/{case}_{rainfall_kw}_{timestep_kw}{ts}_{tile_kw}{tile_ID}.npy
    tile_ID : int
        Tile identifier to scan for (1-based).
    rainfall_kw : str, optional
        Keyword identifying rainfall files (default: "Rainfall").
    timestep_kw : str, optional
        Keyword preceding timestep number (default: "TS").
    tile_kw : str, optional
        Keyword identifying tile ID in filename (default: "tile").
    case_filter : list or set, optional
        If provided, only include cases in this collection.

    Returns
    -------
    dict
        Mapping of case names to sorted lists of available timesteps.
        Structure: {case_name: [timestep1, timestep2, ...]}

    Notes
    -----
    - This function supports the flat dataset structure where files are organized
      in subdirectories named by tile ID.
    - The function scans all subdirectories in data_dir, looking for files
      matching the expected pattern.
    - Timesteps are sorted in ascending order for each case.

    Example
    -------
    >>> ts_by_case = _scan_rainfall_timesteps_for_tile(
    ...     data_dir="/path/to/train_cases-HR_Tile336_npy",
    ...     tile_ID=5,
    ...     rainfall_kw="Rainfall",
    ...     timestep_kw="TS",
    ...     tile_kw="Tile"
    ... )
    >>> print(ts_by_case)
    {'0': [600, 1200, 1800, 2400], '3': [600, 1200]}
    """
    suffix = f"_{tile_kw}{tile_ID}.npy"
    marker = f"_{rainfall_kw}_{timestep_kw}"

    if case_filter is not None:
        case_filter = set(case_filter)

    ts_by_case = {}
    data_dir_subtile = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Scanning for rainfall files in {data_dir}, found subtile directories: {data_dir_subtile}")

    for subtile in data_dir_subtile:
        data_dir_tmp = os.path.join(data_dir, subtile)
        for ent in os.scandir(data_dir_tmp):
            if not ent.is_file():
                continue
            name = ent.name
            if not name.endswith(suffix):
                continue
            if marker not in name:
                continue

            # split "{case}" + marker + "{ts}_tile{tile_ID}.npy"
            cas, rest = name.split(marker, 1)
            if case_filter is not None and cas not in case_filter:
                continue

            # rest begins with digits of ts (e.g. "12_tile3.npy" after split)
            ts_str = rest.split("_", 1)[0]
            try:
                ts = int(ts_str)
            except ValueError:
                continue

            ts_by_case.setdefault(cas, []).append(ts)
            # # ------`ts_by_case.setdefault(cas, []).append(ts)` equal to:
            # if cas not in ts_by_case:
            #     ts_by_case[cas] = []
            # ts_by_case[cas].append(ts)

    for cas in ts_by_case:
        ts_by_case[cas].sort()

    return ts_by_case


def _tile_worker_build_pairs(
    tile_ID_sublist,
    assigned_cases,
    data_dir,
    flood_LR_dir,
    tilepos_by_tile_used,
    loading_flag,
    aux_kwargs,
):
    """
    Build flat paired samples for a sublist of tiles (worker function for parallel processing).

    This function processes a subset of tiles and creates training sample pairs by
    matching rainfall inputs with corresponding flood HR/LR outputs. Each pair represents
    a transition from timestep t to timestep t+1.

    Parameters
    ----------
    tile_ID_sublist : list of int
        List of tile IDs to process in this worker (1-based IDs).
    assigned_cases : list of str
        List of case names to include in the dataset.
    data_dir : str
        Root directory containing HR flood and rainfall tile files.
        Expected structure: {data_dir}/{tile_ID}/{case}_{type}_TS{ts}_{tile_kw}{tile_ID}.npy
    flood_LR_dir : str
        Root directory containing LR flood tile files.
        Expected structure: {flood_LR_dir}/{tile_ID}/{case}_FloodLR_TS{ts}_{tile_kw}{tile_ID}.npy
    tilepos_by_tile_used : dict
        Mapping from tile_ID_used (0-based) to compact position index [0..num_tiles-1].
        Used for indexing into static_by_tile and mask_by_tile arrays.
    loading_flag : bool
        If True, load array data into memory. If False, store file paths only.
    aux_kwargs : dict
        Auxiliary parameters including:
        - tile_kw: Tile keyword in filename (default: "tile")
        - flood_LR_kw: LR flood keyword (default: "FloodLR")
        - flood_HR_kw: HR flood keyword (default: "Flood")
        - rainfall_kw: Rainfall keyword (default: "Rainfall")
        - timestep_kw: Timestep keyword (default: "TS")
        - zero_discard_flag: Whether to discard samples with >90% zero values

    Returns
    -------
    dict
        Dictionary containing flat lists for all samples:
        - "rainfall": List of rainfall arrays/paths for input at timestep t
        - "flood_HR": List of HR flood arrays/paths for target at timestep t+1
        - "flood_LR": List of LR flood arrays/paths for auxiliary input at timestep t+1
        - "tile_idx": List of compact tile position indices for each sample

    Notes
    -----
    - Samples are created by pairing consecutive timesteps: rainfall(t) -> flood(t+1)
    - If zero_discard_flag is True, samples where >90% of HR flood pixels are zero
      are discarded to improve training efficiency.
    - The tile_idx values map samples to their corresponding static features and
      boundary masks via the static_by_tile and mask_by_tile dictionaries.
    """
    tile_kw = aux_kwargs["tile_kw"]
    flood_LR_kw = aux_kwargs["flood_LR_kw"]
    flood_HR_kw = aux_kwargs["flood_HR_kw"]
    rainfall_kw = aux_kwargs["rainfall_kw"]
    timestep_kw = aux_kwargs["timestep_kw"]
    zero_discard_flag = aux_kwargs["zero_discard_flag"]

    out = {"rainfall": [], "flood_HR": [], "flood_LR": [], "tile_idx": []}

    for tile_ID in tile_ID_sublist:
        tile_ID = int(tile_ID)
        tile_ID_used = int(tile_ID - 1)
        tile_pos = tilepos_by_tile_used[tile_ID_used]

        # scan rainfall timesteps once per tile
        ts_by_case = _scan_rainfall_timesteps_for_tile(
            data_dir=data_dir,
            tile_ID=tile_ID,
            rainfall_kw=rainfall_kw,
            timestep_kw=timestep_kw,
            tile_kw=tile_kw,
            case_filter=assigned_cases,
        )

        print("Scanned timesteps for tile", tile_ID)
        for cas in ts_by_case:
            print(f"Tile {tile_ID}, case {cas}, timesteps: {ts_by_case[cas]}")

        for cas in assigned_cases:
            timestep_list = ts_by_case.get(cas, [])

            if len(timestep_list) < 2:
                continue

            for i in range(len(timestep_list) - 1):
                ts_in = int(timestep_list[i])
                ts_out = int(timestep_list[i + 1])

                rf = os.path.join(data_dir, f"{tile_ID}", f"{cas}_{rainfall_kw}_{timestep_kw}{ts_in}_{tile_kw}{tile_ID}.npy")
                ff_HR = os.path.join(data_dir, f"{tile_ID}", f"{cas}_{flood_HR_kw}_{timestep_kw}{ts_out}_{tile_kw}{tile_ID}.npy")
                ff_LR = os.path.join(flood_LR_dir, f"{tile_ID}", f"{cas}_{flood_LR_kw}_{timestep_kw}{ts_out}_{tile_kw}{tile_ID}.npy")

                if not (os.path.exists(rf) and os.path.exists(ff_HR) and os.path.exists(ff_LR)):
                    continue

                if zero_discard_flag:
                    # mmap avoids reading everything eagerly (still touches pages when checking)
                    ff_HR_dta = np.load(ff_HR)
                    num_zero_ratio = np.sum(ff_HR_dta < 1e-6) / ff_HR_dta.size
                    # if num_zero_ratio >= 0.98:
                    if num_zero_ratio >= 0.9:
                        continue

                if loading_flag:
                    rf_item = load_geotiff_dynamic(rf)
                    ff_item = load_geotiff_dynamic(ff_HR)
                    lr_item = load_geotiff_dynamic(ff_LR)
                else:
                    rf_item, ff_item, lr_item = rf, ff_HR, ff_LR

                out["rainfall"].append(rf_item)   # rainfall(t)
                out["flood_HR"].append(ff_item)   # flood_HR(t+1)
                out["flood_LR"].append(lr_item)   # flood_LR(t+1)
                out["tile_idx"].append(tile_pos)

    return out


def preload_dataV2(
    case_namelist,
    data_dir,
    rank,
    world_size,
    tile_ID_list,
    aux_file_info=None,
    mask_file_prefix=None,
    flood_LR_dir=None,
    loading_ratio=1.0,
    tile_kw="tile",
    flood_LR_kw="FloodLR",
    flood_HR_kw="Flood",
    rainfall_kw="Rainfall",
    timestep_kw="TS",
    zero_discard_flag=True,
    num_cpu=4
):
    """
    Flat fast-path preloader for training dataset (optimized for FloodPredictionDatasetV2).

    This function creates a flat dataset structure optimized for fast training, where
    all samples are stored in flat lists rather than nested tile/case dictionaries.
    It supports distributed training by accepting rank and world_size parameters.

    Training Configuration
    ---------------------
    This preloader assumes the following fixed settings:
      - input_flood_sequence_length = 0 (no previous flood frames as input)
      - input_rainfall_sequence_length = 1 (single rainfall frame as input)
      - autoregressive_step = 0 (non-autoregressive mode)
      - use_flood_LR = True (LR flood as auxiliary input)

    Sample Pair Structure
    ---------------------
    Each training sample pair consists of:
      - Input:  rainfall(t), flood_LR(t+1), static_features(tile), boundary_mask(tile)
      - Target: flood_HR(t+1)

    Parameters
    ----------
    case_namelist : list of str
        List of case names (simulation identifiers) to include in the dataset.
    data_dir : str
        Root directory containing HR flood and rainfall tile files.
        Expected structure: {data_dir}/{tile_ID}/{case}_{type}_TS{ts}_{tile_kw}{tile_ID}.npy
    rank : int
        Current process rank in distributed training (0-indexed).
    world_size : int
        Total number of processes in distributed training.
    tile_ID_list : list of int
        List of tile IDs to include in the dataset (1-based).
    aux_file_info : dict, optional
        Dictionary mapping static feature names to their file path prefixes.
        Example: {"DEM": "/path/to/static/DEM", "roughness": "/path/to/static/landuseRoughness"}
        Files should be named: {prefix}_{tile_kw}{tile_ID}.tif
    mask_file_prefix : str, optional
        Prefix for boundary mask files.
        Files should be named: {prefix}_{tile_kw}{tile_ID}.npy
    flood_LR_dir : str
        Root directory containing LR flood tile files (required).
        Expected structure: {flood_LR_dir}/{tile_ID}/{case}_FloodLR_TS{ts}_{tile_kw}{tile_ID}.npy
    loading_ratio : float, optional
        If >= 1.0, load all dynamic arrays into memory (faster training).
        If < 1.0, store file paths only and load on-demand (default: 1.0).
    tile_kw : str, optional
        Tile keyword in filename (default: "tile").
    flood_LR_kw : str, optional
        LR flood keyword in filename (default: "FloodLR").
    flood_HR_kw : str, optional
        HR flood keyword in filename (default: "Flood").
    rainfall_kw : str, optional
        Rainfall keyword in filename (default: "Rainfall").
    timestep_kw : str, optional
        Timestep keyword in filename (default: "TS").
    zero_discard_flag : bool, optional
        If True, discard samples where >90% of HR flood pixels are zero (default: True).
    num_cpu : int, optional
        Number of parallel workers for scanning and loading (default: 4).

    Returns
    -------
    dict
        Flat dataset_info dictionary with the following structure:
        - "rainfall": List of rainfall arrays/paths for input at timestep t
        - "flood_HR": List of HR flood arrays/paths for target at timestep t+1
        - "flood_LR": List of LR flood arrays/paths for auxiliary input at timestep t+1
        - "tile_idx": List of compact tile position indices for each sample
        - "static_by_tile": List of static feature tensors, indexed by tile_idx
        - "mask_by_tile": List of boundary mask tensors, indexed by tile_idx

    Notes
    -----
    - The tile_idx values (0-based compact indices) map each sample to its
      corresponding static features and boundary masks.
    - Static features are loaded as float32 tensors with shape (num_features, H, W).
    - Boundary masks are loaded as float32 tensors with shape (1, H, W).
    - This function uses multiprocessing for parallel file scanning and loading.

    Example
    -------
    >>> dataset_info = preload_dataV2(
    ...     case_namelist=['0', '3', '5'],
    ...     data_dir="/path/to/train_cases-HR_Tile336_npy",
    ...     rank=0,
    ...     world_size=1,
    ...     tile_ID_list=[1, 2, 3, 4, 5],
    ...     aux_file_info={
    ...         "DEM": "/path/to/static_Tile336/DEM",
    ...         "roughness": "/path/to/static_Tile336/landuseRoughness",
    ...         "runoffCoef": "/path/to/static_Tile336/landuseRC"
    ...     },
    ...     mask_file_prefix="/path/to/mask_Tile336/boundaryMask",
    ...     flood_LR_dir="/path/to/train_cases-LR_Tile336_npy",
    ...     loading_ratio=1.0
    ... )
    >>> print(f"Total samples: {len(dataset_info['rainfall'])}")
    Total samples: 1234
    >>> print(f"Number of tiles: {len(dataset_info['static_by_tile'])}")
    Number of tiles: 5
    """

    if flood_LR_dir is None or not os.path.exists(flood_LR_dir):
        raise ValueError("flood_LR_dir must be provided and exist.")

    loading_flag = loading_ratio >= 1.0
    if loading_flag:
        print("Loading dynamic arrays into memory (fastest training).")
    else:
        print("Storing file paths only; dynamic arrays will be loaded in __getitem__().")

    assigned_cases = case_namelist
    print("All cases:", case_namelist, "world_size:", world_size, "rank:", rank)
    print(f"Rank {rank} assigned cases: {assigned_cases}")

    dataset_info = {
        "rainfall": [],
        "flood_HR": [],
        "flood_LR": [],
        "tile_idx": [],
        "static_by_tile": [],
        "mask_by_tile": [],
    }

    # Map tile_ID_used -> compact tile position [0..num_tiles_used-1]
    tilepos_by_tile_used = {}

    for tile_ID in tile_ID_list:  # assumes input tile_ID starts from 1
        tile_ID = int(tile_ID)
        tile_ID_used = int(tile_ID - 1)
        if tile_ID_used in tilepos_by_tile_used:
            continue

        tile_pos = len(dataset_info["static_by_tile"])
        tilepos_by_tile_used[tile_ID_used] = tile_pos

        # static (per tile)
        if aux_file_info is not None:
            aux_dta_list = []
            for k in aux_file_info:
                aux_dta = load_geotiff_static(f"{aux_file_info[k]}_{tile_kw}{tile_ID}.tif")
                aux_dta_list.append(aux_dta)
            static_np = np.stack(aux_dta_list, axis=0) if len(aux_dta_list) > 0 else None
        else:
            static_np = None

        static_tensor = torch.empty(0, dtype=torch.float32) if static_np is None else torch.as_tensor(static_np, dtype=torch.float32)

        # mask (per tile)
        mask_np = None
        if mask_file_prefix is not None:
            mask_file = f"{mask_file_prefix}_{tile_kw}{tile_ID}.npy"
            if os.path.exists(mask_file):
                mask_np = np.load(mask_file).astype(np.float32)

        if mask_np is None:
            mask_tensor = torch.empty(0, dtype=torch.float32)
        else:
            mask_tensor = torch.as_tensor(mask_np, dtype=torch.float32)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)

        dataset_info["static_by_tile"].append(static_tensor)
        dataset_info["mask_by_tile"].append(mask_tensor)

    tile_splits = [arr.tolist() for arr in np.array_split(np.array(tile_ID_list, dtype=int), num_cpu) if len(arr) > 0]
    print(f"Using num_cpu={num_cpu}, tile splits:", tile_splits)

    aux_kwargs = dict(
        tile_kw=tile_kw,
        flood_LR_kw=flood_LR_kw,
        flood_HR_kw=flood_HR_kw,
        rainfall_kw=rainfall_kw,
        timestep_kw=timestep_kw,
        zero_discard_flag=zero_discard_flag,
    )

    # --- run workers
    ctx = mp.get_context("spawn")
    tasks = [
        (tile_sub, 
         assigned_cases, 
         data_dir, 
         flood_LR_dir, 
         tilepos_by_tile_used, 
         loading_flag, 
         aux_kwargs)
        for tile_sub in tile_splits
    ]

    with ctx.Pool(processes=num_cpu) as pool:
        partials = pool.starmap(_tile_worker_build_pairs, tasks)

    # --- merge
    for part in partials:
        dataset_info["rainfall"].extend(part["rainfall"])
        dataset_info["flood_HR"].extend(part["flood_HR"])
        dataset_info["flood_LR"].extend(part["flood_LR"])
        dataset_info["tile_idx"].extend(part["tile_idx"])

    # sanity
    n = len(dataset_info["rainfall"])
    assert len(dataset_info["flood_HR"]) == n
    assert len(dataset_info["flood_LR"]) == n
    assert len(dataset_info["tile_idx"]) == n

    print(f"Total number of paired samples: {n}")
    print(f"Number of tiles (static/mask entries): {len(dataset_info['static_by_tile'])}")

    return dataset_info


def get_autoregressive_dataloader(flood_dataset_info, num_tile, num_case, seq_length, autoregressive_step, batch_size, num_workers, rank, world_size, use_flood_LR=True, loading_ratio=1.0):
    dataset_tmp = FloodPredictionDatasetV2(flood_dataset_info=flood_dataset_info)
    train_sampler = DistributedSampler(dataset_tmp, num_replicas=world_size, rank=rank, shuffle=True)
    #dataset_tmp.load_dynamic_data_memory(loading_ratio, num_cpu=num_workers)
    train_loader = DataLoader(dataset_tmp, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, prefetch_factor=4)

    return train_loader, train_sampler


def get_autoregressive_dataloader_sgpu(flood_dataset_info, num_tile, num_case, seq_length, autoregressive_step, batch_size, num_workers, use_flood_LR=True, loading_ratio=1.0):
    dataset_tmp = FloodPredictionDatasetV2(flood_dataset_info=flood_dataset_info)
    #dataset_tmp.load_dynamic_data_memory(loading_ratio, num_cpu=num_workers)
    train_loader = DataLoader(dataset_tmp, batch_size=batch_size, num_workers=num_workers, prefetch_factor=4, shuffle=True)

    return train_loader


if __name__ == "__main__":
    # for tile_size in [128, 256, 512]:
    for tile_size in [512, 1024]:
        # for ds_type in ["train", "val", "test"]:
        for ds_type in ["train", "val"]:
            batch_convert_geotiff_to_numpy(f"/home/sklhse/WORK/lyy/data/{ds_type}_cases_Tile{tile_size}", 
                                           f"/home/sklhse/WORK/lyy/data/{ds_type}_cases_Tile{tile_size}_npy", 
                                           num_cpu=8)
            
            batch_convert_geotiff_to_numpy(f"/home/sklhse/WORK/lyy/data/{ds_type}_cases_Tile{tile_size}/flood_LR", 
                                           f"/home/sklhse/WORK/lyy/data/{ds_type}_cases_Tile{tile_size}_npy/flood_LR", 
                                           num_cpu=8)