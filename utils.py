"""Utility functions for training configuration.

Provides:
    - Cosine annealing LR cycle calculation with warm restarts
    - Depth distribution analysis for weighted loss computation
"""

import os
import pandas as pd
import numpy as np
import rasterio


#################################################
############ Learning rate cycles ###############
#################################################
# def calculate_nar_cycles(T_0, T_mult, max_epochs):
#     cycles = []
#     epoch_start = 0
#     cycle_length = T_0
#     while epoch_start + cycle_length <= max_epochs:
#         cycle_end = epoch_start + cycle_length - 1
#         if cycle_length >= 10:  # Ensure cycle length is valid
#             cycles.append((epoch_start, cycle_end))
#         epoch_start = cycle_end + 1
#         cycle_length *= T_mult
        
#     return cycles


# def calculate_ar_cycles(start_epoch, cycle_length, max_epochs):
#     cycles = []
#     epoch_start = start_epoch
#     while epoch_start + cycle_length <= max_epochs:
#         cycle_end = epoch_start + cycle_length - 1
#         cycles.append((epoch_start, cycle_end))
#         epoch_start = cycle_end + 1
#     return cycles


def calculate_cosLR_cycles(epoch_start, T_0, T_mult, max_epochs):
    cycles = []
    cycle_length = T_0
    while epoch_start + cycle_length <= max_epochs:
        cycle_end = epoch_start + cycle_length - 1
        cycles.append((epoch_start, cycle_end))
        epoch_start = cycle_end + 1
        cycle_length *= T_mult
    return cycles


def combine_cycles(nar_cycles, ar_cycles):
    return nar_cycles + ar_cycles


def is_last_epochs_of_cycle(epoch, nar_cycles, ar_cycles):
    for cycle_start, cycle_end in nar_cycles:
        cycle_length = cycle_end - cycle_start + 1
        if cycle_length > 10:  # Ensure cycle is longer than 10 epochs
            last_10_start = cycle_end - 9
            if last_10_start <= epoch <= cycle_end:
                return True  

    for cycle_start, cycle_end in ar_cycles:
        cycle_length = cycle_end - cycle_start + 1
        if cycle_length > 10:  # Ensure cycle is longer than 10 epochs
            last_10_start = cycle_end - 9
            if last_10_start <= epoch <= cycle_end:
                return True  

    return False


def get_progress_ratio_of_cycle(epoch, lr_cycle_list):
    for cycle_start, cycle_end in lr_cycle_list:
        cycle_length = cycle_end - cycle_start + 1
        cycle_passed = epoch - cycle_start
        ratio = cycle_passed / cycle_length
        if 0 <= ratio < 1:
            return ratio
#################################################


def get_sampleDataset_varDistribution(train_dir):
    H_MAX = 1.0
    NUM_BINS = 11

    depth_bin = np.linspace(0, H_MAX, NUM_BINS)
    deltaD = H_MAX / (NUM_BINS - 1.0)
    depth_count_ref = {}
    for d in depth_bin:
        depth_count_ref[str(int(d * 100.0))] = 0   # the number of pixels ranging between [d, d + 0.05)

    tile_file_list = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.endswith('.tif') and "_Flood_" in d and "Tile" in d]
    print("Total number of samples:", len(tile_file_list))
    
    for tf in tile_file_list:
        with rasterio.open(os.path.join(train_dir, tf)) as src:
            sample = src.read(1)  # Read the first band
            sample_nodata = src.nodata
            sample[sample == sample_nodata] = 0
            sample[np.isnan(sample)] = 0

            for d in depth_bin:
                if np.abs(d - H_MAX) < 1e-6:
                    depth_count_ref[str(int(d * 100.0))] += np.sum((sample >= d))
                else:
                    depth_count_ref[str(int(d * 100.0))] += np.sum((sample >= d) & (sample < d + deltaD))
            
    depth_df = pd.DataFrame(depth_count_ref.items(), columns=['boundary', 'Count'])
    depth_df.to_csv(os.path.join(train_dir, 'depth_distribution_D10.csv'), index=False)


if __name__ == "__main__":
    get_sampleDataset_varDistribution('/data/lyy/server/pouria_case/UNet_dataset/train_cases_Tile512')