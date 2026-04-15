# LSRUNet: Scalable Urban Pluvial Flood Prediction with Tile-Based Super-Resolution Modeling

[![DOI](https://zenodo.org/badge/1211052454.svg)](https://doi.org/10.5281/zenodo.19584563)

**Pouria Nakhaei**, *State Key Laboratory of Hydro-science and Engineering, Department of Hydraulic Engineering, Tsinghua University, Beijing, China*

---

## Abstract

LSRUNet is a deep learning framework for scalable urban pluvial flood prediction using tile-based super-resolution modeling. The model takes low-resolution (100 m) flood simulations, rainfall data, and static terrain features as input, and predicts high-resolution (10 m) flood inundation maps. The architecture is based on a ResUNet with Squeeze-and-Excitation (SE) channel attention blocks, supporting multi-task learning (flood depth + binary inundation mask) and distributed multi-GPU training.

## Data Description

### Spatial Resolution

The model performs super-resolution from **100 m (LR)** to **10 m (HR)** spatial resolution. All input grids (rainfall, flood LR, DEM, roughness, runoff coefficient) are resampled to the same pixel dimensions before being fed to the network.

The study area is divided into tiles of varying sizes (128x128, 256x256, 512x512, 1024x1024 pixels) to cover different urban sub-regions at 10 m resolution.

### Temporal Resolution

Flood simulations use a temporal resolution of **10 minutes** over **24-hour rainfall events** (144 timesteps per event). Each training sample pairs rainfall at timestep *t* with the corresponding flood depth at timestep *t+1*.

### Input Features

| Input | Description | Resolution | Type |
|-------|-------------|------------|------|
| Rainfall | Precipitation intensity at timestep *t* | 10 m | Dynamic |
| Flood LR | Low-resolution flood depth at timestep *t+1* | 100 m | Dynamic |
| DEM | Digital elevation model | 10 m | Static |
| Roughness | Manning's roughness coefficient | 10 m | Static |
| Runoff Coef. | Runoff coefficient | 10 m | Static |

### Data Format

- **Dynamic data** (rainfall, flood HR, flood LR): NumPy arrays (`.npy`), organized as `{case}_{keyword}_TS{timestep}_Tile{ID}.npy` in per-tile subdirectories
- **Static features** (DEM, roughness, runoff): GeoTIFF files per tile
- **Boundary masks**: NumPy arrays per tile
- **Tile polygons**: GeoJSON with one polygon per tile
- **Case lists**: CSV with `case_ID` column
- **Depth distribution**: CSV with `boundary` (cm) and `Count` columns for weighted loss

### Directory Structure

```
data/
├── train_cases-HR_Tile{size}_npy/       # HR rainfall + flood depth
│   ├── {tile_ID}/
│   │   ├── {case}_Rainfall_TS{ts}_Tile{tile_ID}.npy
│   │   └── {case}_Flood_TS{ts}_Tile{tile_ID}.npy
│   └── ...
├── train_cases-LR_Tile{size}_npy/       # LR flood depth
│   ├── {tile_ID}/
│   │   └── {case}_FloodLR_TS{ts}_Tile{tile_ID}.npy
│   └── ...
├── static_Tile{size}/                   # Static terrain features
│   ├── DEM_Tile{ID}.tif
│   ├── landuseRoughness_Tile{ID}.tif
│   └── landuseRC_Tile{ID}.tif
├── mask_Tile{size}/                     # Boundary masks
│   └── boundaryMask_Tile{ID}.npy
├── tile_polygons.geojson
├── train_cases.csv
├── val_cases.csv
└── depth_distribution_D10.csv
```

## Architecture

The core model (**ResUNet_aux**) is an encoder-decoder network with skip connections:

- **Encoder**: SEResBlock-based encoder with SE channel attention at each level. Channel widths follow a doubling pattern: base, 2x, 4x, 8x, 16x of `num_target_channels / 2^num_levels`.
- **Decoder**: Bilinear upsampling with 1x1 skip projections at each level.
- **Output**: ReLU-activated single-channel flood depth prediction (non-negative).
- **Multi-task variant** (**ResUNet_aux_MTL**): Additional binary flood mask head for simultaneous depth + inundation prediction.

Alternative encoder backends:
- **MaxViT** (`model_vit.py`): Multi-axis ViT with block and grid attention.
- **SwinV2** (`model_swinT.py`): Swin Transformer V2 with scaled cosine attention and log-spaced CPB.

## Project Structure

```
.
├── main.py                          # Distributed training script (DDP via torchrun)
├── model_unet.py                    # ResUNet with SE blocks (ResUNet_aux, ResUNet_aux_MTL)
├── model_vit.py                     # MaxViT U-Net encoder-decoder
├── model_swinT.py                   # Swin Transformer V2 U-Net
├── dataset.py                       # Dataset classes and data loading utilities
├── metric.py                        # Loss functions and evaluation metrics
├── utils.py                         # LR schedule helpers and depth distribution tools
├── train_config.json                # Training configuration (placeholder paths)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Apache License 2.0
└── README.md
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.0 (CUDA)
- See `requirements.txt` for full dependencies.

Install:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
torchrun --nproc_per_node=4 main.py --case_config_path train_config.json
```

### Configuration

Edit `train_config.json` to set your paths and hyperparameters. Key fields:

| Section | Field | Description |
|---------|-------|-------------|
| `model` | `type` | `"UNet"`, `"MaxVIT"`, or `"SwinT"` |
| `model` | `num_target_channels` | Base channel width multiplier |
| `model` | `num_levels` | Encoder depth (default: 4) |
| `model` | `mtl_flag` | Enable multi-task (depth + mask) |
| `model` | `use_flood_LR` | Include LR flood as auxiliary input |
| `dataset` | `train_root_dir` | Path to training HR data |
| `dataset` | `flood_LR_train_root_dir` | Path to training LR flood data |
| `learning_settings` | `batch_size` | Per-GPU batch size |
| `learning_settings` | `loss.thresholds` | Flood depth thresholds for metrics (m) |

All path fields must be updated to point to your local data directories.

### Supported Metrics

Evaluated at configurable flood depth thresholds (e.g., 0.15, 0.27, 0.4, 0.6 m):

- **RMSE** / **MAE**: Regression accuracy (masked by threshold)
- **IoU**: Intersection over Union
- **CSI**: Critical Success Index
- **F2-Score**: Weighted F-score emphasizing recall
- **POD**: Probability of Detection
- **FAR**: False Alarm Rate
- **Bias**: Mean prediction bias

### Loss Function

Weighted Huber loss with depth-bin weighting and optional boundary/bin masks. Loss weights are derived from the depth distribution CSV to address class imbalance (most pixels have zero or shallow flooding).

## Synthetic Demo Dataset

A small synthetic dataset (`synthetic_dataset/`) is included for testing the pipeline without real data. It contains 3 tiles, 2 training + 1 validation case, and 4 timesteps. Generated via `generate_synthetic_data.py`.

Quick start:

```bash
torchrun --nproc_per_node=1 main.py --case_config_path train_config_synthetic.json
```

## Citation


*DOI will be added upon publication.*

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments
