# LSRUNet: Tile-Based Super-Resolution for Urban Pluvial Flood Prediction

[![DOI](https://zenodo.org/badge/1211052454.svg)](https://doi.org/10.5281/zenodo.19584563)

**Pouria Nakhaei**, *State Key Laboratory of Hydro-science and Engineering, Department of Hydraulic Engineering, Tsinghua University, Beijing, China

---

## Abstract

LSRUNet is a deep learning framework for scalable urban pluvial flood prediction using tile-based super-resolution modeling. The model takes low-resolution flood simulations, rainfall data, and static terrain features as input, and predicts high-resolution flood inundation maps. The architecture is based on a ResUNet with Squeeze-and-Excitation (SE) channel attention blocks, supporting multi-task learning (depth + binary flood mask) and distributed multi-GPU training.

## Architecture

The core model (**ResUNet_aux**) is an encoder-decoder network with skip connections:

- **Encoder**: SEResBlock-based encoder with SE channel attention at each level. Channel widths follow a doubling pattern: base, 2x, 4x, 8x, 16x of `num_target_channels / 2^num_levels`.
- **Decoder**: Bilinear upsampling with 1x1 skip projections at each level.
- **Output**: ReLU-activated single-channel flood depth prediction (non-negative).

Alternative encoder backends are also provided:
- **MaxViT** (`model_vit.py`): Multi-axis ViT with block and grid attention.
- **SwinV2** (`model_swinT.py`): Swin Transformer V2 with scaled cosine attention and log-spaced CPB.

### Model Inputs

| Input | Description | Channels |
|-------|-------------|----------|
| Rainfall | Precipitation at current timestep | 1 |
| Flood LR | Low-resolution flood depth (auxiliary) | 1  |
| DEM | Digital elevation model (static) | 1 |
| Roughness | Manning's roughness coefficient (static) | 1 |
| Runoff Coef. | Runoff coefficient (static) | 1 |

## Synthetic Demo Dataset

A small synthetic dataset is included for testing and demonstration. It contains:

- **3 tiles** (256x256 HR, 32x32 LR)
- **2 training cases** + **1 validation case** (4 timesteps each at 600, 1200, 1800, 2400s)
- **Static features**: DEM, Manning's roughness, runoff coefficient (GeoTIFF)
- **Boundary masks** (NumPy)
- **Tile polygons** (GeoJSON)
- **Depth distribution** for weighted loss

Quick start with synthetic data:

```bash
torchrun --nproc_per_node=1 main.py --case_config_path train_config_synthetic.json
```

To regenerate the dataset:

```bash
python generate_synthetic_data.py
```

## Project Structure

```
.
├── main.py              # Distributed training script (DDP via torchrun)
├── model_unet.py        # ResUNet with SE blocks (ResUNet_aux, ResUNet_aux_MTL)
├── model_vit.py         # MaxViT U-Net encoder-decoder
├── model_swinT.py       # Swin Transformer V2 U-Net
├── dataset.py           # Dataset classes and data loading utilities
├── metric.py            # Loss functions and evaluation metrics
├── utils.py             # LR schedule helpers and depth distribution tools
├── train_config.json              # Training configuration (placeholder paths)
├── train_config_synthetic.json    # Config for the synthetic demo dataset
├── generate_synthetic_data.py     # Script to generate synthetic demo data
├── synthetic_dataset/             # Synthetic demo data
├── requirements.txt               # Python dependencies
├── LICENSE              # Apache License 2.0
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
| `dataset` | `train_root_dir` | Path to training data |
| `learning_settings` | `batch_size` | Per-GPU batch size |
| `learning_settings` | `loss.thresholds` | Flood depth thresholds for metrics |

All path fields in the config must be updated to point to your local data directories.

### Supported Metrics

Evaluated at configurable flood depth thresholds (e.g., 0.15m, 0.27m, 0.4m, 0.6m):

- **RMSE** / **MAE**: Regression accuracy (masked by threshold)
- **IoU**: Intersection over Union
- **CSI**: Critical Success Index
- **F2-Score**: Weighted F-score emphasizing recall
- **POD**: Probability of Detection
- **FAR**: False Alarm Rate
- **Bias**: Mean prediction bias

## Citation


*DOI will be added upon publication.*

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments


