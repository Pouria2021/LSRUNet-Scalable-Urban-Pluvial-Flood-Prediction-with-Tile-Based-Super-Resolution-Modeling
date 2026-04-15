"""
Generate a synthetic flood prediction dataset for LSRUNet demo.

Creates a minimal but complete dataset with:
    - 3 tiles (256x256 HR, 32x32 LR)
    - 2 training cases + 1 validation case
    - 4 timesteps per case (600, 1200, 1800, 2400 seconds)
    - Static features: DEM, Manning's roughness, runoff coefficient
    - Boundary masks
    - Tile polygons (GeoJSON)
    - Depth distribution for weighted loss
    - Example training config

Usage:
    python generate_synthetic_data.py

Output:
    synthetic_dataset/   -- all data files
    train_config_synthetic.json  -- ready-to-use config
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TILE_SIZE = 256          # HR tile size in pixels
LR_FACTOR = 8            # HR-to-LR downscaling factor
LR_SIZE = TILE_SIZE // LR_FACTOR
NUM_TILES = 3
TRAIN_CASES = ["rain_event1", "rain_event2"]
VAL_CASES = ["rain_event3"]
TIMESTEPS = [600, 1200, 1800, 2400]
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "synthetic_dataset")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def smooth_noise(shape, scale=32, rng=None):
    """Generate a smooth random field by bilinear upscaling of coarse noise."""
    if rng is None:
        rng = np.random.default_rng()
    ch = shape[0] // scale + 2
    cw = shape[1] // scale + 2
    coarse = rng.standard_normal((ch, cw))

    y = np.linspace(0, ch - 1, shape[0])
    x = np.linspace(0, cw - 1, shape[1])
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, ch - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, cw - 1)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]

    return (coarse[np.ix_(y0, x0)] * (1 - fy) * (1 - fx) +
            coarse[np.ix_(y1, x0)] * fy * (1 - fx) +
            coarse[np.ix_(y0, x1)] * (1 - fy) * fx +
            coarse[np.ix_(y1, x1)] * fy * fx)


def gaussian_bump(shape, center_yx, sigma, amplitude):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = center_yx
    return amplitude * np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))


def downsample(arr, factor):
    """Downsample 2D array by block averaging."""
    h, w = arr.shape
    nh, nw = h // factor, w // factor
    return arr[:nh * factor, :nw * factor].reshape(nh, factor, nw, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Generate per-tile static data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
print("Generating static terrain features...")

dem_list = []
roughness_list = []
runoff_list = []
mask_list = []

for tile_id in range(1, NUM_TILES + 1):
    # DEM: base plane + hills + valley + smooth noise
    dem = np.ones((TILE_SIZE, TILE_SIZE)) * 15.0
    dem += gaussian_bump((TILE_SIZE, TILE_SIZE),
                         (TILE_SIZE * 0.3, TILE_SIZE * 0.5), 40, 8.0)
    dem += gaussian_bump((TILE_SIZE, TILE_SIZE),
                         (TILE_SIZE * 0.7, TILE_SIZE * 0.3), 30, 5.0)
    dem -= gaussian_bump((TILE_SIZE, TILE_SIZE),
                         (TILE_SIZE * 0.5, TILE_SIZE * 0.6), 50, 6.0)
    dem += smooth_noise((TILE_SIZE, TILE_SIZE), scale=32, rng=rng) * 1.0
    dem = np.clip(dem, 0, 30).astype(np.float32)
    dem_list.append(dem)

    # Manning's roughness: smooth field in [0.01, 0.15]
    rough = smooth_noise((TILE_SIZE, TILE_SIZE), scale=16, rng=rng)
    rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)
    rough = (rough * 0.14 + 0.01).astype(np.float32)
    roughness_list.append(rough)

    # Runoff coefficient: smooth field in [0.3, 0.9]
    rc = smooth_noise((TILE_SIZE, TILE_SIZE), scale=16, rng=rng)
    rc = (rc - rc.min()) / (rc.max() - rc.min() + 1e-8)
    rc = (rc * 0.6 + 0.3).astype(np.float32)
    runoff_list.append(rc)

    # Boundary mask: 1 inside, 0 at edges
    mask = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.float32)
    border = 8
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    mask_list.append(mask)

# ---------------------------------------------------------------------------
# Generate dynamic data (rainfall, flood HR, flood LR)
# ---------------------------------------------------------------------------
rainfall_profiles = {
    "rain_event1": [0.0, 25.0, 40.0, 15.0],   # moderate event
    "rain_event2": [0.0, 50.0, 30.0, 10.0],    # intense event
    "rain_event3": [0.0, 15.0, 20.0, 5.0],     # mild event
}

all_cases = TRAIN_CASES + VAL_CASES
print(f"Generating dynamic data for {len(all_cases)} cases x {NUM_TILES} tiles x {len(TIMESTEPS)} timesteps...")

for case in all_cases:
    is_train = case in TRAIN_CASES
    split = "train" if is_train else "val"
    data_dir = os.path.join(BASE_DIR, split)
    lr_dir = os.path.join(data_dir, "flood_LR")

    for tile_id in range(1, NUM_TILES + 1):
        tile_dir = os.path.join(data_dir, str(tile_id))
        lr_tile_dir = os.path.join(lr_dir, str(tile_id))
        os.makedirs(tile_dir, exist_ok=True)
        os.makedirs(lr_tile_dir, exist_ok=True)

        dem = dem_list[tile_id - 1]
        mask = mask_list[tile_id - 1]
        dem_norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
        susceptibility = 1.0 - dem_norm

        flood_accum = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

        for ts_idx, ts in enumerate(TIMESTEPS):
            rf_intensity = rainfall_profiles[case][ts_idx]

            # Rainfall field: mostly uniform with slight spatial variation
            rainfall = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.float32) * rf_intensity
            rainfall += (smooth_noise((TILE_SIZE, TILE_SIZE), scale=32, rng=rng)
                         * rf_intensity * 0.1).astype(np.float32)

            # Flood accumulates in low-lying areas, scaled by rainfall
            flood_accum += rf_intensity * 0.005 * susceptibility
            flood_depth = flood_accum * mask
            flood_depth += (smooth_noise((TILE_SIZE, TILE_SIZE), scale=16, rng=rng)
                            * 0.01).astype(np.float32)
            flood_depth = np.clip(flood_depth, 0, 1.5).astype(np.float32)

            flood_lr = downsample(flood_depth, LR_FACTOR).astype(np.float32)

            np.save(os.path.join(tile_dir, f"{case}_Rainfall_TS{ts}_Tile{tile_id}.npy"),
                    rainfall)
            np.save(os.path.join(tile_dir, f"{case}_Flood_TS{ts}_Tile{tile_id}.npy"),
                    flood_depth)
            np.save(os.path.join(lr_tile_dir, f"{case}_FloodLR_TS{ts}_Tile{tile_id}.npy"),
                    flood_lr)

# ---------------------------------------------------------------------------
# Save static GeoTIFF files
# ---------------------------------------------------------------------------
static_dir = os.path.join(BASE_DIR, "static")
mask_dir = os.path.join(BASE_DIR, "mask")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

print("Saving static GeoTIFF files...")

for tile_id in range(1, NUM_TILES + 1):
    x0 = (tile_id - 1) * TILE_SIZE
    x1 = tile_id * TILE_SIZE
    transform = from_bounds(x0, 0, x1, TILE_SIZE, TILE_SIZE, TILE_SIZE)

    for data, prefix in [
        (dem_list[tile_id - 1], "DEM"),
        (roughness_list[tile_id - 1], "landuseRoughness"),
        (runoff_list[tile_id - 1], "landuseRC"),
    ]:
        filepath = os.path.join(static_dir, f"{prefix}_Tile{tile_id}.tif")
        with rasterio.open(filepath, 'w', driver='GTiff',
                           height=TILE_SIZE, width=TILE_SIZE, count=1,
                           dtype='float32', transform=transform) as dst:
            dst.write(data, 1)

    np.save(os.path.join(mask_dir, f"boundaryMask_Tile{tile_id}.npy"),
            mask_list[tile_id - 1])

# ---------------------------------------------------------------------------
# Tile polygons (GeoJSON)
# ---------------------------------------------------------------------------
print("Generating tile polygons...")

features = []
for tile_id in range(1, NUM_TILES + 1):
    x0 = (tile_id - 1) * TILE_SIZE
    x1 = tile_id * TILE_SIZE
    features.append({
        "type": "Feature",
        "properties": {"tile_id": tile_id},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, 0], [x1, 0], [x1, TILE_SIZE],
                             [x0, TILE_SIZE], [x0, 0]]]
        }
    })

with open(os.path.join(BASE_DIR, "tile_polygons.geojson"), 'w') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

# ---------------------------------------------------------------------------
# Case list CSVs
# ---------------------------------------------------------------------------
pd.DataFrame({"case_ID": TRAIN_CASES}).to_csv(
    os.path.join(BASE_DIR, "train_cases.csv"), index=False)
pd.DataFrame({"case_ID": VAL_CASES}).to_csv(
    os.path.join(BASE_DIR, "val_cases.csv"), index=False)

# ---------------------------------------------------------------------------
# Depth distribution CSV (for weighted loss)
# ---------------------------------------------------------------------------
H_MAX = 1.0
NUM_BINS = 11
depth_bins = np.linspace(0, H_MAX, NUM_BINS)
counts = [int(1e6 * np.exp(-d * 5)) for d in depth_bins]

pd.DataFrame({
    "boundary": [int(d * 100) for d in depth_bins],
    "Count": counts
}).to_csv(os.path.join(BASE_DIR, "depth_distribution_D10.csv"), index=False)

# ---------------------------------------------------------------------------
# Training config for the synthetic dataset
# ---------------------------------------------------------------------------
config = {
    "case_identifier": "synthetic_demo",
    "model": {
        "num_target_channels": 512,
        "num_aux_target_channels": 512,
        "num_levels": 4,
        "mtl_flag": False,
        "use_flood_LR": True,
        "save_root_dir": "./synthetic_results",
        "initial_state": "",
        "type": "UNet"
    },
    "dataset": {
        "tile_keywords": "Tile",
        "num_tile": NUM_TILES,
        "tile_polygon_path": "./synthetic_dataset/tile_polygons.geojson",
        "train_root_dir": "./synthetic_dataset/train",
        "train_case_list": "./synthetic_dataset/train_cases.csv",
        "valid_root_dir": "./synthetic_dataset/val",
        "valid_case_list": "./synthetic_dataset/val_cases.csv",
        "aux_file_info": {
            "DEM": "./synthetic_dataset/static/DEM",
            "runoffCoef": "./synthetic_dataset/static/landuseRC",
            "roughness": "./synthetic_dataset/static/landuseRoughness"
        },
        "mask_file_prefix": "./synthetic_dataset/mask/boundaryMask",
        "flood_LR_train_root_dir": "./synthetic_dataset/train/flood_LR",
        "flood_LR_valid_root_dir": "./synthetic_dataset/val/flood_LR"
    },
    "learning_settings": {
        "batch_size": 2,
        "num_workers": 2,
        "sample_weight_path": "./synthetic_dataset/depth_distribution_D10.csv",
        "learning_rates": {
            "nar_base": 1e-4,
            "ar_base": 1e-5,
            "scaling_factor_nar": 1e-2,
            "scaling_factor_ar": 1e-1
        },
        "loss": {
            "delta": 1.0,
            "thresholds": [0.15, 0.27, 0.4, 0.6]
        },
        "mtl_dynamic_weight": False,
        "autoregressive": {
            "num_prev_step": 0,
            "max_forward_step": 0
        },
        "epochs": {
            "num_non_autoregressive": 2,
            "num_autoregressive": 0
        }
    }
}

config_path = os.path.join(SCRIPT_DIR, "train_config_synthetic.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Synthetic dataset generated successfully!")
print(f"Location: {os.path.abspath(BASE_DIR)}")
print(f"Config:   {os.path.abspath(config_path)}")
print(f"\nTiles: {NUM_TILES} ({TILE_SIZE}x{TILE_SIZE} HR, {LR_SIZE}x{LR_SIZE} LR)")
print(f"Train cases: {TRAIN_CASES}")
print(f"Val cases:   {VAL_CASES}")
print(f"Timesteps:   {TIMESTEPS}")
print("=" * 60)
