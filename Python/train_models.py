from utils_models import Dataset, loop_training
from datetime import datetime
from pathlib import Path
import pandas as pd
import pickle

# -------------------------------
# Run configuration
# -------------------------------

# Dataset "version" selector used by Dataset(...) internal filters.
version = 1

# If you want to reuse a previous run folder, set `ctime_id` to that timestamp
# (e.g., "20250131T104512"). If None, a new timestamped folder is created.
ctime_id = None

# When True, iterate over *all* predictor-set IDs for the chosen version.
# When False, train only the specific `pred_id` defined below.
use_loop = True

if ctime_id is None:
    # Use a wall-clock timestamp as this run's unique ID (YYYYMMDDThhmmss).
    ctime_id = datetime.now().strftime("%Y%m%dT%H%M%S")

# Project root (this script is expected to be run as a file, not in a notebook)
ROOT = Path(__file__).resolve().parent.parent

# Output directory for logs, metrics, confusion matrices, etc.
outpath = Path(ROOT, f"results/logs/train_dataset_v{version}_{ctime_id}")
outpath.mkdir(exist_ok=True, parents=True)

# -------------------------------
# Data loading & preprocessing
# -------------------------------

# Source data produced by the feature engineering pipeline
labels_dataset_path = Path(ROOT, "results/dataset.gpkg")
label_codes_path = Path(ROOT, "data/labels/label_codes.csv")

# Build the Dataset object (handles cleaning, NDVI, PCA features, etc.)
dataset = Dataset(labels_dataset_path, label_codes_path, version)

# If you want to run a single predictor group (when use_loop=False), set it here.
# NOTE: With `use_loop=True`, this value is ignored (kept for convenience).
pred_id = "LspringPCA"

# -------------------------------
# Warm-start / resume support
# -------------------------------

# The script writes three artifacts inside `outpath`:
#   - best_gridcv_stats.csv      : tidy per-run metrics (append-only across runs)
#   - gridcv_stats.pkl           : list of per-run grid-search summaries (params/scores)
#   - confusion_matrices.pkl     : list of per-run confusion matrices

# Initialize stats list, optionally preloading existing CSV to continue appending.
stats_path = Path(outpath, "best_gridcv_stats.csv")
try:
    saved_stats = pd.read_csv(stats_path)
    stats_list = [saved_stats]
    # Optional: if running a single pred_id, ensure we don't duplicate it in-place
    if not use_loop and pred_id in pd.unique(saved_stats["pred_id"]):
        raise ValueError(f"Predictor id '{pred_id}' already exists in {stats_path}.")
except FileNotFoundError:
    stats_list = []

# Initialize grid-search summaries
try:
    with open(Path(outpath, "gridcv_stats.pkl"), "rb") as f:
        grid_list = pickle.load(f)
except FileNotFoundError:
    grid_list = []

# Initialize confusion matrices
try:
    with open(Path(outpath, "confusion_matrices.pkl"), "rb") as f:
        cm_list = pickle.load(f)
except FileNotFoundError:
    cm_list = []

# -------------------------------
# Training
# -------------------------------

if not use_loop:
    # Single-run mode: train just the specified predictor set.
    print(f"Start model training with {pred_id} dataset")

    # BUGFIX: loop_training signature requires (dataset, pred_id, cm_list, stats_list, grid_list)
    # and returns the three lists again. The original code called it with fewer args.
    cm_list, stats_list, grid_list = loop_training(
        dataset, pred_id, cm_list, stats_list, grid_list
    )

else:
    # Loop mode: iterate over all predictor-set IDs that match the current version.
    for pred_id in dataset.get_predictor_groups(version):
        print(f"Start model training with {pred_id} dataset")
        cm_list, stats_list, grid_list = loop_training(
            dataset, pred_id, cm_list, stats_list, grid_list
        )

# -------------------------------
# Persist results
# -------------------------------

# Save/append tidy metrics for the current session
stats_outpath = Path(outpath, "best_gridcv_stats.csv")
pd.concat(stats_list, ignore_index=True).to_csv(stats_outpath, index=False)

# Store GridSearchCV summaries (list of dicts) to a pickle file
with open(Path(outpath, "gridcv_stats.pkl"), "wb") as fp:
    pickle.dump(grid_list, fp)

# Store confusion matrices (list of dicts) to a pickle file
with open(Path(outpath, "confusion_matrices.pkl"), "wb") as fp:
    pickle.dump(cm_list, fp)
