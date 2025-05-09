import torch
import csv
import os
import multiprocessing as mp
from training.subgroup_aware_clipping import *
from utils.evaluation import *
from utils.helper_functions import *
from data.data_generation import *

# Fixed parameters
setting = {"min_fraction": 0.1, "min_signal": 2.0, "flip_min": 0.2}
min_frac = setting["min_fraction"]
min_signal = setting["min_signal"]
flip_min = setting["flip_min"]

maj_signal = -2.0
flip_maj = 0.05
dim = 200
n_train = 30000
num_samples = n_train
seed = 13
lr = 0.1
wd = 1e-4
num_epochs = 100
eps = 1.0
delta = 1 / (n_train ** 2)

maj_clip_vals = [0.5, 0.75, 1.0]
min_clip_vals = [1.0, 1.25, 1.5, 1.75, 2.0]

# Generate once
train_data = generate_data_weight_and_bias(n_train, dim, min_frac, min_signal, maj_signal, flip_min, flip_maj, seed)
loader = make_dataloader_flags(train_data, batch_size=256)

val_data = generate_data_weight_and_bias(500, dim, min_frac, min_signal, maj_signal, flip_min, flip_maj, seed + 1)
X_val, y_val, g_val = map(lambda x: torch.tensor(x, dtype=torch.float32), val_data)
X_val_aug = augment_x(X_val, g_val.view(-1, 1))

# Define this globally for each process
def run_clipping_job(args):
    max_maj, max_min = args
    model = StratifiedLogisticRegression(dim)
    train_private_group_aware(
        model,
        dataloader=loader,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=wd,
        target_epsilon=eps,
        max_majority=max_maj,
        max_minority=max_min,
        total_samples=num_samples,
        delta=delta
    )
    with torch.no_grad():
        overall = mse(model, X_val, y_val, g_val)
        maj_mse, min_mse = group_mse(model, X_val, y_val, g_val)
        gap = abs(maj_mse - min_mse)
    print(f"Finished max_maj={max_maj}, max_min={max_min}, gap={gap:.4f}")
    return [max_maj, max_min, overall, maj_mse, min_mse, gap]

# Run this block in a single notebook cell
def run_parallel_grid():
    os.makedirs("save_results", exist_ok=True)
    log_file = "save_results/grid_search_clip_results.csv"

    clip_grid = [(m1, m2) for m1 in maj_clip_vals for m2 in min_clip_vals]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_clipping_job, clip_grid)

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["MaxMajority", "MaxMinority", "OverallMSE", "MajorityMSE", "MinorityMSE", "GroupGap"])
        writer.writerows(results)

# Run it
run_parallel_grid()
