import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_generation import *
from models.logistic_regression import *
from training.training import *
from utils.evaluation import *
from utils.helper_functions import *

import pandas as pd
import numpy as np
import torch
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--flip_min", type=float, required=True)
args = parser.parse_args()
flip_min = args.flip_min

num_samples = 30000
sample_dim = 200
batch_size = 256
num_runs = 10
val_size = 1000

min_frac = 0.1
min_bias = 3.0
maj_bias = -2.0
flip_maj = 0.05

lr = 0.1
weight_decay = 1e-4
max_grad_norm = 1.0
num_epochs = 100
target_epsilon = 1.0
delta = 1 / (num_samples ** 2)

generator = generate_data_weight_and_bias

results = {
    "overall_nonprivate": [],
    "overall_private": [],
    "minority_nonprivate": [],
    "minority_private": [],
    "majority_nonprivate": [],
    "majority_private": [],
}

for run in range(num_runs):
    print(f"Flip min {flip_min}, Run {run + 1}/{num_runs}")
    train_data = generator(num_samples, sample_dim, min_frac, min_bias, maj_bias, flip_min, flip_maj)
    val_data = generator(val_size, sample_dim, min_frac, min_bias, maj_bias, flip_min, flip_maj)
    X_val, y_val, z_val = val_data
    train_loader = make_dataloader_flags(train_data, batch_size=batch_size)

    np_model = StratifiedLogisticRegression(sample_dim)
    train_nonprivate(np_model, train_loader, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay)
    results["overall_nonprivate"].append(mse(np_model, X_val, y_val, z_val))
    maj, mino = group_mse(np_model, X_val, y_val, z_val)
    results["minority_nonprivate"].append(mino)
    results["majority_nonprivate"].append(maj)

    p_model = StratifiedLogisticRegression(sample_dim)
    train_private_standard(p_model, train_loader, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, target_epsilon=target_epsilon, target_delta=delta, max_grad_norm=max_grad_norm)
    results["overall_private"].append(mse(p_model, X_val, y_val, z_val))
    maj, mino = group_mse(p_model, X_val, y_val, z_val)
    results["minority_private"].append(mino)
    results["majority_private"].append(maj)

os.makedirs("save_results", exist_ok=True)
pkl_path = f"save_results/minflip_{flip_min:.2f}_eps{target_epsilon}.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)

summary_rows = []
for group in ["overall", "minority", "majority"]:
    for kind in ["nonprivate", "private"]:
        vals = results[f"{group}_{kind}"]
        summary_rows.append({
            "flip_min": flip_min,
            "privacy": kind,
            "group": group,
            "mean": np.mean(vals),
            "variance": np.var(vals)
        })

summary_df = pd.DataFrame(summary_rows)
csv_path = f"save_results/minflip_{flip_min:.2f}_eps{target_epsilon}.csv"
summary_df.to_csv(csv_path, index=False)

print(f"Completed flip_min = {flip_min}")
