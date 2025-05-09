from data.data_generation import *
from models.logistic_regression import *
from training.training import *
from utils.evaluation import *
from utils.helper_functions import *

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils

import pickle
import os
import time
start = time.time()

num_samples = 30000
sample_dim = 200
batch_size = 256
num_runs = 10
val_size = 2000

min_fracs = [0.05, 0.1, 0.2, 0.25, 0.3]
min_bias = 3.0
maj_bias = -2.0
flip_min = 0.2
flip_maj = 0.05

lr = 0.1
weight_decay = 1e-4
max_grad_norm = 1.0
num_epochs = 200
target_epsilon = 1.0

generator = generate_data_weight_and_bias

minfrac_results = {
    "overall_nonprivate": [[] for _ in min_fracs],
    "overall_private": [[] for _ in min_fracs],
    "minority_nonprivate": [[] for _ in min_fracs],
    "minority_private": [[] for _ in min_fracs],
    "majority_nonprivate": [[] for _ in min_fracs],
    "majority_private": [[] for _ in min_fracs],
}

for i, min_frac in enumerate(min_fracs):
    print(f"\n== minority fraction {min_frac}")
    delta = 1 / (num_samples **2)
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        train_data = generator(num_samples, sample_dim, min_frac, min_bias, maj_bias, flip_min, flip_maj)
        val_data = generator(val_size, sample_dim, min_frac, min_bias, maj_bias, flip_min, flip_maj)

        X_val, y_val, z_val = val_data # unpack into its parts
        train_loader = make_dataloader_flags(train_data, batch_size = batch_size)

        np_model = StratifiedLogisticRegression(sample_dim)
        train_nonprivate(np_model, train_loader, num_epochs = num_epochs, lr = lr, weight_decay = weight_decay)
        minfrac_results["overall_nonprivate"][i].append(mse(np_model, X_val, y_val, z_val))
        majority, minority = group_mse(np_model, X_val, y_val, z_val)
        minfrac_results["minority_nonprivate"][i].append(minority)
        minfrac_results["majority_nonprivate"][i].append(majority)

        p_model = StratifiedLogisticRegression(sample_dim)
        train_private_standard(p_model, train_loader, num_epochs = num_epochs, lr = lr, weight_decay = weight_decay, target_epsilon = target_epsilon, target_delta = delta, max_grad_norm = max_grad_norm)
        minfrac_results["overall_private"][i].append(mse(p_model, X_val, y_val, z_val))
        majority, minority = group_mse(p_model, X_val, y_val, z_val)
        minfrac_results["minority_private"][i].append(minority)
        minfrac_results["majority_private"][i].append(majority)

print(f"All minority fraction experiments complete.")

# now save all the stuff with pickle 
os.makedirs("save_results", exist_ok=True)
pkl_name = f"minfrac_sweep_eps{target_epsilon}.pkl"
with open(f"save_results/{pkl_name}", "wb") as f:
    pickle.dump(minfrac_results, f)
print(f"Saved raw result lists to save_results/{pkl_name}")

# ---- Compute summary CSV (mean + variance) ----
minfrac_summary_rows = []

for i, min_frac in enumerate(min_fracs):
    for metric in ["overall", "minority", "majority"]:
        for kind in ["nonprivate", "private"]:
            values = minfrac_results[f"{metric}_{kind}"][i]
            mean = np.mean(values)
            var = np.var(values)
            minfrac_summary_rows.append({
                "min_frac": min_frac,
                "privacy": kind,
                "group": metric,
                "mean": mean,
                "variance": var
            })

minfrac_summary_df = pd.DataFrame(minfrac_summary_rows)
csv_name = f"minfrac_analysis_eps{target_epsilon}.csv"
minfrac_summary_df.to_csv(f"save_results/{csv_name}", index=False)
print(f"Saved summary CSV to save_results/{csv_name}")
print(f"Total time: {time.time() - start:.2f} seconds")