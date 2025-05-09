import os
import sys
import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_generation import generate_data_weight_and_bias
from models.logistic_regression import StratifiedLogisticRegression
from training.training import track_per_sample_grads_nonprivate
from utils.helper_functions import make_dataloader_flags


def run_tracking(setting_index):
    # settings = [
    #     {"min_fraction": 0.05, "min_signal": 0.5, "flip_min": 0.15},
    #     {"min_fraction": 0.05, "min_signal": 0.5, "flip_min": 0.2},
    #     {"min_fraction": 0.05, "min_signal": 1.0, "flip_min": 0.15},
    #     {"min_fraction": 0.05, "min_signal": 1.0, "flip_min": 0.2},
    #     {"min_fraction": 0.1,  "min_signal": 0.5, "flip_min": 0.15},
    #     {"min_fraction": 0.1,  "min_signal": 0.5, "flip_min": 0.2},
    #     {"min_fraction": 0.1,  "min_signal": 1.0, "flip_min": 0.15},
    #     {"min_fraction": 0.1,  "min_signal": 1.0, "flip_min": 0.2},
    # ]
    settings = [
        {"min_fraction": 0.1, "min_signal": -0.5, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 0.5, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 1.0, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 1.5, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 2.0, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 2.5, "flip_min": 0.2},
        {"min_fraction": 0.1, "min_signal": 3.0, "flip_min": 0.2},
        
    ]
    

    setting = settings[setting_index]
    min_frac = setting["min_fraction"]
    min_signal = setting["min_signal"]
    flip_min = setting["flip_min"]

    maj_signal = -2.0
    flip_maj = 0.05
    dim = 200
    n_train = 30000
    seed = 13

    train_data = generate_data_weight_and_bias(
        n_train, dim, min_frac,
        min_signal, maj_signal,
        flip_min, flip_maj,
        seed=seed
    )
    print(f"Finished generating data for setting {setting_index}")

    loader = make_dataloader_flags(train_data, batch_size=256, shuffle=True)
    print(f"Prepared DataLoader for setting {setting_index}")

    model = StratifiedLogisticRegression(dim)
    log = track_per_sample_grads_nonprivate(model, loader, 100, lr=0.1, wd=1e-4, sampling_rate=10)

    os.makedirs("save_results", exist_ok=True)
    with open(f"save_results/bias_grads_run_{setting_index}.pkl", "wb") as f:
        pickle.dump(log, f)
    print(f"Saved gradient log for setting {setting_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setting_index", type=int, help="Index of the setting to run (0 through 7)")
    args = parser.parse_args()

    run_tracking(args.setting_index)
