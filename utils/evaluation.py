import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from utils.helper_functions import *
import pickle

def mse(model, features, labels, group_tags=None):
    X = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels,  dtype=torch.float32).view(-1, 1)
    g = (torch.zeros(len(X), 1) if group_tags is None
         else torch.as_tensor(group_tags, dtype=torch.float32).view(-1, 1))
    model.eval()
    with torch.no_grad():
        x_aug = augment_x(X, g)
        p = torch.sigmoid(model(x_aug))
        return F.mse_loss(p, y).item()

def _make_tensor(features, labels, group_tags):
    X = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels,  dtype=torch.float32).view(-1, 1)
    g = torch.as_tensor(group_tags, dtype=torch.bool).view(-1)
    return X, y, g

def group_mse(model, features, labels, group_tags):
    X, y, g = _make_tensor(features, labels, group_tags)
    g_f = g.float().view(-1, 1)
    model.eval()
    with torch.no_grad():
        x_aug = augment_x(X, g_f)
        p = torch.sigmoid(model(x_aug))
    majority = F.mse_loss(p[~g], y[~g]).item() if (~g).any() else float("nan")
    minority = F.mse_loss(p[g],  y[g]).item() if  g.any()  else float("nan")
    return [majority, minority]


# def MSE_model_comparison(private_model, nonprivate_model, validation_features, validation_labels):
#     """
#     get MSE diff for priv and nonpriv models on same data
#     """
#     private_MSE = MSE(private_model, validation_features, validation_labels)
#     nonprivate_MSE = MSE(nonprivate_model, validation_features, validation_labels)
#     return nonprivate_MSE - private_MSE

# def MSE_group_comparison(model, validation_features, validation_labels, validation_group_tags):
#     """
#     gets the difference between the group MSE and the average MSE on the tail groups
#     """
#     group_MSEs = group_MSE(model, validation_features, validation_labels, validation_group_tags)
#     tail_mean = sum(group_MSEs[1:]) / (len(group_MSEs) - 1)
#     diff = group_MSEs[0] - tail_mean
#     return diff

def plot_privacy_versus_accuracy(epsilon_list, MSE_list, figsize = (8,4), title = None):
    """
    assume we get a list of epsilons that were used to train the models, 
    and then a list of MSEs of the models evaluated on the same dataset
    """
    epsilon_list, MSE_list = zip(*sorted(zip(epsilon_list, MSE_list)))
    plt.figure(figsize = figsize)
    plt.plot(epsilon_list, MSE_list, color = "royalblue")
    plt.xlabel("Epsilon values")
    plt.ylabel("MSE for private models,\nevaluated on same validation set")
    if title is not None:
        plt.title(title)
    plt.grid(color = "white")
    plt.show()


def plot_MSE_summary(
    variable_array,
    all_var_MSEs,
    all_var_tail_MSEs,
    all_var_maj_MSEs,
    all_var_MSEs_p,
    all_var_tail_MSEs_p,
    all_var_maj_MSEs_p,
    variable_name,
    figsize=(12, 10)
):
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style='whitegrid')

    variable_array = np.array(variable_array)

    def compute_stats(arr):
        arr = [np.array(x) for x in arr]
        means = np.array([x.mean() for x in arr])
        stds = np.array([x.std() for x in arr])
        return means, stds

    means_np_total, stds_np_total = compute_stats(all_var_MSEs)
    means_p_total, stds_p_total = compute_stats(all_var_MSEs_p)

    means_np_tail, stds_np_tail = compute_stats(all_var_tail_MSEs)
    means_p_tail, stds_p_tail = compute_stats(all_var_tail_MSEs_p)

    means_np_maj, stds_np_maj = compute_stats(all_var_maj_MSEs)
    means_p_maj, stds_p_maj = compute_stats(all_var_maj_MSEs_p)

    np_gap = means_np_tail - means_np_maj
    p_gap = means_p_tail - means_p_maj

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # set the ylim on all the plots to be the same
    all_values = np.concatenate([
        means_np_total - stds_np_total, means_np_total + stds_np_total,
        means_p_total - stds_p_total, means_p_total + stds_p_total,
        means_np_tail - stds_np_tail, means_np_tail + stds_np_tail,
        means_p_tail - stds_p_tail, means_p_tail + stds_p_tail,
        means_np_maj - stds_np_maj, means_np_maj + stds_np_maj,
        means_p_maj - stds_p_maj, means_p_maj + stds_p_maj,
        np_gap, p_gap
    ])
    y_min, y_max = all_values.min() - 0.01, all_values.max() + 0.01


    # total MSE on top left
    axs[0, 0].errorbar(variable_array, means_np_total, yerr=stds_np_total, fmt='-o', label='Nonprivate', capsize=4)
    axs[0, 0].errorbar(variable_array, means_p_total, yerr=stds_p_total, fmt='-o', label='Private', capsize=4)
    axs[0, 0].set_title(f'Total MSE vs {variable_name.title()}')
    axs[0, 0].set_xlabel(f'{variable_name.title()}')
    axs[0, 0].set_ylabel('Total MSE')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(y_min, y_max)

    # gaps on top right
    axs[0, 1].plot(variable_array, np_gap, '-o', label='Nonprivate')
    axs[0, 1].plot(variable_array, p_gap, '-o', label='Private')
    axs[0, 1].set_title(f'Tail - Majority MSE Gap across {variable_name.title()}')
    axs[0, 1].set_xlabel(f'{variable_name.title()}')
    axs[0, 1].set_ylabel('MSE Gap (Tail − Majority)')
    axs[0, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[0, 1].legend()
    axs[0, 1].set_ylim(y_min, y_max)

    # majority bottom left
    axs[1, 0].errorbar(variable_array, means_np_maj, yerr=stds_np_maj, fmt='-o', label='Nonprivate', capsize=4)
    axs[1, 0].errorbar(variable_array, means_p_maj, yerr=stds_p_maj, fmt='-o', label='Private', capsize=4)
    axs[1, 0].set_title(f'Majority MSE vs {variable_name.title()}')
    axs[1, 0].set_xlabel(f'{variable_name.title()}')
    axs[1, 0].set_ylabel('Majority MSE')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(y_min, y_max)

    # tail bottom right
    axs[1, 1].errorbar(variable_array, means_np_tail, yerr=stds_np_tail, fmt='-o', label='Nonprivate', capsize=4)
    axs[1, 1].errorbar(variable_array, means_p_tail, yerr=stds_p_tail, fmt='-o', label='Private', capsize=4)
    axs[1, 1].set_title(f'Tail MSE vs {variable_name.title()}')
    axs[1, 1].set_xlabel(f'{variable_name.title()}')
    axs[1, 1].set_ylabel('Tail MSE')
    axs[1, 1].legend()
    axs[1, 1].set_ylim(y_min, y_max)

    plt.suptitle(f"MSE Comparisons for Logistic Regression on Data Varying {variable_name.title()}", fontsize = 15)
    plt.tight_layout()  # leave room at the top for suptitle
    plt.show()

def plot_summary_from_csvs(varname, var_values, varname_title, epsilon=1.0, save_dir="save_results", figsize=(12, 10)):

    plt.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style='whitegrid')

    means = {
        "overall_nonprivate": [],
        "overall_private": [],
        "minority_nonprivate": [],
        "minority_private": [],
        "majority_nonprivate": [],
        "majority_private": [],
    }
    stds = {k: [] for k in means}

    # get the csvs I made in the parallel sweep experiments
    for val in var_values:
        if varname == "samplesize":
            csv_path = os.path.join(save_dir, f"{varname}_{val}_eps{epsilon}.csv")   
        else: 
            csv_path = os.path.join(save_dir, f"{varname}_{val:.2f}_eps{epsilon}.csv")
        df = pd.read_csv(csv_path)
        for group in ["overall", "minority", "majority"]:
            for kind in ["nonprivate", "private"]:
                row = df[(df["group"] == group) & (df["privacy"] == kind)].iloc[0]
                means[f"{group}_{kind}"].append(row["mean"])
                stds[f"{group}_{kind}"].append(np.sqrt(row["variance"]))

    var_array = np.array(var_values)
    def arr(k): return np.array(means[k])
    def err(k): return np.array(stds[k])

    # get MSE gaps
    np_gap = arr("minority_nonprivate") - arr("majority_nonprivate")
    p_gap = arr("minority_private") - arr("majority_private")

    # get global axis limits
    all_vals = np.concatenate([
        arr("overall_nonprivate") - err("overall_nonprivate"), arr("overall_nonprivate") + err("overall_nonprivate"),
        arr("overall_private") - err("overall_private"), arr("overall_private") + err("overall_private"),
        arr("minority_nonprivate") - err("minority_nonprivate"), arr("minority_nonprivate") + err("minority_nonprivate"),
        arr("minority_private") - err("minority_private"), arr("minority_private") + err("minority_private"),
        arr("majority_nonprivate") - err("majority_nonprivate"), arr("majority_nonprivate") + err("majority_nonprivate"),
        arr("majority_private") - err("majority_private"), arr("majority_private") + err("majority_private"),
        np_gap, p_gap
    ])
    y_min, y_max = all_vals.min() - 0.01, all_vals.max() + 0.01

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # overall mse
    axs[0, 0].errorbar(var_array, arr("overall_nonprivate"), yerr=err("overall_nonprivate"), fmt='-o', label='Nonprivate', capsize=4, color = "royalblue")
    axs[0, 0].errorbar(var_array, arr("overall_private"), yerr=err("overall_private"), fmt='-o', label='Private', capsize=4, color = "mediumaquamarine")
    axs[0, 0].set_title(f'Total MSE vs {varname_title.title()}')
    axs[0, 0].set_xlabel(varname_title.title())
    axs[0, 0].set_ylabel('Total MSE')
    axs[0, 0].legend()
    axs[0, 0].set_ylim(y_min, y_max)
    axs[0, 0].axhline(0.25, color='red', linestyle='--', linewidth=1)

    # mse gap
    axs[0, 1].plot(var_array, np_gap, '-o', label='Nonprivate', color = "royalblue")
    axs[0, 1].plot(var_array, p_gap, '-o', label='Private', color = "mediumaquamarine")
    axs[0, 1].set_title(f'Minority - Majority MSE Gap across {varname_title.title()}')
    axs[0, 1].set_xlabel(varname_title.title())
    axs[0, 1].set_ylabel('MSE Gap (Minority − Majority)')
    axs[0, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[0, 1].legend()
    axs[0, 1].set_ylim(y_min, y_max)
    # axs[0, 1].axhline(0.25, color='red', linestyle='--', linewidth=1)

    # maj mse
    axs[1, 0].errorbar(var_array, arr("majority_nonprivate"), yerr=err("majority_nonprivate"), fmt='-o', label='Nonprivate', capsize=4, color = "royalblue")
    axs[1, 0].errorbar(var_array, arr("majority_private"), yerr=err("majority_private"), fmt='-o', label='Private', capsize=4, color = "mediumaquamarine")
    axs[1, 0].set_title(f'Majority MSE vs {varname_title.title()}')
    axs[1, 0].set_xlabel(varname_title.title())
    axs[1, 0].set_ylabel('Majority MSE')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(y_min, y_max)
    axs[1, 0].axhline(0.25, color='red', linestyle='--', linewidth=1)

    axs[1, 1].errorbar(var_array, arr("minority_nonprivate"), yerr=err("minority_nonprivate"), fmt='-o', label='Nonprivate', capsize=4, color = "royalblue")
    axs[1, 1].errorbar(var_array, arr("minority_private"), yerr=err("minority_private"), fmt='-o', label='Private', capsize=4, color = "mediumaquamarine")
    axs[1, 1].set_title(f'Minority MSE vs {varname_title.title()}')
    axs[1, 1].set_xlabel(varname_title.title())
    axs[1, 1].set_ylabel('Minority MSE')
    axs[1, 1].legend()
    axs[1, 1].set_ylim(y_min, y_max)
    axs[1, 1].axhline(0.25, color='red', linestyle='--', linewidth=1)

    plt.suptitle(f"MSE Comparisons for Logistic Regression on Data Varying {varname_title.title()}", fontsize=15)
    plt.tight_layout()
    plt.show()

def run_anova_gap(varname, var_values, epsilon = 1.0, save_dir = "save_results"):
    rows = []

    for val in var_values:
        if varname == "samplesize":
            pkl_path = os.path.join(save_dir, f"{varname}_{val}_eps{epsilon}.pkl")   
        else: 
            pkl_path = os.path.join(save_dir, f"{varname}_{val:.2f}_eps{epsilon}.pkl")
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)

        for priv in ["nonprivate", "private"]:
            minority_vals = results[f"minority_{priv}"]
            majority_vals = results[f"majority_{priv}"]
            for run_id, (min_val, maj_val) in enumerate(zip(minority_vals, majority_vals)):
                gap = min_val - maj_val
                rows.append({
                    varname: val,
                    "privacy": priv,
                    "gap": gap,
                    "run": run_id
                })

    gap_df = pd.DataFrame(rows)
    gap_df["privacy"] = gap_df["privacy"].astype("category")
    gap_df[varname] = gap_df[varname].astype("category")

    formula = f"gap ~ C(privacy) + C({varname}) + C(privacy):C({varname})"
    model = ols(formula, data=gap_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table

def print_anova_table(anova_df, varname_title):
    title = f"Two-way ANOVA Results for Group MSE Gap by Privacy and {varname_title.title()}"
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    print(anova_df.to_string(index=True))

def plot_gradient_norms(setting_index, settings, save_dir="save_results", other = None):
    path = os.path.join(save_dir, f"grads_run_{setting_index}.pkl")
    if other is not None:
        path = os.path.join(save_dir, f"{other}_grads_run_{setting_index}.pkl")
    with open(path, "rb") as f:
        log = pickle.load(f)

    norm_data = {0: {}, 1: {}}  # {group: {epoch: [norms]}}
    
    for entry in log:
        group = entry["tail_tag"]
        epoch = entry["epoch"]
        grad_vec = torch.cat([p.flatten() for p in entry["grads"]])
        norm = grad_vec.norm().item()

        if epoch not in norm_data[group]:
            norm_data[group][epoch] = []
        norm_data[group][epoch].append(norm)

    def compute_stats(group_data):
        epochs = sorted(group_data.keys())
        means = [np.mean(group_data[e]) for e in epochs]
        stds  = [np.std(group_data[e]) for e in epochs]
        return epochs, np.array(means), np.array(stds)

    epochs_0, mean_0, std_0 = compute_stats(norm_data[0])
    epochs_1, mean_1, std_1 = compute_stats(norm_data[1])

    plt.figure(figsize=(8, 5))
    plt.errorbar(epochs_0, mean_0, yerr=std_0, fmt='-o', capsize=4, label='Majority (0)', color = "royalblue")
    plt.errorbar(epochs_1, mean_1, yerr=std_1, fmt='-o', capsize=4, label='Minority (1)', color = "mediumaquamarine")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm (L2)")
    plt.axhline(1.0, color = "red")
    setting = settings[setting_index]
    min_fraction = setting["min_fraction"]
    min_signal = setting["min_signal"]
    flip_min = setting["flip_min"]
    plt.title(f"Per-Group Gradient Norms for Setting Minority\n Fraction = {min_fraction}, Bias = {min_signal}, Noise = {flip_min}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_group_norm_averages(setting_index, settings, save_dir="save_results", other=None):
    path = os.path.join(save_dir, f"grads_run_{setting_index}.pkl")
    if other is not None:
        path = os.path.join(save_dir, f"{other}_grads_run_{setting_index}.pkl")

    with open(path, "rb") as f:
        log = pickle.load(f)

    group_norms = {0: [], 1: []}

    for entry in log:
        group = entry["tail_tag"]  # 0 for majority, 1 for minority
        grad_vec = torch.cat([p.flatten() for p in entry["grads"]])
        norm = grad_vec.norm().item()
        group_norms[group].append(norm)

    avg_maj = np.mean(group_norms[0]) if group_norms[0] else float("nan")
    avg_min = np.mean(group_norms[1]) if group_norms[1] else float("nan")

    print(f"Setting {setting_index}:")
    print(f"  Avg Majority Norm: {avg_maj:.4f}")
    print(f"  Avg Minority Norm: {avg_min:.4f}")
    print(f"  Ratio (Minority / Majority): {avg_min / avg_maj:.2f}" if avg_maj > 0 else "  Undefined ratio")

    return avg_maj, avg_min

def plot_avg_gradient_norms_vs_bias(settings2, save_dir="save_results", other=None):
    avg_majorities = []
    avg_minorities = []
    min_biases = []

    for i, setting in enumerate(settings2):
        min_bias = setting["min_signal"]  # for x-axis
        min_biases.append(min_bias)

        path = os.path.join(save_dir, f"grads_run_{i}.pkl")
        if other is not None:
            path = os.path.join(save_dir, f"{other}_grads_run_{i}.pkl")

        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            avg_majorities.append(float("nan"))
            avg_minorities.append(float("nan"))
            continue

        with open(path, "rb") as f:
            log = pickle.load(f)

        group_norms = {0: [], 1: []}

        for entry in log:
            group = entry["tail_tag"]
            grad_vec = torch.cat([p.flatten() for p in entry["grads"]])
            norm = grad_vec.norm().item()
            group_norms[group].append(norm)

        avg_maj = np.mean(group_norms[0]) if group_norms[0] else float("nan")
        avg_min = np.mean(group_norms[1]) if group_norms[1] else float("nan")
        avg_majorities.append(avg_maj)
        avg_minorities.append(avg_min)

    plt.figure(figsize=(8, 5))
    plt.plot(min_biases, avg_majorities, '-o', label="Majority", color="royalblue")
    plt.plot(min_biases, avg_minorities, '-o', label="Minority", color="mediumaquamarine")
    plt.xlabel("min_bias")
    plt.ylabel("Average Gradient Norm (L2)")
    plt.title("Avg Gradient Norm vs min_bias")
    plt.axhline(1.0, color = "red")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_gap_traces(min_biases, save_dir="save_results"):
    
    group_csv = os.path.join(save_dir, "grid_search_across_biases.csv")
    group_df = pd.read_csv(group_csv)

    combos = [(0.5, 1.5), (0.5, 2.5)]
    group_gaps = {combo: [] for combo in combos}

    for combo in combos:
        max_maj, max_min = combo
        for mb in min_biases:
            row = group_df[(group_df["MinBias"] == mb) &
                           (group_df["MaxClip"] == max_maj) &
                           (group_df["MinClip"] == max_min)]
            if not row.empty:
                group_gaps[combo].append(row.iloc[0]["GroupGap"])
            else:
                group_gaps[combo].append(np.nan)

    std_gaps = []
    for mb in min_biases:
        df = pd.read_csv(os.path.join(save_dir, f"minbias_{mb:.2f}_eps1.0.csv"))
        maj = df[(df["group"] == "majority") & (df["privacy"] == "private")].iloc[0]["mean"]
        min_ = df[(df["group"] == "minority") & (df["privacy"] == "private")].iloc[0]["mean"]
        std_gaps.append(min_ - maj)

    np_gaps = []
    for mb in min_biases:
        dim = 200
        n_train = 2500
        flip_min = 0.2
        flip_maj = 0.05
        maj_bias = -2.0
        seed = 13
        lr = 0.1
        wd = 1e-4
        num_epochs = 30

        data = generate_data_weight_and_bias(n_train, dim, 0.1, mb, maj_bias, flip_min, flip_maj, seed)
        loader = make_dataloader_flags(data, batch_size=256)

        val_data = generate_data_weight_and_bias(500, dim, 0.1, mb, maj_bias, flip_min, flip_maj, seed+1)
        X_val, y_val, g_val = map(lambda x: torch.tensor(x, dtype=torch.float32), val_data)
        X_val_aug = augment_x(X_val, g_val.view(-1, 1))

        model = StratifiedLogisticRegression(dim)
        train_nonprivate(model, loader, num_epochs, lr, wd)

        with torch.no_grad():
            maj_mse, min_mse = group_mse(model, X_val, y_val, g_val)
            np_gaps.append(min_mse - maj_mse)

    plt.figure(figsize=(8, 6))
    var_array = np.array(min_biases)
    plt.plot(var_array, std_gaps, '-o', label="StandardPrivate", color="mediumaquamarine")
    plt.plot(var_array, np_gaps, '-o', label="NonPrivate", color="royalblue")
    plt.plot(var_array, group_gaps[(0.5, 1.5)], '-o', label="GroupAware (0.5, 1.5)", color="red")
    plt.plot(var_array, group_gaps[(0.5, 2.5)], '-o', label="GroupAware (0.5, 2.5)", color="purple")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Minority Bias")
    plt.ylabel("MSE Gap (Minority − Majority)")
    plt.title("Comparison of Group Gaps Across Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()