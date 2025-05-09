# calls data generation, training, and evaluation
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.logistic_regression import *
import data.data_generation as dg


def get_sets(num_datasets, data_params, seed = None):
    """
    data_params is a list of the inputs to the generate_data function
    remember to NOT set the seed for the internal one in data_params!!
    """
    if seed is not None:
        np.random.seed(seed)
    datasets = []
    for i in range(num_datasets):
        dataset = dg.generate_single_feature_bias(*data_params)
        datasets.append(dataset)
    return datasets # this is a list of tuples

def make_dataloader_flags(np_dataset, batch_size = 256, shuffle=True):
    """
    makes the dataloader for training with the flags
    """
    X, y, flags = np_dataset
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    flags = torch.tensor(flags, dtype=torch.long)       # 0 = majority, 1 = tail
    return DataLoader(TensorDataset(X, y, flags), batch_size=batch_size, shuffle=shuffle)

def split_sets(datasets, training_fraction, batch_size=128):
    """
    splits a set of datasets into training and validation, given fraction wanted for training
    """
    cutoff = int(len(datasets) * training_fraction)
    train_sets, val_sets = datasets[:cutoff], datasets[cutoff:]
    train_loaders = [make_dataloader_flags(ds, batch_size=batch_size, shuffle=True) for ds in train_sets]
    return train_loaders, val_sets

def augment_x(x, g):
    """
    augments input x with group membership g for stratified models
    x is tensor of shape (B, d) and g is tensor of shape (B, 1) or (B,)
    returns: tensor of shape (B, 2d)
    """
    g = g.view(-1, 1).float()
    x_aug = torch.cat([(1 - g) * x, g * x], dim=1) 
    return x_aug
