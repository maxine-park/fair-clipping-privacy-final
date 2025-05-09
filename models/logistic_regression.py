import torch
import torch.nn as nn


class StratifiedLogisticRegression(nn.Module):
    """
    Call signature changed to model(x_aug) so remember to augment x before feeding in
    Learnable weights for each group as well as learnable biases
    Group flags are assumed to be 1 (minority) and 0 (majority)
    Returns logits
    """
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(2 * d, 1, bias=False)  # linear layer with no bias term, takes 2-dim input and makes 1-dim output
        self.bias_maj = nn.Parameter(torch.zeros(1)) # learnable bias for majority
        self.bias_min = nn.Parameter(torch.zeros(1)) # learnable bias for minority

    def forward(self, x_augmented):
        # for each sample, first d is the feature and last d are all 0s (for min group) or all 1s (maj group)
        d = x_augmented.shape[1] // 2 # calculate d
        g = (x_augmented[:, d:] != 0).any(dim=1).float().view(-1, 1)  # infer group from second half
        logits = self.w(x_augmented) + (1 - g) * self.bias_maj + g * self.bias_min
        return logits