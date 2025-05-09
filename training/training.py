import torch.nn as nn, torch.optim as optim
from opacus import PrivacyEngine
from utils.helper_functions import *

def _as_float(x):          # one‑liner to avoid repeating .float()
    return x.float()

def train_nonprivate(model, dataloader, num_epochs = 100, lr = 0.1, weight_decay = 1e-4):
    """
    normal nonprivate training
    """
    crit, opt = nn.BCEWithLogitsLoss(), optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(num_epochs):
        for xb, yb, gb in dataloader:                
            xb = _as_float(xb)
            yb = _as_float(yb).view(-1, 1)
            gb = gb.float().view(-1, 1)               

            opt.zero_grad()
            x_aug = augment_x(xb, gb) 
            loss = crit(model(x_aug), yb)
            loss.backward()
            opt.step()


def train_private_standard(model, dataloader, num_epochs,
                           lr, weight_decay, target_epsilon, target_delta,
                           max_grad_norm):
    """
    normal private training with standard opacus uniform clipping DP-SGD
    """
    crit, opt = nn.BCEWithLogitsLoss(), optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    engine = PrivacyEngine()

    model, opt, dataloader = engine.make_private_with_epsilon(
        module         = model,
        optimizer      = opt,
        data_loader    = dataloader,
        epochs         = num_epochs,
        target_epsilon = target_epsilon,
        target_delta   = target_delta,
        max_grad_norm  = max_grad_norm,
    )

    model.train()
    for _ in range(num_epochs):
        for xb, yb, gb in dataloader:
            xb = _as_float(xb)
            yb = _as_float(yb).view(-1, 1)
            gb = gb.float().view(-1, 1)

            opt.zero_grad()
            x_aug = augment_x(xb, gb)
            loss = crit(model(x_aug), yb)

            loss.backward()
            opt.step()

    print(f"(ε = {engine.get_epsilon(target_delta):.2f}, δ = {target_delta})")


def track_per_sample_grads_nonprivate(model, loader, num_epochs, lr, wd, sampling_rate):
    """
    nonprivate training with per-sample gradient tracking by subgroup every sampling_rate epochs
    """
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss(reduction='none')
    log = []
        # by the end of this, will have num_epochs/sampling_rate samples
        # each "sample" will be 2 dictionaries, one for nonpriv and one for priv
        # each dictionary has the epoch number, tail_tag, and the gradients as a lsit
    model.train()  # enable grad sampling
    for epoch in range(num_epochs):
        for xb, yb, flagb in loader:
            xb, yb = _as_float(xb), _as_float(yb).view(-1, 1)
            flagb_f = flagb.float().view(-1, 1)

            if (epoch+1) % sampling_rate == 0:
                for grp in (0, 1):
                    mask = flagb == grp
                    if mask.any():
                        opt.zero_grad()
                        xb_mask_aug = augment_x(xb[mask], flagb_f[mask])
                        loss = crit(model(xb_mask_aug), yb[mask]).mean()
                        loss.backward()
                        log.append({'epoch': epoch+1, 
                                    'tail_tag': grp,
                                    'grads': [p.grad.detach().clone() for p in model.parameters()]})
                    print(f"finished for group {grp} in epoch {epoch + 1}")
            xb_aug = augment_x(xb, flagb_f)
            opt.zero_grad()
            crit(model(xb_aug), yb).mean().backward()
            opt.step()
    return log



