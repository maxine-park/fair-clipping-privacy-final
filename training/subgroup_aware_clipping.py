import torch
from torch import nn, optim
from utils.helper_functions import *
from training.training import *
import torch
from torch import nn, optim
from torch.nn import functional as F
from opacus.accountants.rdp import RDPAccountant

def find_noise_for_target_epsilon(target_epsilon, sample_rate, steps, delta, tol=0.01):
    low, high = 0.5, 20.0
    for _ in range(50):
        mid = (low + high) / 2
        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = accountant.get_epsilon(delta)
        if abs(eps - target_epsilon) < tol:
            return mid
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return high  # fallback if exact match not found

def train_private_group_aware(model, dataloader, num_epochs,
                                  lr, weight_decay, target_epsilon,
                                  max_majority, max_minority,
                                  total_samples, delta):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    accountant = RDPAccountant()
    crit = nn.BCEWithLogitsLoss(reduction="none")

    param_shapes = [p.shape for p in model.parameters()]
    param_sizes = [p.numel() for p in model.parameters()]
    total_param_size = sum(param_sizes)

    sample_rate = next(iter(dataloader))[0].shape[0] / total_samples
    steps = num_epochs * len(dataloader)
    noise_multiplier = find_noise_for_target_epsilon(target_epsilon, sample_rate, steps, delta)
    print(f"Calibrated noise_multiplier = {noise_multiplier:.4f} for ε = {target_epsilon}")

    for epoch in range(num_epochs):
        for xb, yb, gb in dataloader:
            xb = xb.float()
            yb = yb.float()
            gb = gb.float().view(-1, 1)
            x_aug = augment_x(xb, gb)

            per_sample_grads = []

            for i in range(xb.shape[0]):
                optimizer.zero_grad()
                out = model(x_aug[i].unsqueeze(0))
                loss = F.binary_cross_entropy_with_logits(out, yb[i].view(1, 1), reduction='sum')
                grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False, create_graph=False) # backprop is done manually for each sample
                flat = torch.cat([g.contiguous().view(-1) for g in grads])
                per_sample_grads.append(flat)

            per_sample_grads = torch.stack(per_sample_grads)

            norms = per_sample_grads.norm(p=2, dim=1)
            clip_bounds = torch.where(gb.view(-1) == 1, max_minority, max_majority).to(norms)
            scale = (clip_bounds / (norms + 1e-6)).clamp(max=1.0).view(-1, 1)
            clipped_grads = per_sample_grads * scale

            grad_mean = clipped_grads.mean(dim=0)
            noise_std = noise_multiplier * max(max_majority, max_minority) / xb.shape[0]
            noisy_grad = grad_mean + torch.randn_like(grad_mean) * noise_std

            offset = 0
            for p, sz in zip(model.parameters(), param_sizes):
                p.grad = noisy_grad[offset:offset + sz].view(p.shape).detach()
                offset += sz

            optimizer.step()
            optimizer.zero_grad()
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        if (epoch + 1) % 10 == 0:
            print(f"done with epoch {epoch+1}/{num_epochs}")

    epsilon = accountant.get_epsilon(delta)
    print(f"(ε = {epsilon:.2f}, δ = {delta})")