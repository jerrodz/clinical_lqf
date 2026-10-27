import torch


def ordinal_targets(labels):
    """labels [B] int 0..4 -> [B,4] binary cumulative."""
    device = labels.device
    b = labels.shape[0]
    t = torch.zeros(b, 4, device=device)
    for j in range(4):
        t[:, j] = (labels > j).float()
    return t


def regression_target(labels: "torch.Tensor") -> "torch.Tensor":
    return labels.float()
