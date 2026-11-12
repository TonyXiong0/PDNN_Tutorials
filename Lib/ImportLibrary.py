"""Shared device, random seed, and plotting setup for the tutorials."""

import matplotlib as mpl
import numpy as np
import torch


def configure_notebook(seed=503):
    """Set the notebook random seed and plotting defaults, then select a device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    mpl.rcParams.update({
        "figure.dpi": 120,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    print(f"Using PyTorch device: {device}")
    return device
