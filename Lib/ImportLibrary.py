import torch
from torch.autograd import grad
import numpy as np
import numpy.random as npr
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
from mpl_toolkits.mplot3d import Axes3D
from pyevtk.hl import gridToVTK
import gc
import os
from pyevtk.vtk import VtkGroup
import shutil
# from torch.utils.tensorboard import SummaryWriter
from pyDOE import lhs
import math
# import imageio
from scipy.spatial import KDTree
import random
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    # torch.set_default_dtype(torch.float)
    # torch.set_default_device(dev)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
if torch.backends.mps.is_available():
    print("MPS is available, running on MPS")
    dev = torch.device("mps")

else:
    print("CUDA not available, running on CPU")
mpl.rcParams['figure.dpi'] = 350
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.family'] = ['Times New Roman']
# mpl.rcParams['axes.labelsize'] = 50

# fix random seeds
npr.seed(503)
torch.manual_seed(503)
