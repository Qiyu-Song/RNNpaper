import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset
import netCDF4 as nc
import os, sys

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import copy


class data_struct:
    def __init__(self, ii, tt, name):
        self.input = torch.from_numpy(ii)
        self.target = torch.from_numpy(tt)
        self.experiment = name
        self.nstep = ii.shape[0]
        self.target_std = torch.std(self.target, axis=0, keepdim=True)  # original std
        self.normalized = False

    def normalize(self, mean_std):
        assert self.normalized == False
        self.target = self.target / mean_std
        self.normalized = True

    def denormalize(self, mean_std):
        assert self.normalized == True
        self.target = self.target * mean_std
        self.normalized = False

    def slice_data(self, seqlen, spinup, skip=1):
        # only for training, so should be normalized
        # assert self.normalized == True
        inputs_start_indices = np.arange(0, self.nstep - seqlen + 1, step=skip)
        targets_start_indices = inputs_start_indices + spinup
        sample_size = inputs_start_indices.size
        inputs_slice = torch.zeros(seqlen, sample_size, self.input.shape[-1])
        targets_slice = torch.zeros(seqlen - spinup, sample_size, self.target.shape[-1])
        for i in range(seqlen):
            inputs_slice[i, :, :] = copy.deepcopy(self.input[i + inputs_start_indices, :])
        for i in range(seqlen - spinup):
            targets_slice[i, :, :] = copy.deepcopy(self.target[i + targets_start_indices, :])
        print(f'Sliced data has {sample_size} samples.')
        return inputs_slice, targets_slice


def read_params_from_ssm(hidden_size, input_size=40, output_size=40, name_fix=None):
    # read in matrix
    if name_fix is None:
        allMtq = open(f'./allMtq_{hidden_size}.txt', 'r')
    else:
        allMtq = open(f'./allMtq_{name_fix}_{hidden_size}.txt', 'r')
    A = np.zeros((hidden_size, hidden_size), dtype=np.float64)
    B = np.zeros((hidden_size, input_size), dtype=np.float64)
    C = np.zeros((output_size, hidden_size), dtype=np.float64)
    for var in [A, B, C]:
        print(var.shape)
        for i in range(var.shape[0]):
            var[i, :] = allMtq.readline().split()
    return A, B, C
