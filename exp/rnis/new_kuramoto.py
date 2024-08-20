from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import torch
from torch.utils.data import Dataset
import numpy as np

import os
import copy

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import multivariate_normal



class KuramotoModel(Dataset):
    def __init__(self, steps, dt, sz, groups, coupling, sample_step=5):
        """
        TODO
        
        :param size_list: List of initial state sizes.
        :param steps: Number of steps to run (including the starting point).
        :param dt: Step size.
        :param interval: Sampling interval.
        :param sigma: Standard deviation of noise.
        :param rho: Correlation coefficient of noise.
        """
        self.steps = steps
        self.dt = dt
        self.coupling = coupling
        self.sz = sz
        self.groups = groups
        self.obj_matrix = np.zeros([sz,sz])
        self.group_matrix = np.zeros([sz, groups])
        for i in range(sz//groups):
            for j in range(sz//groups):
                for k in range(groups):
                    self.obj_matrix[i + k * sz // groups, j + k * sz // groups] = 1
                    self.group_matrix[i + k * sz // groups, k] = 1
                    self.group_matrix[j + k * sz // groups, k] = 1
        self.obj_matrix = self.obj_matrix - np.eye(sz) * self.obj_matrix
        self.omegas = np.random.randn(sz)
        self.input, self.output, _, _ = self.simulate_oneserie(sample_step=5)


    def one_step(self, thetas, noise_p=0):
        thetas_unsqueezed = thetas[np.newaxis, :]
        ii = np.tile(thetas_unsqueezed, (len(thetas), 1))
        jj = ii.T
        dff = jj - ii
        sindiff = np.sin(dff)
        mult = self.coupling * self.obj_matrix @ sindiff
        dia =  np.diagonal(mult)
        noise = np.random.rand(self.sz) * noise_p
        thetas = self.dt * (self.omegas + dia + noise) + thetas
        return thetas


    def simulate_oneserie(self, batch_size=1, sample_step=5, noise_p=0):
        """
        TODO
        """

        time_steps = self.steps * sample_step

        states = np.zeros([batch_size, self.sz, time_steps // sample_step])
        centers = np.zeros([batch_size, 2*self.groups, time_steps // sample_step])
        for i in range(batch_size):
            thetas = np.random.rand(self.sz) * 2 * np.pi
            for t in range(time_steps):
                thetas = self.one_step(thetas, noise_p)
                if t % sample_step == 0:
                    states[i, :, t // sample_step] = thetas
                    cos_ccs = (np.cos(thetas) @ self.group_matrix) * self.groups / self.sz
                    sin_ccs = (np.sin(thetas) @ self.group_matrix) * self.groups / self.sz
                    centers[i, :, t // sample_step] = np.concatenate((sin_ccs, cos_ccs), axis=0) 
        state_s = np.sin(states[:, :, :-1])
        state_next_s = np.sin(states[:, :, 1:])
        state_c = np.cos(states[:, :, :-1])
        state_next_c = np.cos(states[:, :, 1:])
        state = np.concatenate((state_s, state_c), axis=1)
        state_next = np.concatenate((state_next_s, state_next_c), axis=1)
        latent = centers[:, :, :-1]
        latent_next = centers[:, :, 1:]
        return state.reshape(2*self.sz, -1).T, state_next.reshape(2*self.sz, -1).T, \
                    latent.reshape(2*self.groups, -1).T, latent_next.reshape(2*self.groups, -1).T

    def _simulate_multiseries(self):
        
        pass

    def reshape(self, sir_data_all):
        
        pass

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.input)

    def __getitem__(self, idx):
        """
        Return an item from the dataset.
        
        :param idx: Index of the item.
        :return: A tuple of torch.Tensor representing the input and output.
        """
        return torch.tensor(self.input[idx], dtype=torch.float), torch.tensor(self.output[idx], dtype=torch.float)
