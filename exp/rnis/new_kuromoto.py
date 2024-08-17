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



class KuromotoModel(Dataset):
    def __init__(self, size_list, steps, dt, interval, rho, sz, groups, coupling, use_cache=True):
        """
        TODO
        
        :param size_list: List of initial state sizes.
        :param steps: Number of steps to run (including the starting point).
        :param dt: Step size.
        :param interval: Sampling interval.
        :param sigma: Standard deviation of noise.
        :param rho: Correlation coefficient of noise.
        """
        self.size_list = size_list
        self.rho = rho
        self.steps = steps
        self.dt = dt
        self.interval = interval
        self.coupling = coupling
        self.init_total_number = np.sum(size_list)

        #self.data = self.simulate_multiseries(size_list)
        self.prior = multivariate_normal(mean=np.zeros(2), cov=np.array([[1, rho], [rho, 1]]))
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
        self.input, self.output, _, _ = self.simulate_oneserie()


    def one_step(self, thetas, dt=0.01):
        # ii = np.expand_dims(thetas, 1).repeat(self.sz, 1)
        # # jj = ii.transpose(0, 1)
        # jj = ii.T
        # print(ii, jj)
        # dff = jj - ii
        # sindiff = np.sin(dff)
        # mult = self.coupling * self.obj_matrix @ sindiff
        # dia =  np.diagonal(mult)
        # noise = np.random.rand(self.sz) * 0 #10
        # thetas = self.dt * (self.omegas + dia + noise) + thetas
        # # print(thetas.shape, (self.omegas + dia + noise).shape)
        # return thetas

        thetas = torch.tensor(thetas)
        ii = thetas.unsqueeze(0).repeat(thetas.size()[0], 1)
        jj = ii.transpose(0, 1)
        dff = jj - ii
        sindiff = torch.sin(dff)
        mult = self.coupling * torch.tensor(self.obj_matrix) @ sindiff
        dia =  torch.diagonal(mult)
        noise = torch.randn(self.sz) * 0#10
        thetas = 0.01 * (torch.tensor(self.omegas) + dia + noise) + thetas
        return np.array(thetas)


    def simulate_oneserie(self, batch_size=1, sample_step=5):
        """
        TODO
        """

        time_steps = self.steps * sample_step

        states = np.zeros([batch_size, self.sz, time_steps // sample_step])
        centers = np.zeros([batch_size, self.groups, time_steps // sample_step])
        for i in range(batch_size):
            thetas = np.random.rand(self.sz) * 2 * np.pi
            for t in range(time_steps):
                thetas = self.one_step(thetas, self.obj_matrix)
                if t % sample_step == 0:
                    states[i, :, t // sample_step] = thetas
                    cos_ccs = (np.cos(thetas) @ self.group_matrix) * self.groups / self.sz
                    sin_ccs = (np.sin(thetas) @ self.group_matrix) * self.groups / self.sz
                    #rr = np.sqrt(cos_ccs ** 2 + sin_ccs ** 2)
                    #phi = np.arcsin(sin_ccs / rr)
                    #centers[i, : groups, t // sample_step] = cos_ccs
                    centers[i, :, t // sample_step] = sin_ccs
        state = np.sin(states[:, :, :-1])
        state_next = np.sin(states[:, :, 1:])
        latent = centers[:, :, :-1]
        latent_next = centers[:, :, 1:]
        return state.reshape(self.sz, -1).T, state_next.reshape(self.sz, -1).T, \
                    latent.reshape(self.groups, -1).T, latent_next.reshape(self.groups, -1).T

    def _simulate_multiseries(self):
        """
        Simulate multiple time series from various starting points to create the main dataset.
        
        :return: sir_input and sir_output arrays.
        """
        num_obs = int(self.steps / self.interval)
        sir_data_all = np.zeros([self.init_total_number, num_obs, 4])
        num_strip = len(self.size_list)
        frac = 1 / num_strip
        
        for strip in range(num_strip):
            sir_data_part = np.zeros([self.size_list[strip], num_obs, 4])
            boundary_left = strip * frac
            boundary_right = boundary_left + frac
            S_init = np.random.uniform(boundary_left, boundary_right, self.size_list[strip])
            I_init = []
            while len(I_init) < self.size_list[strip]:
                I = np.random.rand(1)[0]
                S = S_init[len(I_init)]
                if S + I <= 1:
                    sir_data_part[len(I_init),:,:] = self.simulate_oneserie(S, I)
                    I_init.append(I)
            size_list_cum = np.cumsum(self.size_list)
            size_list_cum = np.concatenate([[0], size_list_cum])
            sir_data_all[size_list_cum[strip]:size_list_cum[strip+1], :, :] = sir_data_part
        sir_input, sir_output = self.reshape(sir_data_all = sir_data_all)
        return sir_input, sir_output

    def reshape(self, sir_data_all):
        """
        Reshape the generated multi-time series into input and output arrays.
        
        :param sir_data_all: Array of all time series data.
        :return: sir_input and sir_output arrays.
        """
        sir_input = sir_data_all[:, :-1, :].reshape(-1, self.sz)
        sir_output = sir_data_all[:, 1:, :].reshape(-1, self.sz)
        return sir_input, sir_output

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.sir_input)

    def __getitem__(self, idx):
        """
        Return an item from the dataset.
        
        :param idx: Index of the item.
        :return: A tuple of torch.Tensor representing the input and output.
        """
        return torch.tensor(self.sir_input[idx], dtype=torch.float), torch.tensor(self.sir_output[idx], dtype=torch.float)
