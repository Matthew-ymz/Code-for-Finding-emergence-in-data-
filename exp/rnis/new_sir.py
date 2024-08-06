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


class SIRModel(Dataset):
    def __init__(self, size_list, beta, gamma, steps, dt, interval, sigma, rho, use_cache=True):
        """
        Initialize the SIR model dataset.
        
        :param size_list: List of initial state sizes.
        :param beta: Infection rate.
        :param gamma: Recovery rate.
        :param steps: Number of steps to run (including the starting point).
        :param dt: Step size.
        :param interval: Sampling interval.
        :param sigma: Standard deviation of noise.
        :param rho: Correlation coefficient of noise.
        """
        self.size_list = size_list
        self.beta, self.gamma = beta, gamma
        self.sigma, self.rho = sigma, rho
        self.steps = steps
        self.dt = dt
        self.interval = interval
        self.init_total_number = np.sum(size_list)

        #self.data = self.simulate_multiseries(size_list)
        self.prior = multivariate_normal(mean=np.zeros(2), cov=np.array([[1, rho], [rho, 1]]))

        # cache_key = f"SIR_{size_list}_{beta}_{gamma}_{steps}_{dt}_{interval}_{sigma}_{rho}"
        # cache_data_fp = os.path.join(
        #     os.getcwd(),
        #     'data',
        #     'SIR',
        #     "%s.npy" % cache_key
        # )

        # if use_cache and os.path.isfile(cache_data_fp):
        #     loaded_data_dict = np.load(cache_data_fp, allow_pickle=True).item()
        #     self.sir_input = loaded_data_dict['input']
        #     self.sir_output = loaded_data_dict['output']

        # else:
        self.sir_input, self.sir_output = self._simulate_multiseries()
        # data_dict = {
        #     'input': self.sir_input,
        #     'output': self.sir_output,
        # }

        # np.save(cache_data_fp, data_dict)

    def perturb(self, S, I):
        """
        Add observational noise to the macro states S and I.
        
        :param S: Susceptible population.
        :param I: Infected population.
        :return: Observed states with noise.
        """
        noise_S = self.prior.rvs(size=1) * self.sigma
        noise_I = self.prior.rvs(size=1) * self.sigma
        S_obs0 = np.expand_dims(S + noise_S[0], axis=0)
        S_obs1 = np.expand_dims(S + noise_S[1], axis=0)
        I_obs0 = np.expand_dims(I + noise_I[0], axis=0)
        I_obs1 = np.expand_dims(I + noise_I[1], axis=0)
        SI_obs = np.concatenate((S_obs0, I_obs0, S_obs1, I_obs1), 0)
        return SI_obs
    
    def simulate_oneserie(self, S, I):
        """
        Simulate a single time series from a specific starting point.
        
        :param S: Initial susceptible population (as a ratio).
        :param I: Initial infected population (as a ratio).
        :return: Time series data.
        """
        sir_data = []
        for k in range(self.steps):
            if k % self.interval == 0:
                SI_obs = self.perturb(S, I)
                sir_data.append(SI_obs)
                
            new_infected = self.beta * S * I 
            new_recovered = self.gamma * I
            S = S - new_infected * self.dt
            I = I + (new_infected - new_recovered) * self.dt
        return np.array(sir_data)

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
        sir_input = sir_data_all[:, :-1, :].reshape(-1, 4)
        sir_output = sir_data_all[:, 1:, :].reshape(-1, 4)
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
