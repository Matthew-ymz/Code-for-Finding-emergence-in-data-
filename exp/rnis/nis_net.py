import torch
from torch import nn
from torch import distributions
from torch.autograd.functional import jacobian
import numpy as np

class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask):
        super(InvertibleNN, self).__init__()
        
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size(0) // 2
        self.t = nn.ModuleList([nett() for _ in range(length)])
        self.s = nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size(1)
    
    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0])
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    

class NISNet(nn.Module):
    def __init__(self, 
                 input_size: int = 4, 
                 latent_size: int = 2, 
                 output_size: int = 4, 
                 hidden_units: int = 64, 
                 hidden_units_dyn: int = 64,
                 is_normalized: bool = True
                ) -> None:
        super(NISNet, self).__init__()
        if input_size % 2 != 0:
            input_size += 1
            
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units_dyn = hidden_units_dyn
        self.pi = torch.tensor(torch.pi)
        self.func = lambda x: (self.dynamics(x) + x)

        nets = lambda: nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, input_size),
            nn.Tanh()
        )
        
        nett = lambda: nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, input_size)
        )
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_size, hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units_dyn, latent_size)
        )

        mask1 = torch.cat((torch.zeros(1, input_size // 2), torch.ones(1, input_size // 2)), 1)
        mask2 = 1 - mask1
        masks_enc = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        self.flow = InvertibleNN(nets, nett, masks_enc)
        self.is_normalized = is_normalized
        
    def encoding(self, x):
        h, _ = self.flow.f(x)
        return h[:, :self.latent_size]
    
    def decoding(self, h_t1):
        sz = self.input_size - self.latent_size
        means = torch.zeros(sz, dtype=h_t1.dtype, device=h_t1.device)
        covs = torch.eye(sz, dtype=h_t1.dtype, device=h_t1.device)
        if sz > 0:
            noise = distributions.MultivariateNormal(means, covs).sample((h_t1.size(0),))
            h_t1 = torch.cat((h_t1, noise), dim=1)
        x_t1_hat, _ = self.flow.g(h_t1)
        return x_t1_hat
    
    def cal_EI_1(self, h_t, num_samples, L):
        if self.training is not True:
            input_size = h_t.size(1)
            jac_in = L * (2 * torch.rand(num_samples, input_size, dtype=h_t.dtype, device=h_t.device) - 1)
            jacobian_matrix = jacobian(self.func, jac_in)
            diag_matrices = jacobian_matrix.permute(0, 2, 1, 3).diagonal(dim1=0, dim2=1).permute(2, 0, 1)
            det_list = torch.linalg.det(diag_matrices)
            mask = det_list == 0
            count = mask.sum().item()
            det_list[mask] = 1  # 避免在 log 中计算 0
            avg_log_jacobian = torch.log(det_list.abs()).mean()
        
        else:
            count = 0
            avg_log_jacobian = 0
        return count, avg_log_jacobian
    
    def forward(self, x_t, x_t1, L=1, num_samples=1000):
        h_t = self.encoding(x_t)

        count, avg_log_jacobian = self.cal_EI_1(h_t, num_samples, L)
        
        h_t1 = self.encoding(x_t1)
        h_t1_hat = self.func(h_t)
        
        if self.is_normalized:
            h_t1_hat = torch.tanh(h_t1_hat)
        
        x_t1_hat = self.decoding(h_t1_hat)
        
        ei_items = {"h_t": h_t,
                    "h_t1": h_t1,
                    "h_t1_hat": h_t1_hat,
                    "avg_log_jacobian": avg_log_jacobian,
                    "count": count}
        
        return x_t1_hat, ei_items


class NISPNet(NISNet):
    def __init__(self, 
                 input_size: int = 4, 
                 latent_size: int = 2, 
                 output_size: int = 4, 
                 hidden_units: int = 64, 
                 hidden_units_dyn: int = 64,
                 is_normalized: bool = True
                ) -> None:
        super().__init__(input_size, latent_size, output_size, hidden_units, hidden_units_dyn, is_normalized)
        # if input_size % 2 != 0:
        #     input_size += 1
            
        # self.latent_size = latent_size
        # self.input_size = input_size
        # self.output_size = output_size
        # self.pi = torch.tensor(torch.pi)
        # self.func = lambda x: (self.dynamics(x) + x)

        # nets_flow = lambda: nn.Sequential(
        #     nn.Linear(input_size, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, input_size),
        #     nn.Tanh()
        # )
        
        # nett_flow = lambda: nn.Sequential(
        #     nn.Linear(input_size, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, hidden_units), 
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, input_size)
        # )

        # self.dynamics = nn.Sequential(
        #     nn.Linear(latent_size, hidden_units), 
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, hidden_units), 
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, latent_size)
        # )

        self.inv_dynamics = nn.Sequential(
            nn.Linear(latent_size, self.hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, self.hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, latent_size)
        )

        # mask1 = torch.cat((torch.zeros(1, input_size // 2), torch.ones(1, input_size // 2)), 1)
        # mask2 = 1 - mask1
        # masks_enc = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        # self.flow = InvertibleNN(nets_flow, nett_flow, masks_enc)
        # self.is_normalized = is_normalized
        
    # def encoding(self, x):
    #     h, _ = self.flow.f(x)
    #     return h[:, :self.latent_size]
    
    # def decoding(self, h_t1):
    #     sz = self.input_size - self.latent_size
    #     means = torch.zeros(sz, dtype=h_t1.dtype, device=h_t1.device)
    #     covs = torch.eye(sz, dtype=h_t1.dtype, device=h_t1.device)
    #     if sz > 0:
    #         noise = distributions.MultivariateNormal(means, covs).sample((h_t1.size(0),))
    #         h_t1 = torch.cat((h_t1, noise), dim=1)
    #     x_t1_hat, _ = self.flow.g(h_t1)
    #     return x_t1_hat
    
    # def cal_EI_1(self, h_t, num_samples, L):
    #     if self.training is not True:
    #         input_size = h_t.size(1)
    #         jac_in = L * (2 * torch.rand(num_samples, input_size, dtype=h_t.dtype, device=h_t.device) - 1)
    #         jacobian_matrix = jacobian(self.func, jac_in)
    #         diag_matrices = jacobian_matrix.permute(0, 2, 1, 3).diagonal(dim1=0, dim2=1).permute(2, 0, 1)
    #         det_list = torch.linalg.det(diag_matrices)
    #         mask = det_list == 0
    #         count = mask.sum().item()
    #         det_list[mask] = 1  # 避免在 log 中计算 0
    #         avg_log_jacobian = torch.log(det_list.abs()).mean()
        
    #     else:
    #         count = 0
    #         avg_log_jacobian = 0
    #     return count, avg_log_jacobian
    
    # def forward(self, x_t, x_t1, L=1, num_samples=1000):
    #     h_t = self.encoding(x_t)

    #     count, avg_log_jacobian = self.cal_EI_1(h_t, num_samples, L)
        
    #     h_t1 = self.encoding(x_t1)
    #     h_t1_hat = self.func(h_t)
        
    #     if self.is_normalized:
    #         h_t1_hat = torch.tanh(h_t1_hat)
        
    #     x_t1_hat = self.decoding(h_t1_hat)
        
    #     ei_items = {"h_t": h_t,
    #                 "h_t1": h_t1,
    #                 "h_t1_hat": h_t1_hat,
    #                 "avg_log_jacobian": avg_log_jacobian,
    #                 "count": count}
        
    #     return x_t1_hat, ei_items

    def back_forward(self, x_t1, L=1, num_samples=1000):
        h_t1 = self.encoding(x_t1)
        h_t_hat = self.inv_dynamics(h_t1) + h_t1
        return h_t_hat

    def reweight(self, h_t, L=1):
        bandwidth = 0.05
        temperature = 1
        samples = h_t.size(0)
        scale = h_t.size(1)
        n = h_t.shape[0]
        log_density = torch.zeros(n, dtype=h_t.dtype, layout=h_t.layout, device=h_t.device)

        for i in range(n):
            kernels = torch.exp(-0.5 * ((h_t - h_t[i]) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
            density = torch.sum(kernels) / n
            log_density[i] = torch.log(density)

        log_rho = - scale * np.log(2.0 * L)
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(logp / temperature)
        weights = weights * samples
        return weights


class RNISNet(NISNet):
    def __init__(self, 
                 input_size: int = 4, 
                 latent_size: int = 2, 
                 output_size: int = 4, 
                 hidden_units: int = 64, 
                 hidden_units_dyn: int = 64,
                 is_normalized: bool = True
                ) -> None:
        super().__init__(input_size, latent_size, output_size, hidden_units, hidden_units_dyn, is_normalized)
        # if input_size % 2 != 0:
        #     input_size += 1
            
        # self.latent_size = latent_size
        # self.input_size = input_size
        # self.output_size = output_size
        # self.pi = torch.tensor(torch.pi)
        self.func = lambda x: (self.dynamics.f(x)[0])

        # nets_flow = lambda: nn.Sequential(
        #     nn.Linear(input_size, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, input_size),
        #     nn.Tanh()
        # )
        
        # nett_flow = lambda: nn.Sequential(
        #     nn.Linear(input_size, hidden_units),
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, hidden_units), 
        #     nn.LeakyReLU(), 
        #     nn.Linear(hidden_units, input_size)
        # )

        nets_dyn = lambda: nn.Sequential(
            nn.Linear(latent_size, self.hidden_units_dyn),
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, self.hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, latent_size)
        )

        nett_dyn = lambda: nn.Sequential(
            nn.Linear(latent_size, self.hidden_units_dyn),
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, self.hidden_units_dyn), 
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_units_dyn, latent_size)
        )

        # mask1 = torch.cat((torch.zeros(1, input_size // 2), torch.ones(1, input_size // 2)), 1)
        # mask2 = 1 - mask1
        # masks_enc = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        # self.flow = InvertibleNN(nets_flow, nett_flow, masks_enc)
        mask1 = torch.cat((torch.zeros(1, latent_size // 2), torch.ones(1, latent_size // 2)), 1)
        mask2 = 1 - mask1
        masks_dyn = torch.cat((mask1, mask2, mask1, mask2), 0)
        self.dynamics = InvertibleNN(nets_dyn, nett_dyn, masks_dyn)
        # self.is_normalized = is_normalized
        
    # def encoding(self, x):
    #     h, _ = self.flow.f(x)
    #     return h[:, :self.latent_size]
    
    # def decoding(self, h_t1):
    #     sz = self.input_size - self.latent_size
    #     means = torch.zeros(sz, dtype=h_t1.dtype, device=h_t1.device)
    #     covs = torch.eye(sz, dtype=h_t1.dtype, device=h_t1.device)
    #     if sz > 0:
    #         noise = distributions.MultivariateNormal(means, covs).sample((h_t1.size(0),))
    #         h_t1 = torch.cat((h_t1, noise), dim=1)
    #     x_t1_hat, _ = self.flow.g(h_t1)
    #     return x_t1_hat
    
    # def cal_EI_1(self, h_t, num_samples, L):
    #     if self.training is not True:
    #         input_size = h_t.size(1)
    #         jac_in = L * (2 * torch.rand(num_samples, input_size, dtype=h_t.dtype, device=h_t.device) - 1)
    #         jacobian_matrix = jacobian(self.func, jac_in)
    #         diag_matrices = jacobian_matrix.permute(0, 2, 1, 3).diagonal(dim1=0, dim2=1).permute(2, 0, 1)
    #         det_list = torch.linalg.det(diag_matrices)
    #         mask = det_list == 0
    #         count = mask.sum().item()
    #         det_list[mask] = 1  # 避免在 log 中计算 0
    #         avg_log_jacobian = torch.log(det_list.abs()).mean()
        
    #     else:
    #         count = 0
    #         avg_log_jacobian = 0
    #     return count, avg_log_jacobian
    
    # def forward(self, x_t, x_t1, L=1, num_samples=1000):
    #     h_t = self.encoding(x_t)

    #     count, avg_log_jacobian = self.cal_EI_1(h_t, num_samples, L)
        
    #     h_t1 = self.encoding(x_t1)
    #     h_t1_hat = self.func(h_t)
        
    #     if self.is_normalized:
    #         h_t1_hat = torch.tanh(h_t1_hat)
        
    #     x_t1_hat = self.decoding(h_t1_hat)
        
    #     ei_items = {"h_t": h_t,
    #                 "h_t1": h_t1,
    #                 "h_t1_hat": h_t1_hat,
    #                 "avg_log_jacobian": avg_log_jacobian,
    #                 "count": count}
        
    #     return x_t1_hat, ei_items

    def back_forward(self, x_t1, L=1, num_samples=1000):
        h_t1 = self.encoding(x_t1)
        h_t_hat, _ = self.dynamics.g(h_t1)
        return h_t_hat

    def reweight(self, h_t, L=1):
        h_t = h_t.cpu()
        bandwidth = 0.05
        temperature = 1
        samples = h_t.size(0)
        scale = h_t.size(1)
        n = h_t.shape[0]
        log_density = torch.zeros(n, dtype=h_t.dtype, layout=h_t.layout, device=h_t.device)

        for i in range(n):
            kernels = torch.exp(-0.5 * ((h_t - h_t[i]) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
            density = torch.sum(kernels) / n
            log_density[i] = torch.log(density)

        log_rho = - scale * np.log(2.0 * L)
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(logp / temperature)
        weights = weights * samples
        weights = weights.cuda()
        return weights