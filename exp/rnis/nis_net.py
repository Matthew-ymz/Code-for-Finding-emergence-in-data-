import torch
from torch import nn
from torch import distributions
from torch.autograd.functional import jacobian

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
                 is_normalized: bool = True
                ) -> None:
        super(NISNet, self).__init__()
        if input_size % 2 != 0:
            input_size += 1
            
        self.latent_size = latent_size
        self.input_size = input_size
        self.output_size = output_size
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
            nn.Linear(latent_size, hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, hidden_units), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_units, latent_size)
        )

        mask1 = torch.cat((torch.zeros(1, input_size // 2), torch.ones(1, input_size // 2)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        self.flow = InvertibleNN(nets, nett, masks)
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
        h_t1_hat = self.dynamics(h_t) + h_t
        
        if self.is_normalized:
            h_t1_hat = torch.tanh(h_t1_hat)
        
        x_t1_hat = self.decoding(h_t1_hat)
        
        ei_items = {"h_t": h_t,
                    "h_t1": h_t1,
                    "h_t1_hat": h_t1_hat,
                    "avg_log_jacobian": avg_log_jacobian,
                    "count": count}
        
        return x_t1_hat, ei_items
