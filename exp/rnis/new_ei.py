import torch
import numpy as np
from torch import nn
from torchmetrics import Metric
from torch import tensor
from sklearn.neighbors import KernelDensity

def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy()) #bindwidth=0.02
    log_density = kde.score_samples(X.cpu().data.numpy())
    return log_density, kde


def to_weights(log_w, temperature=1): 
    #Normalize log_w using softmax and obtain the weights.
    logsoft = nn.LogSoftmax(dim = 0)
    weights = torch.exp(logsoft(log_w/temperature))
    return weights


class EI(Metric):

    def __init__(self):
        super().__init__()
        self.dei = 0
        self.term1 = 0
        self.term2 = 0
        self.add_state("ei_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ei_term1", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ei_term2", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=tensor(0), dist_reduce_fx="sum")

    def update(self, ei_items):
        bandwidth = 0.05
        temperature = 1
        L = 1
        h_t = ei_items["h_t"]
        h_t1 = ei_items["h_t1"]
        h_t1_hat = ei_items["h_t1_hat"]
        count = ei_items["count"]
        avg_log_jacobian = ei_items["avg_log_jacobian"]
        samples = h_t.size(0)
        scale = h_t.size(1)
        n = h_t.shape[0]
        log_density = torch.zeros(n, dtype=h_t.dtype, layout=h_t.layout, device=h_t.device)
        output_size = scale
        input_size = scale

        for i in range(n):
            kernels = torch.exp(-0.5 * ((h_t - h_t[i]) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
            density = torch.sum(kernels) / n
            log_density[i] = torch.log(density)

        log_rho = - scale * np.log(2.0 * L)
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(logp / temperature)
        weights = weights * samples    
       
        mae_loss = torch.nn.L1Loss(reduction='none')
        
        # Expand weights to match dimensions for broadcasting
        weights = weights.unsqueeze(1)
        
        # Repeat weights along the second dimension based on the scale
        weights = weights.repeat_interleave(scale, dim=1)
        # Calculate MAE between predicted and actual values, weighted by weights
        mae = mae_loss(h_t1_hat, h_t1) * weights
        sigmas = mae.mean(axis=0)
        det_sigma = torch.log(sigmas).sum()

        # Calculate the first term of EI (Shannon Entropy)
        term1 = - (output_size + output_size * np.log(2 * np.pi) + det_sigma) / 2 * (1 - count / samples)
        
        # Calculate the second term of EI
        term2 = input_size * np.log(2 * L) + avg_log_jacobian
        
        # Calculate EI and its variants
        EI = max(term1 + term2, 0)
        d_EI = EI / output_size

        self.ei_sum += d_EI
        self.ei_term1 += term1
        self.ei_term2 += term2
        self.n += 1
        
        self.dei = d_EI
        self.term1 = term1
        self.term2 = term2
        return

    def compute(self):
        return self.dei, self.term1, self.term2 #self.ei_sum / self.n, self.ei_term1 / self.n, self.ei_term2 / self.n


if __name__ == '__main__':
    ei = EI()
    count = 0
    h_t = torch.randn(600, 2)
    h_t1 = torch.randn(600, 2)
    h_t1_hat = torch.randn(600, 2)
    avg_log_jacobian = torch.Tensor(np.asarray(0.0071).astype(np.float16))
    ei_items = {"h_t": h_t,
                "h_t1": h_t1,
                "h_t1_hat": h_t1_hat,
                "avg_log_jacobian": avg_log_jacobian,
                "count": count}
    d_EI = ei(ei_items)

    print(d_EI)
