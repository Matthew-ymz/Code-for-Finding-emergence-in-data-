import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
from sklearn.neighbors import KernelDensity


def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy())
    log_density = kde.score_samples(X.cpu().data.numpy())
    return log_density, kde


def to_weights(log_w, temperature=1):
    logsoft = nn.LogSoftmax(dim = 0)
    weights = torch.exp(logsoft(log_w/temperature))
    return weights


def approx_ei_nn(dynamics, state, latent, latent_p, device, num_samples= 1000, L=100):
    # state [50, 36]  latent [50, 36]  latent_p [50, 36]
    ll = torch.nn.L1Loss(reduction='none')
    scale = latent.size()[1]
    log_density, _ = kde_density(latent.cpu().data)
    log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))
    logp = log_rho - log_density
    weights = to_weights(logp) * latent.size()[0]
    weights = weights.to(device)
    weights = weights.unsqueeze(1)
    new_weights = weights.clone()
    for _ in range(scale-1):
        new_weights = torch.cat((new_weights,weights),1)
    mae1 = ll(state, latent_p) * new_weights
    sigmas = mae1.mean(axis=0)
    sigmas_matrix = torch.diag(sigmas)
    ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                       num_samples=num_samples, L=L, easy=True, device=device) 
    return ei[0]


def approx_ei(input_size, output_size, sigmas_matrix, func, num_samples = 1000, L = 100, easy=True, device=None):
    # Approximated calculation program for various EI (Dimensionally averaged EI, Eff, and EI) and related 
    # Quantities on Gaussian neural network
    # input variables：
    #input_size: the dimension of input to the func (neural network) (x_dim)
    #output_size: the dimension of the output of the func (neural network) (y_dim)
    #sigma_matrix: the inverse of the covariance matrix of the gaussian distribution on Y: dim: y_dim * y_dim
    #func: any function, can be a neural network
    #L the linear size of the box of x on one side (-L,L)
    #num_samples: the number of samples of the monte carlo integration on x
    
    # output variables：
    # d_EI： dimensionally averaged EI
    # eff： EI coefficient （EI/H_max)
    # EI: EI, effective information (common)
    # term1: - Shannon Entropy
    # term2: EI+Shannon Entropy (determinant of Jacobian)
    # -np.log(rho): - ln(\rho), where \rho=(2L)^{-output_size} is the density of uniform distribution

    rho=1/(2*L)**input_size #the density of X even distribution
    dett=1.0
    if easy:
        dd = torch.diag(sigmas_matrix)
        dett = torch.log(dd).sum()
    else:
        #dett = np.log(np.linalg.det(sigmas_matrix_np))
        dett = torch.log(torch.linalg.det(sigmas_matrix))
    term1 = - (output_size + output_size * np.log(2*np.pi) + dett)/2 
    #sampling x on the space [-L,L]^n, n is the number of samples
    xx=L*2*(torch.rand(num_samples, input_size, device=device)-1/2)
    dets = 0
    logdets = 0
    #iterate all samples of x
    for i in range(xx.size()[0]):
        jac=jacobian(func, xx[i,:]) #use pytorch's jacobian function to obtain jacobian matrix
        det=torch.abs(torch.det(jac)) #calculate the determinate of the jacobian matrix
        dets += det.item()
        if det!=0:
            logdets+=torch.log(det).item() #log jacobian
        else:
            logdet = -(output_size+output_size*np.log(2*np.pi)+dett)
            logdets+=logdet.item()
    
    int_jacobian = logdets / xx.size()[0] #take average of log jacobian
    
    term2 = input_size * np.log(2 * L) + int_jacobian # derive the 2nd term
    
    if dets==0:
        term2 = - term1
    EI = max(term1 + term2, 0)
    if torch.is_tensor(EI):
        EI = EI.item()
    eff = EI / (input_size * np.log(2 * L))
    d_EI = EI/output_size
    
    return d_EI, eff, EI, term1, term2, input_size * np.log(2 * L)