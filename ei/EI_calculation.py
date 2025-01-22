import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
from sklearn.neighbors import KernelDensity
import ei.entropy_estimators as ee

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')


def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy()) #bindwidth=0.02
    log_density = kde.score_samples(X.cpu().data.numpy())
    return log_density, kde


def to_weights(log_w, temperature=1): 
    #Normalize log_w using softmax and obtain the weights.
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
    EI = term1 +term2
    if torch.is_tensor(EI):
        EI = EI.item()
    eff = EI / (input_size * np.log(2 * L))
    d_EI = EI/output_size
    
    return d_EI, eff, EI, term1, term2, input_size * np.log(2 * L)

def test_model_causal_multi_sis(spring_data,MAE_raw,net1,sigma,scale,L=1, num_samples = 1000, temperature=1):
    #EI calculation function
    sigmas_matrix=torch.zeros([2,2],device=device)
    s,sp,l,lp=spring_data
    samples = s.size()[0]
    encode=net1.encoding(sp)
    predicts1, latent1, latentp1 = net1(s)
    log_density, k_model_n = kde_density(latent1)
    log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #Uniform distribution probability distribution
    logp = log_rho - log_density  #The difference between two probability distributions.
    weights = to_weights(logp, temperature) * samples
    if use_cuda:
        weights = weights.cuda(device=device)
    weights=weights.unsqueeze(1)
    weights_ = weights
    for k in range(scale-1):
        weights_ = torch.cat((weights_,weights),1)
    mae1 = MAE_raw(latentp1, encode) * weights_
    sigmas=mae1.mean(axis=0)
    sigmas_matrix = torch.diag(sigmas)
    ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(net1.dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                       num_samples = 1000, L=L, easy=True, device=device) 
    return ei, sigmas,weights

def EmergencePsi(X, V, tau=1, method=None):
    """
    Compute causal emergence criterion from data
    
    Args:
    - X: Micro time series data of shape (T, D)
    - V: Macro time series data of shape (T, R)
    - tau: Time delay (default: 1)
    - method: Estimation method for mutual information ('discrete' or 'gaussian') (default: None)
    
    Returns:
    - psi: Causal emergence criterion
    - v_mi: Mutual information in macro variables
    - x_mi: Mutual information in micro variables
    """
    
    # Parameter checks and initialisation
    if X.ndim != 2 or V.ndim != 2:
        raise ValueError("X and V have to be 2D matrices.")
    if X.shape[0] != V.shape[0]:
        raise ValueError("X and V must have the same height.")
    if tau is None:
        tau = 1
    if method is None:
        isdiscrete = np.isclose(X, np.round(X)).all() and np.isclose(V, np.round(V)).all()
        if isdiscrete:
            method = 'discrete'
        else:
            method = 'gaussian'
    
    if method.lower() == 'gaussian':
        MI_fun = ee.mi#GaussianMI
    elif method.lower() == 'discrete':
        MI_fun = ee.midd#DiscreteMI
    else:
        raise ValueError("Unknown method. Implemented options are 'gaussian' and 'discrete'.")
    
    # Compute mutual infos and psi
    v_mi = MI_fun(V[:-tau, :], V[tau:, :])
    x_mi = sum([MI_fun(np.array([X[:-tau, j]]).T, V[tau:, :]) for j in range(X.shape[1])])
    psi = v_mi - x_mi
    
    if len(X.shape) < 2:
        x_mi = None
    if len(V.shape) < 2:
        v_mi = None
    return psi, v_mi, x_mi

def test_vae_causal_multi_sis(spring_data,MAE_raw,vae,sigma,scale,L=1, num_samples = 1000, temperature=1):
    #EI calculation function
    sigmas_matrix=torch.zeros([2,2],device=device)
    s,sp,l,lp=spring_data
    samples = s.size()[0]
    latent_params = vae.encoder(sp)
    mu_p, logvar_p = torch.chunk(latent_params, 2, dim=1)
    predicts, mu, logvar = vae.forward(s)
    log_density, k_model_n = kde_density(mu)
    log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #Uniform distribution probability distribution
    logp = log_rho - log_density  #The difference between two probability distributions.
    weights = to_weights(logp, temperature) * samples
    if use_cuda:
        weights = weights.cuda(device=device)
    weights=weights.unsqueeze(1)
    mae1 = MAE_raw(mu, mu_p) * torch.cat((weights,weights),1)
    sigmas=mae1.mean(axis=0)
    
    sigmas_matrix = torch.diag(sigmas)
    ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(vae.dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                       num_samples = 1000, L=L, easy=True, device=device) 
    return ei, sigmas,weights

def test_model_causal_rnis_sis(spring_data,MAE_raw,net1,sigma,scale,L=1, num_samples = 1000, temperature=1):
    #EI calculation function
    sigmas_matrix=torch.zeros([2,2],device=device)
    s,sp,l,lp=spring_data
    samples = s.size()[0]
    encode=net1.encoding(sp)
    predicts1, latent1, latentp1 = net1(s)
    log_density, k_model_n = kde_density(latent1)
    log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #Uniform distribution probability distribution
    logp = log_rho - log_density  #The difference between two probability distributions.
    weights = to_weights(logp, temperature) * samples
    if use_cuda:
        weights = weights.cuda(device=device)
    weights=weights.unsqueeze(1)
    weights_ = weights
    for k in range(scale-1):
        weights_ = torch.cat((weights_,weights),1)
    mae1 = MAE_raw(latentp1, encode) * weights_
    sigmas=mae1.mean(axis=0)
    sigmas_matrix = torch.diag(sigmas)
    ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(net1.dynamics.f(x.unsqueeze(0))[0]), 
                       num_samples = 1000, L=L, easy=True, device=device) 
    return ei, sigmas,weights

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)