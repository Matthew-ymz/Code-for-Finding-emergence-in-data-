import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from sklearn.linear_model import LinearRegression
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from importlib import reload
from ei.EI_calculation import approx_ei
from ei.EI_calculation import calculate_multistep_predict
from ei.EI_calculation import test_model_causal_multi_sis
from ei.EI_calculation import to_weights
from ei.EI_calculation import kde_density
from datetime import datetime
t0 = datetime.now()
def cpt(s):
    #Timing function
    global t0
    t = datetime.now()
    print(f'check point{s:->10}-> {t.time()}; lasting {t - t0} seconds')
    t0 = t
    
class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask, device):
        super(InvertibleNN, self).__init__()
        
        self.device = device
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size()[0] // 2
        self.t = torch.nn.ModuleList([nett() for _ in range(length)]) #repeating len(masks) times
        self.s = torch.nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size()[1]
    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0], device=self.device)
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0], device=self.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
        
class Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Renorm_Dynamic, self).__init__()
        if sym_size % 2 !=0:
            sym_size = sym_size + 1
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        nets = lambda: nn.Sequential(nn.Linear(sym_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, sym_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(sym_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, sym_size))
        self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, latent_size))
        self.inv_dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                 nn.Linear(hidden_units, latent_size))
        mask1 = torch.cat((torch.zeros(1, sym_size // 2, device=self.device), torch.ones(1, sym_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        
        prior = distributions.MultivariateNormal(torch.zeros(latent_size), torch.eye(latent_size))
        self.flow = InvertibleNN(nets, nett, masks, self.device)
        self.normalized_state=normalized_state
        self.is_random = is_random
        if is_random:
            self.sigmas = torch.nn.parameter.Parameter(torch.rand(1, latent_size, device=self.device))
            
    def forward(self, x):
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        
        s_next = self.dynamics(s) + s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
        y = self.decoding(s_next)
        return y, s, s_next
    
    def multi_back_forward(self, x, steps):
        batch_size = x.size()[0]
        x_hist = x
        predict, latent, latent_n = self.back_forward(x)
        z_hist = latent
        n_hist = torch.zeros(x.size()[0], x.size()[1]-latent.size()[1], device = self.device)
        for t in range(steps):    
            z_next=self.inv_dynamics(latent) - latent
            x_next = self.decoding(z_next)
            z_hist = torch.cat((z_hist, z_next), 0)
            x_hist = torch.cat((x_hist, self.eff_predict(x_next)), 0)
            latent = z_next
        return x_hist[batch_size:,:], z_hist[batch_size:,:]
        
    def back_forward(self, x):
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.inv_dynamics(s) + s
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
        y = self.decoding(s_next)
        return y, s, s_next
        
    def multi_step_forward(self, x, steps):
        batch_size = x.size()[0]
        x_hist = x
        predict, latent, latent_n = self.forward(x)
        z_hist = latent
        n_hist = torch.zeros(x.size()[0], x.size()[1]-latent.size()[1], device = self.device)
        for t in range(steps):    
            z_next, x_next = self.simulate(latent)
            z_hist = torch.cat((z_hist, z_next), 0)
            x_hist = torch.cat((x_hist, self.eff_predict(x_next)), 0)
            #n_hist = torch.cat((n_hist, noise), 0)
            latent = z_next
        return x_hist[batch_size:,:], z_hist[batch_size:,:]
        
    def decoding(self, s_next,de_noi_size=1):
        sz = self.sym_size - self.latent_size
        if sz>0:
            noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((s_next.size()[0], 1))
            noise = noise.to(self.device)
            #print(noise.size(), s_next.size(1))
            if s_next.size()[0]>1:
                noise = noise.squeeze(1)
            else:
                noise = noise.squeeze(0)
            #print(noise.size())
            zz = torch.cat((s_next, noise), 1)
        else:
            zz = s_next
        y,_ = self.flow.g(zz)
        return y

    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        s, _ = self.flow.f(xx)
        #if self.normalized_state:
        #    s = torch.tanh(s)
        return s[:, :self.latent_size]
        
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
        
    def simulate(self, x):
        x_next = self.dynamics(x) + x
        if self.normalized_state:
            x_next = torch.tanh(x_next)
        if self.is_random:
            x_next = x_next + torch.relu(self.sigmas.repeat(x_next.size()[0],1)) * torch.randn(x_next.size(), device=self.device)
        decode= self.decoding(x_next)
        return x_next, decode
        
    def multi_step_prediction(self, s, steps):
        s_hist = s
        z_hist = self.encoding(s)
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next = self.simulate(z)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist
        

def train(train_data, test_data, sz, scale, mae2_w, T2, T1 = 3001, encoder_interval = 1000, temperature=1, m_step = 10, test_start = 0, test_end = 0.3, sigma=0.03, rou=-0.5, L=1, hidden_units = 64, batch_size = 700, framework = 'nis'):
    MAE = torch.nn.L1Loss()
    MAE_raw = torch.nn.L1Loss(reduction='none')
    ss,sps,ls,lps = train_data
    sample_num = ss.size()[0] # batch_size * (steps * (k1 + k2) -1)
    weights = torch.ones(sample_num, device=device) 
    net = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
                         hidden_units = hidden_units, normalized_state=True, device = device)
    
    net.load_state_dict(torch.load('netnn_init_trnorm0.1+zero_seed=4.mdl').state_dict())  
    net.to(device=device)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)    
    result_nn = []
    eis =[]
    term1s =[]
    term2s =[]
    losses =[]
    MAEs_mstep =[]


    for epoch in range(T1):
        start = np.random.randint(ss.size()[0]-batch_size)
        end = start+batch_size
        s,sp,l,lp, w = ss[start:end], sps[start:end], ls[start:end], lps[start:end], weights[start:end]
        predicts, latent, latentp= net.forward(s)
        loss = MAE(sp, predicts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if framework == 'nis+':
            if epoch > 0 and epoch % encoder_interval == 0:
                cpt('w_0')
                # preparing training data
                net_temp = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
                             hidden_units = hidden_units, normalized_state=False, device = device)
                net_temp.load_state_dict(net.state_dict())
                net_temp.to(device=device)
                encodings = net_temp.encoding(ss)  
                cpt('w_1')
                log_density, k_model_n = kde_density(encodings)  # Probability Distribution of Encoded Data
                cpt('w_2')
                log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #Probability Distribution of Uniform Distribution
                logp = log_rho - log_density  
                weights = to_weights(logp, temperature) * sample_num
                if use_cuda:
                    weights = weights.cuda(device=device)
                weights=torch.where(weights<10,weights,10.)
                cpt('w_3')
            
            for p in net.flow.parameters():
                p.requires_grad = False
                
            latent1=net.encoding(s)
            predicts0, latent0, latentp0 = net.back_forward(sp) 
            mae2 = (MAE_raw(latent1, latentp0).mean(axis=1) * w).mean()  #
            optimizer.zero_grad()
            mae2.backward()
            optimizer.step()
            
            for p in net.flow.parameters():
                p.requires_grad = True

        
        if epoch % 500 == 0:
            cpt('o_0')
            print('Epoch:', epoch)
            mae_mstep = 0
            for s in np.linspace(test_start,test_end,20):
                s=float(s)
                i=(1-s)/2 #sir
                mae_mstep += calculate_multistep_predict(net,s,i,steps=m_step, sigma=sigma, rou=rou)
            mae_mstep /= 20
            ei1, sigmas1,weightsnn = test_model_causal_multi_sis(test_data,MAE_raw,net,sigma,scale, L=L,num_samples = 1000)
            print('Train loss: CommonNet= %.4f' %  loss.item())
            print('dEI: CommonNet= %.4f' % ei1[0])
            print('term1: CommonNet= %.4f'% ei1[3])
            print('term2: CommonNet= %.4f'% ei1[4])
            print(120*'-')
  
            eis.append(ei1[0])
            term1s.append(ei1[3].item())
            term2s.append(ei1[4].item())
            losses.append(loss.item())
            MAEs_mstep.append(mae_mstep) 
            cpt('o_1')

    for epoch in range(T1,T2):
        if framework == 'nis':
            start = np.random.randint(ss.size()[0]-batch_size)
            end = start+batch_size
            s,sp,l,lp = ss[start:end], sps[start:end], ls[start:end], lps[start:end]
            predicts, latent, latentp= net.forward(s)
            loss = MAE(sp, predicts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        elif framework == 'nis+':
            start = np.random.randint(ss.size()[0]-batch_size)
            end = start+batch_size
            s,sp,l,lp, w = ss[start:end], sps[start:end], ls[start:end], lps[start:end], weights[start:end]
            predicts1, latent1, latentp1 = net.forward(s)
            predicts0, latent0, latentp0 = net.back_forward(sp) 
            mae1 = (MAE_raw(sp, predicts1).mean(axis=1) * w).mean() 
            mae2 = (MAE_raw(latent1,latentp0).mean(axis=1) * w).mean() 
            loss = mae1+mae2_w*mae2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
            if epoch > 0 and epoch % encoder_interval == 0:
                cpt('w_0')
                # preparing training data
                net_temp = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
                             hidden_units = hidden_units, normalized_state=False, device = device)
                net_temp.load_state_dict(net.state_dict())
                net_temp.to(device=device)
                encodings = net_temp.encoding(ss)  
                cpt('w_1')
                log_density, k_model_n = kde_density(encodings)  # Probability Distribution of Encoded Data
                cpt('w_2')
                log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #Probability Distribution of Uniform Distribution
                logp = log_rho - log_density  
                weights = to_weights(logp, temperature) * sample_num
                if use_cuda:
                    weights = weights.cuda(device=device)
                weights=torch.where(weights<10,weights,10.)
                cpt('w_3')
            
        if epoch % 500 == 0:
            cpt('o_0')
            print('Epoch:', epoch)
            mae_mstep = 0
            for s in np.linspace(test_start,test_end,20):
                s=float(s)
                i=(1-s)/2 #sir
                mae_mstep += calculate_multistep_predict(net,s,i,steps=m_step,sigma=sigma, rou=rou)
            mae_mstep /= 20
            ei1, sigmas1,weightsnn = test_model_causal_multi_sis(test_data,MAE_raw,net,sigma,scale, L=L,num_samples = 1000)
            print('Train loss: CommonNet= %.4f' %  loss.item())
            print('dEI: CommonNet= %.4f' % ei1[0])
            print('term1: CommonNet= %.4f'% ei1[3])
            print('term2: CommonNet= %.4f'% ei1[4])
            print(120*'-')
  
            eis.append(ei1[0])
            term1s.append(ei1[3].item())
            term2s.append(ei1[4].item())
            losses.append(loss.item())
            MAEs_mstep.append(mae_mstep) 
            cpt('o_1')
            
    return eis, term1s, term2s, losses, MAEs_mstep