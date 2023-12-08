import torch
import numpy as np
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from EI_calculation import approx_ei
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        #state_dim = x.size()[1]
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
    def back_forward(self, x):
        #state_dim = x.size()[1]
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.inv_dynamics(s) - s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
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
            z_next, x_next, noise = self.simulate(latent)
            z_hist = torch.cat((z_hist, z_next), 0)
            x_hist = torch.cat((x_hist, self.eff_predict(x_next)), 0)
            n_hist = torch.cat((n_hist, noise), 0)
            latent = z_next
        return x_hist[batch_size:,:], z_hist[batch_size:,:], n_hist[batch_size:,:]
    def decoding(self, s_next):
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
    def decoding1(self, s_next):
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
            noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((s_next.size()[0], 1))
            noise = noise.to(self.device)
            #print(noise.size(), s_next.size(1))
            if s_next.size()[0]>1:
                noise = noise.squeeze(1)
            else:
                noise = noise.squeeze(0)
            zz = s_next
        y,_ = self.flow.g(zz)
        return y, noise
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        s, _ = self.flow.f(xx)
        if self.normalized_state:
            s = torch.tanh(s)
        return s[:, :self.latent_size]
    def encoding1(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        s, _ = self.flow.f(xx)
        if self.normalized_state:
            s = torch.tanh(s)
        return s[:, :self.latent_size], s[:,self.latent_size:]
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    def simulate(self, x):
        x_next = self.dynamics(x) + x
        if self.normalized_state:
            x_next = torch.tanh(x_next)
        if self.is_random:
            x_next = x_next + torch.relu(self.sigmas.repeat(x_next.size()[0],1)) * torch.randn(x_next.size(), device=self.device)
        decode,noise = self.decoding1(x_next)
        return x_next, decode, noise
    def multi_step_prediction(self, s, steps):
        s_hist = s
        z_hist = self.encoding(s)
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next, _ = self.simulate(z)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist

class Stacked_Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, cut_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Stacked_Renorm_Dynamic, self).__init__()
        if latent_size < 1 or latent_size > sym_size:
            print('Latent Size is too small(<1) or too large(>input_size):', latent_size)
            raise
            return
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        i = sym_size
        flows = []
        
        while i > latent_size:
            input_size = max(latent_size, i)
            if input_size % 2 !=0:
                input_size = input_size + 1
            flow = self.build_flow(input_size, hidden_units)
            flows.append(flow)
            i = i // cut_size
        self.flows = nn.ModuleList(flows)
        self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, latent_size))
        
        self.normalized_state=normalized_state
        self.is_random = is_random
        
        
        
        
        #if sym_size % 2 !=0:
        #    sym_size = sym_size + 1
        #self.device = device
        #self.latent_size = latent_size
        #self.effect_size = effect_size
        #self.sym_size = sym_size
        #i = sym_size
        #flows = []
        #while i > latent_size:
        #    if i // cut_size <= latent_size:
        #        i = latent_size
        #    nets = lambda: nn.Sequential(nn.Linear(i, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, i), nn.Tanh())
        #    nett = lambda: nn.Sequential(nn.Linear(i, hidden_units), nn.LeakyReLU(), 
        #                                 nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                                 nn.Linear(hidden_units, i))
        #    mask1 = torch.cat((torch.zeros(1, i // 2, device=self.device), 
        #                       torch.ones(1, i // 2, device=self.device)), 1)
        #    mask2 = 1 - mask1
        #    masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        #    flow = InvertibleNN(nets, nett, masks, self.device)
        #    flows.append(flow)
        #    i = i // cut_size
        #self.flows = nn.ModuleList(flows)
        
        #self.dynamics = nn.Sequential(nn.Linear(latent_size, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
        #                             nn.Linear(hidden_units, latent_size))
        
        #self.normalized_state=normalized_state
        #self.is_random = is_random
        
        
    def build_flow(self, input_size, hidden_units):
        nets = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))

        mask1 = torch.cat((torch.zeros(1, input_size // 2, device=self.device), 
                           torch.ones(1, input_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.stack([mask1, mask2]*6, dim=0)
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow
    def build_dynamics(self, mid_size, hidden_units):
        dynamics = nn.Sequential(nn.Linear(mid_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, mid_size))
        return dynamics
        
    def forward(self, x):
        #state_dim = x.size()[1]
        
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        
        s = self.encoding(x)
        s_next = self.dynamics(s) + s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)/3
        y = self.decoding(s_next)
        return y, s, s_next
    def decoding(self, s_next):
        y = s_next
        for i in range(len(self.flows))[::-1]:
            flow = self.flows[i]
            end_size = self.latent_size
            if i < len(self.flows)-1:
                flow_n = self.flows[i+1]
                end_size = flow_n.size
            sz = flow.size - end_size
            if sz>0:
                noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((y.size()[0], 1))/3
                noise = noise.to(self.device)
                #print(noise.size(), s_next.size(1))
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        for i,flow in enumerate(self.flows):
            y,_ = flow.f(y)
            if self.normalized_state:
                y = torch.tanh(y)
            if i < len(self.flows)-1:
                lsize = self.flows[i+1].size
            else:
                lsize = self.latent_size
            y = y[:, :lsize]
        return y
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    def simulate(self, x):
        x_next = self.dynamics(x) + x
        decode = self.decoding(x_next)
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

#Learn more on Parallel
class Parellel_Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, cut_size, hidden_units,normalized_state,device,is_random=False):
        #latent_size: input size
        #effect_size: scale, effective latent dynamics size
        super(Parellel_Renorm_Dynamic, self).__init__()
        if latent_size < 1 or latent_size > sym_size:
            print('Latent Size is too small(<1) or too large(>input_size):', latent_size)
            return
        
        #Set the devcie
        self.device = device
        #Set the latent size, unknown
        self.latent_size = latent_size
        #set the effect size, unknown
        self.effect_size = effect_size
        #set the sym size unknown
        self.sym_size = sym_size
        #Take i to be the sym size, we need to do it since we need mutiple scales
        i = sym_size
        #Create flows
        flows = []
        #Create dynamic modules
        dynamics_modules = []
        #Create inverse dynamic modules
        inverse_dynamics_modules = []
        
        #Justify whether sym size is larger than latent size
        while i > latent_size:
            #to compare sym size and latent size
            input_size = max(latent_size, i)
            #Don't know why we need to repeat it since i don't change
            if i == sym_size:
                #Mid_size is also input size
                mid_size = sym_size
                #build dynamics
                dynamics = self.build_dynamics(mid_size, hidden_units)
                #append dynamics modules given build dynamic
                dynamics_modules.append(dynamics)
                #Add the inverse dynamics
                inverse_dynamics = self.build_dynamics(mid_size, hidden_units)
                inverse_dynamics_modules.append(inverse_dynamics)
                #build the ecndoing
                flow = self.build_flow(input_size, hidden_units)
                flows.append(flow)
            
            #build the flow again and append it
            flow = self.build_flow(input_size, hidden_units)
            flows.append(flow)
            #cut the middle size to make mutiple scale
            mid_size = max(latent_size, int(i // cut_size))
            #add another dynamics
            dynamics = self.build_dynamics(mid_size, hidden_units)
            dynamics_modules.append(dynamics)
            inverse_dynamics = self.build_dynamics(mid_size, hidden_units)
            inverse_dynamics_modules.append(inverse_dynamics)
            #Update i to update the scale
            i = int(i // cut_size)
         
        #Build the decodind
        self.flows = nn.ModuleList(flows)
        self.dynamics_modules = nn.ModuleList(dynamics_modules)
        self.inverse_dynamics_modules = nn.ModuleList(inverse_dynamics_modules)
        
        #see whther we need to normalize and is random
        self.normalized_state=normalized_state
        self.is_random = is_random
        
    #See why we need to build flow
    def build_flow(self, input_size, hidden_units):
        #change it to be the even number so that we can scale it with 2
        if input_size % 2 !=0 and input_size > 1:
            input_size = input_size - 1
        #Add the MLP
        nets = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))
        
        #Make masks
        mask1 = torch.cat((torch.zeros(1, input_size // 2, device=self.device), 
                           torch.ones(1, input_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2,
                           mask1, mask2, mask1, mask2, mask1, mask2), 0)
        #masks = torch.stack([mask1, mask2]*3, dim=0)
        
        #Then, make the invertible ne
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow
    
    #See how we build the flow
    def build_dynamics(self, mid_size, hidden_units):
        #The MLP layers for prediction
        dynamics = nn.Sequential(nn.Linear(mid_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, mid_size))
        return dynamics
        
    def forward(self, x, delay=1):
        #state_dim = x.size()[1]
        
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        #Encoding part
        ss = self.encoding(x)
        #However,we don't need to encode the first layer
        #ss[0]=x
        
        s_nexts = []
        ys = []
        
        for i,s in enumerate(ss):
            #update a version of prediction in several steps
            for t in range(delay):
                s_next = self.dynamics_modules[i](s) + s
                s=s_next
            
            #add the layer for normilization, not needed for brain data
            if self.normalized_state:
                s_next = torch.tanh(s_next)
            #default to be false, not needed
            if self.is_random:
                s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(),
                                                                                                 device=self.device)
            #What we mean by i and s
            #see more details in encdoing and decoding
            if i > 0:
                y = self.decoding(s_next, i)
            else:
                #y = s_next
                y = self.decoding(s_next, i)
            s_nexts.append(s_next)
            ys.append(y)
        return ys, ss, s_nexts
    
    def back_forward(self, x, where_to_weight,delay=1):
        #state_dim = x.size()[1]
        
        if len(x.size())<=1:
            x = x.unsqueeze(0)
            
        #Still, put the data in encoding1
        ss = self.encoding1(x)
        
        #However,we don't need to encode the first layer
        #ss[0]=x
        
        s_nexts = []
        ys = []
        for i,s in enumerate(ss):
            #Take he inverse dynamic
            for t in range(delay):
                s_next = self.inverse_dynamics_modules[i](s) + s
                s=s_next
            #if self.normalized_state:
            #s_next = torch.tanh(s_next)
            if self.is_random:
                s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(),
                                                                                                device=self.device)
            if i > 0:
                y = self.decoding1(s_next, i)
            else:
                #y = s_next
                y = self.decoding1(s_next, i)
            s_nexts.append(s_next)
            ys.append(y)
        return ys, ss, s_nexts
    
    def decoding(self, s_next, level):
        #Make the prediction
        y = s_next
        for i in range(level+1)[::-1]:
            flow = self.flows[i]
            end_size = self.latent_size
            if i < len(self.flows)-1:
                flow_n = self.flows[i+1]
                end_size = max(y.size()[1], flow_n.size)
            #print(flow.size, end_size, y.size()[1])
            sz = flow.size - end_size
            
            if sz>0:
                noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((y.size()[0], 1))/3
                noise = noise.to(self.device)
                #print(noise.size(), s_next.size(1))
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    
    def decoding1(self, s_next, level):
        y = s_next
        for i in range(level+1)[::-1]:
            flow = self.flows[i]
            end_size = self.latent_size
            if i < len(self.flows)-1:
                flow_n = self.flows[i+1]
                end_size = max(y.size()[1], flow_n.size)
            #print(flow.size, end_size, y.size()[1])
            sz = flow.size - end_size
            
            if sz>0:
                noise = distributions.MultivariateNormal(torch.zeros(sz), torch.eye(sz)).sample((y.size()[0], 1))/3
                noise = noise.to(self.device)
                #print(noise.size(), s_next.size(1))
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                #Make the torch cat
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        ys = []
        for i,flow in enumerate(self.flows):
            """
            if i==0:
                ys.append(y)
                continue
            """
            if y.size()[1] > flow.size:
                #y = torch.cat((y, y[:,:1]), 1)
                y = y[:, :flow.size]
            y,_ = flow.f(y)
            
            #if self.normalized_state:
            #y = torch.tanh(y)
            
            pdict = dict(self.dynamics_modules[i].named_parameters())
            lsize = pdict['0.weight'].size()[1]
            y = y[:, :lsize]
            ys.append(y)
        return ys
    
    def encoding1(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        ys = []
        for i,flow in enumerate(self.flows):
            """
            if i==0:
                ys.append(y)
                continue
            """
            if y.size()[1] > flow.size:
                #y = torch.cat((y, y[:,:1]), 1)
                y = y[:, :flow.size]
            y,_ = flow.f(y)
            #if self.normalized_state:
            #y = torch.tanh(y)
            pdict = dict(self.inverse_dynamics_modules[i].named_parameters())
            lsize = pdict['0.weight'].size()[1]
            y = y[:, :lsize]
            ys.append(y)
        return ys
    
    def loss(self, predictions, real, loss_f):
        losses = []
        sum_loss = 0
        for i, predict in enumerate(predictions):
            loss = loss_f(real, predict)
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(predictions)
    
    #See why we need loss weighhts
    #Not use the mean but the sum so that the loss won't look so small
    def loss_weights_general(self, predictions, real, where_to_weight,w, indexes,loss_f, forward_func):
        losses = []
        sum_loss = 0
        samples=1
        #samples=1
        #kernel = 'gaussian'
        #bandwidth = 0.05
        #atol = 0.2
        #L=100
    

        # Reshape the dataset so that it can be directly predicted
        # dataset = dataset.reshape(-1, dataset.shape[2])
        # Define where the predicts on the whole set
        if forward_func==self.forward:
            #predicts_whole, latents, latent_ps = forward_func(dataset)
            #predicts_whole_np = [predict.cpu().data.numpy() for predict in predicts_whole]
            for i, predict in enumerate(predictions):
                predict_np = predict.cpu().data.numpy()
                loss_mean=loss_f(real, predict).mean(1)
                if i in where_to_weight:
                    #In which dimension we care about
                    #scale=len(latents[i][0])
                    # Find out the kde function to fit
                    #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm='kd_tree').fit(predicts_whole_np[i])
                    #log_density  = torch.tensor(kde.score_samples(predicts_whole_np[i]))
                    #log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #均匀分布的概率分布
                    #logp = log_rho - log_density  #两种概率分布的差
            
                    # Calculate the weight and rescale it
                    #w = self.to_weights(logp, temperature=1)

                    #print(w.max())
                    loss = (loss_mean * w[indexes]*samples).mean()
                    #loss = ((loss_mean * w[indexes]).sum()/w[indexes].sum()+loss_mean.mean())/2
                else:
                    loss = loss_f(real, predict).mean()
                losses.append(loss)
                sum_loss += loss
        else:
            for i, predict in enumerate(predictions):
                predict_np = predict.cpu().data.numpy()
                loss_mean=loss_f(real, predict).mean(1)
                if i in where_to_weight:
                    #In which dimension we care about
                    #scale=len(latents[i][0])
                    # Find out the kde function to fit
                    #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm='kd_tree').fit(predicts_whole_np[i])
                    #log_density  = torch.tensor(kde.score_samples(predicts_whole_np[i]))
                    #log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #均匀分布的概率分布
                    #logp = log_rho - log_density  #两种概率分布的差
            
                    # Calculate the weight and rescale it
                    #w = self.to_weights(logp, temperature=1)

                    #print(w.max())
                    loss = (loss_mean * w[indexes]*samples).mean()
                    #loss = ((loss_mean * w[indexes]).sum()/w[indexes].sum()+loss_mean.mean())/2
                else:
                    loss = loss_f(real, predict).mean()
                losses.append(loss)
                sum_loss += loss
        return losses, sum_loss / len(predictions)

    def loss_weights(self, predictions, real, where_to_weight,w,indexes, loss_f):
        return self.loss_weights_general(predictions, real, where_to_weight,w,indexes,loss_f, self.forward)

    def loss_weights_back(self, predictions, real, where_to_weight,w, indexes,loss_f):
        return self.loss_weights_general(predictions, real, where_to_weight,w,indexes, loss_f, self.back_forward)
    
    #
    def calc_EIs(self, real, latent_ps, device):
        sp = self.encoding(real)
        eis = []
        sigmass = []
        scales = []
        for i,state in enumerate(sp):
            latent_p = latent_ps[i]
            flow = self.flows[i]
            dynamics = self.dynamics_modules[i]
            dd = dict(dynamics.named_parameters())
            scale = dd['0.weight'].size()[1]
            
            sigmas = torch.sqrt(torch.mean((state-latent_p)**2, 0))
            print(sigmas.shape)
            sigmas_matrix = torch.diag(sigmas)
            #Here we set L to be large
            ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                           num_samples = 1000, L=1, easy=True, device=device)
            eis.append(ei)
            sigmass.append(sigmas)
            scales.append(scale)
        return eis, sigmass, scales
    
    def to_weights(self,log_w, temperature=10):  #？重加权应该是个优化模型
        #将log_w做softmax归一化，得到权重
        logsoft = nn.LogSoftmax(dim = 0)
        weights = torch.exp(logsoft(log_w/temperature))
        return weights
    
    def kde_density(self,X):
        is_cuda = X.is_cuda  #True为储存在GPU
        ldev = X.device  #分配内存在哪里运行
        dim = X.size()[1] #获取数据的列数即维数
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.1, atol=0.005).fit(X.cpu().data.numpy())
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy())
        log_density = kde.score_samples(X.cpu().data.numpy())
        return log_density, kde
    
    def calc_EIs_kde(self,s,sp,samples,MAE_raw,L,bigL,device):
        #spring_data生成方式：spring_data = spring.generate_multistep(size=1000, steps=10, sigma=sigma, lam=1,miu=0.5)，为多步数据，样本点等于size*steps
        #samples跟spring_data样本点数一样，数越大运算会越慢，主要是kde估计很耗时间
        #MAE_raw = torch.nn.L1Loss(reduction='none')
        #L=bigL,根据隐空间范围进行调试，要保证能覆盖隐空间同时尽可能小
        #sigmas_matrix=torch.zeros([2,2],device=device)
        encodes=self.encoding(sp)
        predicts1, latent1s, latentp1s = self.forward(s)
        eis=[]
        sigmass = []
        weightss=[]
        for index in range(len(predicts1)):
            dynamics = self.dynamics_modules[index]
            #out the latent space
            latent1=latent1s[index]
            latentp1=latentp1s[index]
            #The scale
            scale=len(latent1[0])
            encode=encodes[index]
            
            #Do the reduction of dimenion first
            #Try to do the dimension reduction
            latent1=latent1.cpu().detach().numpy()
            #latents1=normalize(latents1)
            #sns.histplot(latent1.flatten())
            #plt.show()
            if latent1.shape[1]>10:
                scaler = StandardScaler()
                scaler.fit(latent1)
                latent1_zscore=scaler.transform(latent1)
                target_dim=10
                pca = PCA(n_components=target_dim)
                latent1_zscore= pca.fit_transform(latent1_zscore)
                #latent1_zscore=normalize(latent1_zscore)
                latent1_zscore=torch.tensor(latent1_zscore)
                log_density, k_model_n = self.kde_density(latent1_zscore)
            else:
                #latent1=normalize(latent1)
                latent1=torch.tensor(latent1)
                log_density, k_model_n = self.kde_density(latent1)

            #log_density, k_model_n = self.kde_density(latent1)
            #Take latent1 to be kde
            log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #均匀分布的概率分布
            #print(scale)
            #print(torch.log(2.0*torch.from_numpy(np.array(L))).shape)
            logp = log_rho - log_density  #两种概率分布的差
            #print(logp.max())
            weights = self.to_weights(logp, temperature=1) * samples
            #print(weights)
            #weights=torch.exp(logp)
            #print(weights)
            #weights=logp
            #print(torch.any(torch.isnan(weights)))
            #weights=(weights-weights.min())/(weights.max()-weights.min())
            #weights=weights/weights.sum()*200
            #print(weights.sum())
            
            #print(torch.any(torch.isnan(weights)))
            #sns.histplot(weights)
            #plt.show()
            if use_cuda:
                weights = weights.cuda(device=device)
            weights=weights.unsqueeze(1)
            #print(weights)
            #weights[weights>10]=10
            
            mae1 = MAE_raw(latentp1, encode)*weights
            #print(torch.any(torch.isnan(mae1)))
            #mae1 = MAE_raw(latentp1, encode) * torch.cat((weights,weights),1) 
            #mae1 = MAE_raw(torch.cat([latentp1 for i in range(len(latent1s[0][0]))],0), encode) * torch.cat(
            #    [torch.cat([weights for i in range(scale)],1) for j in range(len(latent1s[0][0]))],0)  
                #两维的情况，根据维度情况需要调整，weights的多维直接copy就行
            sigmas=mae1.mean(axis=0)
            sigmas_matrix = torch.diag(sigmas)
            ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                           num_samples = 1000, L=L, easy=True, device=device)  #approx_ei函数本身没有变化
    
            eis.append(ei)
            sigmass.append(sigmas)
            weightss.append(weights)
        return eis, sigmass,weightss
    
    #Out of the size that we think that will be intersting
    def eff_predict(self, prediction):
        return prediction[:, :self.effect_size]
    
    #Make the simulate
    def simulate(self, x, level):
        if level > len(self.dynamics_modules) or level<0:
            print('input error: level must be less than', len(self.dynamics_modules))
        dynamics = self.dynamics_modules[level]
        x_next = dynamics(x) + x
        decode = self.decoding(x_next, level)
        return x_next, decode
    
    def multi_step_prediction(self, s, steps, level):
        if level > len(self.dynamics_modules) or level<0:
            print('input error: level must be less than', len(self.dynamics_modules))
        s_hist = s
        ss = self.encoding(s)
        z_hist = ss[level]
        z = z_hist[:1, :]
        for t in range(steps):    
            z_next, s_next = self.simulate(z, level)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            z = z_next
        return s_hist, z_hist
    
    def multi_step_prediction_0(self, s, steps, level):
        if level > len(self.dynamics_modules) or level<0:
            print('input error: level must be less than', len(self.dynamics_modules))
        predicts=[]
        latents=[]
        s_hist = s
        #Get the result for dimension reduction
        ss = self.encoding(s)
        #choose the level that we are intersted in
        z_hist = ss[level]
        #Get all of data
        z = z_hist[:, :]
        predicts.append(s)
        #Here, the latents mean that the result for dimension reduction
        latents.append(z)
        for t in range(steps):
            #Make the prediction
            #Not sure why wee need hist here since it seems that we don'y utilize
            z_next, s_next = self.simulate(z, level)
            z_hist = torch.cat((z_hist, z_next), 0)
            s_hist = torch.cat((s_hist, self.eff_predict(s_next)), 0)
            predicts.append(s_next)
            latents.append(z_next)
            z = z_next
        return predicts, latents
    

#Since the neural data usually follows similar 
#The format of data should be two dimensional: MXN
#M:The number of time steps in time series
def train_and_memorize(data,data_name,method,epoches=5e6+1,hidden_units = 64,scale=1,batch_size =100,version=0,use_cuda_f=0):
    #data format: M x T x N
    #M: The number of subjects or experiments
    #T: The number of time steps
    #N: The number of nodes
    #What we should memorize during our training
    global use_cuda
    use_cuda=use_cuda_f
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    if version==0:
        #Remember the Effective Information
        EIs=[]
        #Remember the Causual Information
        CEs=[]
        #Remmber the training Losses
        LOSSes=[]
        L=1
        #The cut size for the whole system
        #cut_size=1.9
        #The cut size for the visual system
        cut_size=1.9
        #How many batch we want to the neural network to train
        epoches=int(epoches)
        #Hidden_units might be too less for our study 
        hidden_units = hidden_units
        #Define some seed for generating random values
        #torch.manual_seed(50)
        #However, since we shuffle the order, better not to set any seed
        #it relates the croaning,but now I don't understand why we need that
        scale = scale
        scales=scale_calculate(data.shape[2],[],cut_size)
        print(scales)
        #print(scales)
        #In our real data,the batch size is more a training technique instead of simulation parameter
        batch_size =batch_size
        #Since it's a time series prediction of real values,we will try to use MAE instead of other losses
        MAE = torch.nn.L1Loss()
        #MAE_raw = torch.nn.L1Loss(reduction='none')
        MSE = torch.nn.MSELoss(reduction='none')
        #Define the net
        net = Parellel_Renorm_Dynamic(sym_size = data.shape[2], latent_size = scale, effect_size = data.shape[2],
                             cut_size=cut_size, hidden_units = hidden_units, normalized_state = True, device=device)
        #net.load_state_dict(torch.load('data/AOMIC_preprocessed_schaefer_100_model30000.pkl',map_location=torch.device('cpu')))
        #saved_parameters =torch.load('data/AOMIC_resample_dict_learning_model65000.pkl',map_location=device)
        #net.load_state_dict(saved_parameters)
        #for name, param in net.flows[0].named_parameters():
            #print(param.requires_grad)
            #param.data = saved_parameters[name].data
            #param.requires_grad=saved_parameters[name].requires_grad
        #param.requires_grad = saved_parameters[name].requires_grad
        #change to cuda setting
        net = net.cuda() if use_cuda else net
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
        #With all of above are prepared,let's start to train
        #reshape the data so that it can work 
        for t in tqdm(range(epoches)):
            #add some time delays during trainings
            max_time_delay=1
            time_delay=int(np.random.choice(np.arange(1,6),1,replace=False,p=np.exp(-np.arange(1,6))/np.exp(-np.arange(1,6)).sum()))
            input=data[:,:data.shape[1]-time_delay,:].reshape(-1,data.shape[2])
            target=data[:,time_delay:data.shape[1],:].reshape(-1,data.shape[2])
            #Different from simulation data, we can't use a function to generate the data we need
            #Instead,we should use the random.choice to select indexes
            #subject_index=np.random.choice(data.shape[0],1,replace=False)
            #state_indexes=np.random.choice(data.shape[1]-1,batch_size,replace=False)
            #state=data[subject_index,state_indexes,:]
            #state_next=data[subject_index,state_indexes+1,:]
            state_indexes=np.random.choice(data.shape[0]*(data.shape[1]-time_delay),batch_size,replace=False)
            state=input[state_indexes,:]
            #print(net1.encoding(state[:,:-1])[-1].flatten().shape)
            #print(state[:,-1].shape)
            #state[:,-1]=net1.encoding(state[:,:-1])[-1].flatten()
            state_next=target[state_indexes,:]
            #state_next[:,-1]=net1.encoding(state_next[:,:-1])[-1].flatten()
            #the prediction
            predicts,latents,latent_ps=net.forward(state,delay=time_delay)
            #print([latents[i].abs().max() for i in range(len(latents))])
            #We should break the prediction into several steps
            losses_returned,loss1 = net.loss(predicts, state_next, MAE)
            optimizer.zero_grad()
            loss=loss1
            #loss1.backward(retain_graph=True)
            #loss1.backward()
            #Try to train a model according to different stages
            """
            if t<8000:
                loss=losses_returned[0]
            elif (t-8000)//8000<len(scales)-1:
                loss=losses_returned[(t-8000)//8000+1]
            else:
                loss=loss1
            #make some parameters untrainable after one procedure
            for i in range(len(scales)):
                if t>=8000*(i+1)+8000:
                    for name,parms in net.flows[i+1].named_parameters():
                        parms.requires_grad=False
            if t>=8000:
                for name,parms in net.flows[0].named_parameters():
                    parms.requires_grad=False
            """
            #start to train
            loss.backward()
            optimizer.step()

            #Define a file name as data name
            if t % 500 == 0:
                #randomly select 200 data for EI approximation
                #subject_index=np.random.choice(data.shape[0],1,replace=False)
                #state_indexes=np.random.choice(data.shape[1]-1,batch_size*2,replace=False)
                #state_for_EI=data[subject_index,state_indexes,:]
                #state_next_for_EI=data[subject_index,state_indexes+1,:]
                state_indexes=np.random.choice(data.shape[0]*(data.shape[1]-1),1000,replace=False)
                input=data[:,:data.shape[1]-1,:].reshape(-1,data.shape[2])
                target=data[:,1:data.shape[1],:].reshape(-1,data.shape[2])
                state_for_EI=input[state_indexes,:]
                state_next_for_EI=target[state_indexes,:]
                predicts_for_EI,latents_for_EI,latent_ps_for_EI=net(state_for_EI,delay=1)
                #Then,we approximate EI
                #calculate eis, the sample is around 3000
                eis,sigmass,weights = net.calc_EIs_kde(state_for_EI,state_next_for_EI,1000,MSE,L,L,device)
                #eis,sigmass,_ = net.calc_EIs(state_for_EI,latent_ps_for_EI,device)
                print('iter %s:' % t, 'loss = %.8f' % loss1)
                print('iter %s:' % t, 'loss = %.8f'% loss)
                print('iter %s:' % t, 'loss = ',[loss1.detach().item() for loss1 in losses_returned])
                print('scales:', [s for s in scales])
                print('dEIs = ', [ei[0] for ei in eis])
                print('sigmas =', [sigma.mean() for sigma in sigmass])
                print('Causal Emergence = ', [ei[0] - eis[0][0] for ei in eis])
                
                with open('data/'+data_name+'_'+method+'.txt', 'a') as file1:
                    file1.write(
                        str(t)+ ' ' +str(loss1.item()) + ' ' + str([s for s in scales]) + ' ' + str([ei[0] for ei in eis]) + ' ' + str([ei[0] - eis[0][0] for ei in eis]) + '\n')
                if t % 1000 == 0:
                    torch.save(net.state_dict(), 'data/'+data_name+'_'+method+'_model%s.pkl'%t)

                EIs.append([ei[0] for ei in eis])
                CEs.append([ei[0] - eis[0][0] for ei in eis])
                #Which loss is important might need investigation
                LOSSes.append(loss1)
                
                #arraylize all tensors and save 
                loss_array=np.array([LOSSes[i].item() for i in range(len(LOSSes))])
                np.save('AOMIC_preprocessed_Results/Losses_kde.npy',loss_array)
                EIs_array=np.array(EIs)
                np.save('AOMIC_preprocessed_Results/EIs_kde.npy',EIs_array)
                CEs_array=np.array(CEs)
                np.save('AOMIC_preprocessed_Results/CEs_kde.npy',CEs_array)   
                
                            
    elif version==1:
    #Remember the Effective Information
        EIs=[]
        #Remember the Causual Information
        CEs=[]
        #Remmber the training Losses
        LOSSes=[]
        #Not exactly how we define L
        L=1
        #How many batch we want to the neural network to train
        epoches=int(epoches)
        #Hidden_units might be too less for our study 
        hidden_units = hidden_units
        #Define some seed for generating random values
        #torch.manual_seed(50)
        #However, since we shuffle the order, better not to set any seed
        #it relates the croaning,but now I don't understand why we need that
        scale = scale
        #just force it to be this here
        #After trying for new results, try to test it
        cut_size=2
        scales=scale_calculate(data.shape[2],[],cut_size=cut_size)
        #We care about the second dimension for our analysis here
        #Since the importance is checked by the Parallel
        
        kernel = 'gaussian'
        bandwidth = 0.05
        atol = 0.2
        algorithm='kd_tree'
        #algorithm='ball_tree'
        #Define the weights for the loss
        #Here,we should kde all of sample values
        #After completing the kde functions, we can then generate weights based on the functions
        #Take random indexes to fit the function
        #indexes=np.random.choice(data.reshape(-1,data.size(2)).size(0)-1,batch_size*10,replace=False)
        #In our real data,the batch size is more a training technique instead of simulation parameter
        batch_size =batch_size
        #Since it's a time series prediction of real values,we will try to use MAE instead of other losses
        MAE = torch.nn.L1Loss(reduction='none')
        MSE = torch.nn.MSELoss(reduction='none')
        # MAE_raw = torch.nn.L1Loss()
        #Define the net
        net = Parellel_Renorm_Dynamic(sym_size = data.shape[2], latent_size = scale, effect_size = data.shape[2],
                             cut_size=2, hidden_units = hidden_units, normalized_state = True, device=device)
        #chage to cuda setting
        #net = net.cuda() if use_cuda else net
        #net.load_state_dict(torch.load('data/AOMIC_preprocessed_schaefer_100_model30000.pkl',map_location=torch.device('cpu')))
        net.load_state_dict(torch.load('AOMIC_preprocessed_Results/system/NIS_aal_cut=2_hidden=256_parallel_training=800/AOMIC_preprocessed_aal_model50000.pkl',map_location=torch.device('cpu')))
        #
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
        #reshape the data so that it can work 
        #input=data[:,:data.shape[1]-1,:].reshape(-1,data.shape[2])
        #target=data[:,1:data.shape[1],:].reshape(-1,data.shape[2])
        #With all of above are prepared,let's start to train
        for t in tqdm(range(epoches)):
            #
            max_time_delay=1
            time_delay=int(np.random.choice(np.arange(1,max_time_delay+1),1,replace=False,p=np.exp(-np.arange(1,max_time_delay+1))/np.exp(-np.arange(1,max_time_delay+1)).sum()))
            input=data[:,:data.shape[1]-time_delay,:].reshape(-1,data.shape[2])
            target=data[:,time_delay:data.shape[1],:].reshape(-1,data.shape[2])
            #
            where_to_weight=[0]
            """
            if t<8000:
                where_to_weight=[0]
            elif (t-8000)//8000<len(scales)-1:
                where_to_weight=[(t-8000)//8000+1]
            else:
                where_to_weight=[]
             """
            #Different from simulation data, we can't use a function to generate the data we need
            #Instead,we should use the random.choice to select indexes
            #subject_index=np.random.choice(data.shape[0],1,replace=False)
            #subject_index=np.random.choice(data.shape[0],1,replace=False)
            #state_indexes=np.random.choice(data.shape[1]-1,batch_size,replace=False)
            #state_indexes=np.random.choice(data.shape[0]*(data.shape[1]-1),batch_size,replace=False)
            state_indexes=np.random.choice(data.shape[0]*(data.shape[1]-time_delay),batch_size,replace=False)
            state=input[state_indexes,:]
            state_next=target[state_indexes,:]
            #state=data[subject_index,state_indexes,:]
            #state_next=data[subject_index,state_indexes+1,:]
            #the prediction
            predicts,latents,latent_ps=net(state,delay=time_delay)
            #Seed the net backward
            predicts_0,latents_0,latent_ps0=net.back_forward(state_next,where_to_weight,delay=time_delay)
            #Given the kde,let get the weight
            #weights=net.to_weights(torch.tensor(kde.score_samples(state.cpu()),device=device))
            #weights=[weights for i in range(int(np.log2(data.shape[2]))+1)]
            #calculate loss and loss/prediction
            #print((MAE(state_next,predict[0]).mean(1)*weights[0]).mean())
            #initilize the weights
            #w_forward=torch.ones(data.shape[1])
            #w_backward=w_forward
            if t==0:
                w_forward=torch.ones(data.shape[0]*data.shape[1])
                w_backward=torch.ones(data.shape[0]*data.shape[1])
            if t % 1000 == 0 and t!=(epoches-1) and t!=0:
                #initilize the weights     
                for i, predict in enumerate(predicts):
                    if i in where_to_weight:
                        #latents_whole=[]
                        #only care about one dimension
                        #Select people randomly to approximate the whole distribution
                        #available_numbers = np.setdiff1d(np.arange(data.shape[0]),subject_index)
                        #random_subject_index=np.random.choice(data.shape[0],data.shape[0],replace=False)
                        #random_subject_index=np.insert(random_subject_index, 0,subject_index)
                        #kde_models=[]
                        #latents_mean=[]
                        latents_kde=[]
                        #predicts_single, latents, latent_ps = net.forward(input)
                        for j in range(data.shape[0]):
                            predicts_single, latents, latent_ps = net.forward(data[j,:,:])
                            #latents_kde_norm=normalize(latents[i].cpu().data.numpy())
                            #latents=np.mean(latents[i].cpu().data.numpy(),0)
                            #latents_mean.append(latents)
                            latents_kde.append(latents[i].cpu().data.numpy())
                            #latents_kde.append(latents_kde_norm)
                            #kde=KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm=algorithm).fit(latents[i].cpu().data.numpy())
                            #kde_models.append(kde)
                        #print(len(latents_mean))
                        #predicts_whole=np.concatenate(predicts_whole,0)
                        latents_kde=np.concatenate(latents_kde,0)
                        #latents_kde=latents[i].cpu().data.numpy()
                        #Change latents_kde to be the right scale (0,1)
                        #Then change it to be the setting
                        #latents_kde_norm=normalize(latents_kde)
                        #Try to do the dimension reduction
                        if latents_kde.shape[1]>10:
                            scaler = StandardScaler()
                            scaler.fit(latents_kde)
                            #fit the norm data
                            latents_kde_1=scaler.transform(latents_kde)
                            target_dim=10
                            pca = PCA(n_components=target_dim)
                            latents_kde_1 = pca.fit_transform(latents_kde_1)
                        else:
                            latents_kde_1=latents_kde
                            #latents_kde_norm=normalize(latents_kde_1)
                            
                        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm=algorithm).fit(latents_kde_1)
                        #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm=algorithm).fit(latents_whole)
                        #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm=algorithm).fit(predicts_whole)
                        # Find out the kde function to fit
                        #The datasize is too large
                        #We need to split the dataset so that it can be processed
                        #Then calculate the log density
                        #Prepare for the data
                        # log_density=torch.tensor(log_density)
                        #log_density=torch.tensor(kde.score_samples(latents_whole))
                        log_density=torch.tensor(kde.score_samples(latents_kde_1))
                        #log_densities  = torch.tensor(kde.score_samples(predicts_whole_np[i]))
                        scale=len(latents[i][0])
                        #scale=len(latents)
                        log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #均匀分布的概率分布
                        logp = log_rho - log_density  #两种概率分布的差
                        # Calculate the weight and rescale it
                        w_forward = net.to_weights(logp, temperature=1)*data.shape[0]*data.shape[1]
                        #w_forward[w_forward>10]=10
                        #w_forward = torch.exp(logp)
                        #w_forward=(w_forward-w_forward.min())/(w_forward.max()-w_forward.min())
                        #w_forward=w_forward/w_forward.sum()*data.shape(0)*data.shape(1)
                #Make the backward weight to be same with the forward weight
                w_backward = w_forward 
                sns.histplot(w_forward)
                plt.xlim([0,5])
                plt.show()   
            #Then we start to run
            losses_returned,loss1 = net.loss_weights(predicts, state_next,where_to_weight,w_forward,state_indexes,MAE)
            #Back to make the prediction
            losses_returned_0,loss2 = net.loss_weights_back(predicts_0,state,where_to_weight,w_backward,state_indexes, MAE)
            #sum up bidirection loss
            loss=losses_returned[0]+0.5*losses_returned_0[0]
            """
            if t<8000:
                loss=losses_returned[0]+losses_returned_0[0]
            elif (t-8000)//8000<len(scales)-1:
                loss=losses_returned[(t-8000)//8000+1]+losses_returned_0[(t-8000)//8000+1]
            else:
                loss=loss1+loss2
            #make some parameters untrainable after one procedure
            for i in range(len(scales)):
                if t>=8000*(i+1)+8000:
                    for name,parms in net.flows[i+1].named_parameters():
                        parms.requires_grad=False
            if t>=8000:
                for name,parms in net.flows[0].named_parameters():
                    parms.requires_grad=False
            """
                    
            #loss=torch.tensor(losses_returned)+torch.tensor(losses_returned_0)
            optimizer.zero_grad()
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            #Define a file name as data name
            if t % 200 == 0:
                #randomly select 200 data for EI approximation
                #subject_index=np.random.choice(data.shape[0],1,replace=False)
                #state_indexes=np.random.choice(data.shape[1]-1,batch_size*2,replace=False)
                #state_for_EI=data[subject_index,state_indexes,:]
                #state_next_for_EI=data[subject_index,state_indexes+1,:]
                state_indexes=np.random.choice(data.shape[0]*(data.shape[1]-1),1000,replace=False)
                input=data[:,:data.shape[1]-1,:].reshape(-1,data.shape[2])
                target=data[:,1:data.shape[1],:].reshape(-1,data.shape[2])
                state_for_EI=input[state_indexes,:]
                state_next_for_EI=target[state_indexes,:]
                #state_for_EI=torch.tensor(data.cpu().data.numpy()[subject_index,state_indexes,:].reshape(1000,-1),device=device).to(torch.float32)
                #state_next_for_EI=torch.tensor(data.cpu().data.numpy()[subject_index,state_indexes+1,:].reshape(1000,-1),device=device).to(torch.float32)
                #Then,we approximate EI
                eis_kde,sigmass_kde,weights = net.calc_EIs_kde(state_for_EI,state_next_for_EI,1000,MSE,L,L,device)
                print('iter %s:' % t, 'loss = %.8f' % (loss1+loss2))
                print('iter %s:' % t, 'loss = %.8f'% loss)
                print('iter %s:' % t, 'forward loss = ',[loss0.detach().item() for loss0 in losses_returned])
                print('iter %s:' % t, 'backward loss = ',[loss0.detach().item() for loss0 in losses_returned_0])
                print('scales:', [s for s in scales])
                print('dEIs kde = ', [ei_kde[0] for ei_kde in eis_kde])
                print('sigmas kde=', [sigma_kde.mean() for sigma_kde in sigmass_kde])
                print('Causal Emergence kde = ', [ei_kde[0] - eis_kde[0][0] for ei_kde in eis_kde])
                
                EIs.append([ei[0] for ei in eis_kde])
                CEs.append([ei[0] - eis_kde[0][0] for ei in eis_kde])
                #Which loss is important might need investigation
                LOSSes.append(loss)
                
                #arraylize all tensors and save 
                loss_array=np.array([LOSSes[i].item() for i in range(len(LOSSes))])
                np.save('AOMIC_preprocessed_Results/Losses_kde.npy',loss_array)
                EIs_array=np.array(EIs)
                np.save('AOMIC_preprocessed_Results/EIs_kde.npy',EIs_array)
                CEs_array=np.array(CEs)
                np.save('AOMIC_preprocessed_Results/CEs_kde.npy',CEs_array)     
                
        
                #print(LOSSes)
                
                with open('data/'+data_name+'_'+method+'_'+'kde.txt', 'a') as file1:
                    file1.write(
                        str(t)+ ' ' +str(loss.item()) + ' ' + str([ei[0] for ei in eis_kde]) + '\n')
                if t % 1000 == 0:
                    torch.save(net.state_dict(), 'data/'+data_name+'_'+method+'_model_kde%s.pkl'%t)
            if t==epoches-1:
                #A final save
                torch.save(net.state_dict(), 'AOMIC_preprocessed_Results/'+data_name+'_'+method+'_model_kde%s.pkl'%t)
                #arraylize all tensors and save 
                loss_array=np.array([LOSSes[i].item() for i in range(len(LOSSes))])
                np.save('AOMIC_preprocessed_Results/Losses_kde.npy',loss_array)
                EIs_array=np.array(EIs)
                np.save('AOMIC_preprocessed_Results/EIs_kde.npy',EIs_array)
                CEs_array=np.array(CEs)
                np.save('AOMIC_preprocessed_Results/CEs_kde.npy',CEs_array)
            
    return EIs,CEs,LOSSes

#For figures
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def generate_color(dimensions):
    #Transform dimension into length
    length=int(np.log2(dimensions))+1
    colors=[]
    #Then, we make a color list
    for j in range(length):
        #Make a color by this one line
        colors.append('#%06X' % random.randint(0, 0xFFFFFF))
    return colors;

def read_and_visulize(EIs_array,CEs_array,loss_array,dim,path_name,method,version=0):
    scales=scale_calculate(dim,[],cut_size=1.9)
    if version==0:
        #plot losses
        len_loss=len(loss_array)
        plt.plot([500*i for i in range(len_loss)],loss_array)
        plt.xlabel('epoches')
        plt.ylabel('Losses')
        plt.show()
        #generate color acccording to its
        col1=generate_color(dim)
        for i in range(EIs_array.shape[1]):
            plt.plot([500*i for i in range(len(EIs_array))],EIs_array[0:len(EIs_array),i],color=col1[i],alpha=0.1)
        #where we want to start to draw
        length=2
        for i in range(EIs_array.shape[1]):
            plt.plot([500*i for i in range(length,len(EIs_array)-length)],moving_average(EIs_array[0:len(EIs_array),i],2)[length:len(EIs_array)-length],label=scales[i],color=col1[i])
        plt.legend()
        plt.xlabel('epoch',fontsize=12)
        plt.ylabel('dEI',fontsize=12)
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        plt.savefig(path_name+'/'+method+'_EIs.svg', dpi=600, format='svg')
        plt.show()
        #
        for i in range(CEs_array.shape[1]):
            plt.plot([500*i for i in range(len(CEs_array))],CEs_array[0:len(CEs_array),i],color=col1[i],alpha=0.1)
        #where we want to start to draw
        for i in range(CEs_array.shape[1]):
            plt.plot([500*i for i in range(length,len(CEs_array)-length)],moving_average(CEs_array[0:len(CEs_array),i],2)[length:len(CEs_array)-length],label=scales[i],color=col1[i])
        plt.legend()
        plt.xlabel('epoch',fontsize=12)
        plt.ylabel('Causal Emergence',fontsize=12)
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        plt.savefig(path_name+'/'+method+'_CEs.svg', dpi=600, format='svg')
        plt.show()
    elif version==1:
        #plot losses
        len_loss=len(loss_array)
        plt.plot([200*i for i in range(len_loss)],loss_array)
        plt.xlabel('epoches')
        plt.ylabel('Losses_kde')
        plt.show()
        #generate color acccording to its
        col1=generate_color(dim)
        for i in range(EIs_array.shape[1]):
            plt.plot([200*i for i in range(len(EIs_array))],EIs_array[0:len(EIs_array),i],color=col1[i],alpha=0.1)
        #where we want to start to draw
        length=10
        for i in range(EIs_array.shape[1]):
            plt.plot([200*i for i in range(length,len(EIs_array)-length)],moving_average(EIs_array[0:len(EIs_array),i],2)[length:len(EIs_array)-length],label=scales[i],color=col1[i])
        plt.legend()
        plt.xlabel('epoch',fontsize=12)
        plt.ylabel('dEI',fontsize=12)
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        plt.savefig(path_name+'/'+method+'_EIs_kde.svg', dpi=600, format='svg')
        plt.show()
        #
        for i in range(CEs_array.shape[1]):
            plt.plot([200*i for i in range(len(CEs_array))],CEs_array[0:len(CEs_array),i],color=col1[i],alpha=0.1)
        #where we want to start to draw
        for i in range(CEs_array.shape[1]):
            plt.plot([200*i for i in range(length,len(CEs_array)-length)],moving_average(CEs_array[0:len(CEs_array),i],2)[length:len(CEs_array)-length],label=scales[i],color=col1[i])
        plt.legend()
        plt.xlabel('epoch',fontsize=12)
        plt.ylabel('Causal Emergence',fontsize=12)
        plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        plt.savefig(path_name+'/'+method+'_CEs_kde.svg', dpi=600, format='svg')
        plt.show()
        
#
def scale_calculate(dim,scale,cut_size):
    if dim==1:
        scale.append(dim)
    else:
        scale.append(dim)
        dim=int(dim//cut_size)
        #print(dim)
        scale_calculate(dim,scale,cut_size)
    return scale


def normalize(data,target_range=(-1, 1)):
    """
    归一化函数，将输入的三维数组中的每个第三维数据归一化，并将归一化后的和缩放为1
    :param data: 输入的三维数组
    :param dim: 归一化的维度
    :param target_range: 归一化的目标范围，默认为(-1, 1)
    :return: 归一化后的三维数组
    """
    # 获取数据的形状
    shape = data.shape
    # 计算数据的最小值和最大值
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    #eps = 1e-16  # 设置一个很小的数
    # 计算归一化后的数据
    norm_data = (data - min_val) / (max_val - min_val) * (target_range[1] - target_range[0]) + target_range[0]
    """
    # 对归一化后的每个第三维数据进行缩放，使其和为1
    norm_data_sum = np.sum(norm_data, axis=dim)+eps
    if dim == 2:
        norm_data = norm_data / norm_data_sum[:, :, np.newaxis]
    if dim == 1:
        norm_data = norm_data / norm_data_sum[:, np.newaxis, :]
    # 返回归一化后的数据
    """
    return norm_data

def normalize_3d(data,target_range=(-1, 1)):
    """
    归一化函数，将输入的三维数组中的每个第三维数据归一化，并将归一化后的和缩放为1
    :param data: 输入的三维数组
    :param dim: 归一化的维度
    :param target_range: 归一化的目标范围，默认为(-1, 1)
    :return: 归一化后的三维数组
    """
    # 获取数据的形状
    shape = data.shape
    #data=np.swapaxes(data,1,2)
    # 计算数据的最小值和最大值
    #min_val = np.min(data, axis=1)
    #max_val = np.max(data, axis=1)
    mean_val=np.mean(data,axis=1)
    #The reason why it's 17 is because we need to rescale all of it to be [-1,1]
    std_val=np.std(data,axis=1)*16
    eps = 1e-32  # 设置一个很小的数
    #data=data.reshape(-1,shape[1])
    # 计算归一化后的数据
    #norm_data = (data - min_val[:,np.newaxis,:]) / (max_val[:,np.newaxis,:] - min_val[:,np.newaxis,:]+eps) * (target_range[1] - target_range[0]) + target_range[0]
    norm_data=(data-mean_val[:,np.newaxis,:])/(std_val[:,np.newaxis,:]+eps)
    #norm_data=np.swapaxes(norm_data,1,2)
    """
    # 对归一化后的每个第三维数据进行缩放，使其和为1
    norm_data_sum = np.sum(norm_data, axis=dim)+eps
    if dim == 2:
        norm_data = norm_data / norm_data_sum[:, :, np.newaxis]
    if dim == 1:
        norm_data = norm_data / norm_data_sum[:, np.newaxis, :]
    # 返回归一化后的数据
    """
    return norm_data