import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from EI_calculation import approx_ei_nn
import math


class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask, device):
        super(InvertibleNN, self).__init__()
        self.device = device
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size()[0] // 2
        self.t = torch.nn.ModuleList([nett() for _ in range(length)])  # repeating len(masks) times
        self.s = torch.nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size()[1]

    def g(self, z):
        x = z
        log_det_J = x.new_zeros((x.shape[0], x.shape[1], x.shape[2]), device=self.device)
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=3)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros((x.shape[0], x.shape[1], x.shape[2]), device=self.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=3)
        return z, log_det_J

class Parellel_Renorm_Dynamic(nn.Module):
    def __init__(self, spatial_size, spatial_scale, effect_size, time_size, time_scale, grid_size, sub_grid_size, hidden_units,normalized_state,device,is_random=False):
        super(Parellel_Renorm_Dynamic, self).__init__()
        if spatial_scale < 1 or spatial_scale > spatial_size:
            print('Latent Size is too small(<1) or too large(>input_size):', spatial_scale)
            raise
            return
        self.device = device            
        self.dim = spatial_scale           # 1
        self.effect_size = effect_size     # 9
        self.sp_vnp_size = spatial_size     # 9
        self.time_size = time_size        # 2
        self.time_scale = time_scale      # 2
        self.tm_vnp_size = time_scale     # 2
        self.grid_size = grid_size            # 18
        self.sub_grid_size = sub_grid_size    # 3

        if spatial_size % 2 != 0:
            self.sp_vnp_size = spatial_size + 1
        if time_scale % 2 != 0:
            self.tm_vnp_size = time_scale + 1
        i = grid_size
        spatial_flows = []
        time_flows = []
        dynamics_modules = []
        inverse_dynamics_modules = []
        j = time_size
        while i // sub_grid_size >= 2 and i % sub_grid_size == 0 and j // time_scale >= 1 and j % time_scale == 0:    
            if i == grid_size:
                spatial_flow = self.build_flow(self.sp_vnp_size, hidden_units)
                spatial_flows.append(spatial_flow)
                time_flow = self.build_flow(self.tm_vnp_size, hidden_units)
                time_flows.append(time_flow)

            spatial_flow = self.build_flow(self.sp_vnp_size, hidden_units)
            spatial_flows.append(spatial_flow)
            time_flow = self.build_flow(self.tm_vnp_size, hidden_units)
            time_flows.append(time_flow)

            j = j // time_scale
            i = i // sub_grid_size
            dynamics = self.build_dynamics(i*i, hidden_units)
            dynamics_modules.append(dynamics)

            inverse_dynamics = self.build_dynamics(i*i, hidden_units)
            inverse_dynamics_modules.append(inverse_dynamics)
            
        self.spatial_flows = nn.ModuleList(spatial_flows)
        self.time_flows = nn.ModuleList(time_flows)

        self.dynamics_modules = nn.ModuleList(dynamics_modules)
        self.inverse_dynamics_modules = nn.ModuleList(inverse_dynamics_modules)
        
        self.normalized_state=normalized_state
        self.is_random = is_random
    
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
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2), 0)
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow

    def build_dynamics(self, input_size, hidden_units):
        dynamics = nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))
        return dynamics

    def forward(self, x):        
        ss = self.encoding(x)
        s_nexts = []
        ys = []
        for i,s in enumerate(ss):
            if i > 0:
                s_next = self.dynamics_modules[0](s) + s
                if self.normalized_state:
                    s_next = torch.tanh(s_next)
                if self.is_random:
                    s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
                #[100,6*6]-->[100,1,36,1]
                y = self.decoding(s_next.reshape(s_next.size()[0], 1, -1, self.dim), i+1)
                s_nexts.append(s_next)
                ys.append(y)
        return ys, ss, s_nexts

    def back_forward(self, x):        
        ss = self.encoding(x)
        s_nexts = []
        ys = []
        for i,s in enumerate(ss):
            if i > 0:
                s_next = self.inverse_dynamics_modules[0](s) - s
                if self.normalized_state:
                    s_next = torch.tanh(s_next)
                if self.is_random:
                    s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
                s_nexts.append(s_next)  
        return ys, ss, s_nexts

    def convert_to_matrix_data(self, y_hat, s, state_size): 
        length = 0
        height = 0
        for j in range(1, s.size()[2] + 1):
            y_hat[:, :, height:height + self.sub_grid_size, length:length + self.sub_grid_size] = s[:,:,j-1]
            length += self.sub_grid_size
            if j % state_size == 0:
                height += self.sub_grid_size
                length = 0
        return y_hat

    def convert_to_realvnp_data(self, state_size, sub_state, s):
        sub_number = 0
        for k in range(0, state_size, self.sub_grid_size):
            for p in range(0, state_size, self.sub_grid_size):
                sub_state[:,:,sub_number] = s[:,:,k:k + self.sub_grid_size, p:p + self.sub_grid_size].reshape(s.size()[0], s.size()[1], -1)
                sub_number += 1
        return sub_state
    
    def decoding(self, s_next, level):
        y = s_next
        yy = s_next  # [100,1,36,1]
        for i in range(level)[::-1]:
            sp_flow = self.spatial_flows[i]
            tm_flow = self.time_flows[i] 
            if i == 0:
                 # [100, 2, 36, 9] 
                zz = y.transpose(1,2)     # [100, 36, 2, 9] 
                zz = zz.transpose(2,3)    # [100,36,9,2]
                zz, _ = tm_flow.g(zz)    # [100,36,9,2]
                zz = zz.transpose(2,3)    # [100,36,2,9] 
                out_zz = zz.transpose(1,2)  # [100,2,36,9]
                yy = torch.zeros((out_zz.size()[0], out_zz.size()[1], 18, 18), device=self.device)  # [100,2,18,18]
                sp_sz = 1  
                noise = distributions.MultivariateNormal(torch.zeros(sp_sz), torch.eye(sp_sz)).sample((out_zz.size()[0], out_zz.size()[1], out_zz.size()[2]))
                noise = noise.to(self.device)  # [100, 2, 36, 1]
                out_zz = torch.cat((out_zz, noise), 3)  # [100, 2, 36, 10]
                zz, _ = sp_flow.g(out_zz)        # [100, 2, 36, 10]
                if self.effect_size % 2 != 0:
                    zz = zz[:,:,:,:-1]      # [100, 2, 36, 9]                         # [100, 2, 36, 3, 3]
                yy = self.convert_to_matrix_data(yy, zz.reshape(zz.size()[0], zz.size()[1], zz.size()[2], self.sub_grid_size, self.sub_grid_size), 6)
            else:
                    # [100,2,36,1]
                # out_zz = torch.zeros((y.size()[0], y.size()[1] * self.time_scale, y.size()[2], 1), device=self.device)
                tm_sz = self.tm_vnp_size - y.size()[3] # 1 = 2-1 
                noise = distributions.MultivariateNormal(torch.zeros(tm_sz), torch.eye(tm_sz)).sample((y.size()[0], y.size()[1], y.size()[2]))
                noise = noise.to(self.device) # [100, 1, 36, 1]
                zz = torch.cat((y, noise), 3) # [100, 1, 36, 2]
                zz, _ = tm_flow.g(zz) # [100, 1, 36, 2]
                if self.time_scale % 2 != 0:
                    zz = zz[:,:,:-1] 
                zz = zz.transpose(2,3) # [100, 1, 2, 36]
                out_zz = zz.reshape(zz.size()[0], zz.size()[1]*zz.size()[2], zz.size()[3], 1) # [100, 2, 36, 1]
                state_size = int(math.sqrt(out_zz.size()[2])) # 6                   # [100, 2, 18, 18]
                yy = torch.zeros((out_zz.size()[0], out_zz.size()[1], state_size*self.sub_grid_size, state_size*self.sub_grid_size), device=self.device)
                sp_sz = self.sp_vnp_size - self.dim  # 9 = 10 -1
                noise = distributions.MultivariateNormal(torch.zeros(sp_sz), torch.eye(sp_sz)).sample((out_zz.size()[0], out_zz.size()[1], out_zz.size()[2]))
                noise = noise.to(self.device)  # [100, 2, 36, 9]
                out_zz = torch.cat((out_zz, noise), 3)  # [100, 2, 36, 10]
                zz, _ = sp_flow.g(out_zz)        # [100, 2, 36, 10]
                if self.effect_size % 2 != 0:
                    zz = zz[:,:,:,:-1]      # [100, 2, 36, 9]   
                y = zz
                                 
        return yy.reshape(yy.size()[0], yy.size()[1], -1) # [100, 2, 18*18]
    
    def encoding(self, x):
        ys = []
        for i,sp_flow in enumerate(self.spatial_flows):
            if i == 0:
                xx = x
                if len(x.size()) > 1:
                    if x.size()[3] < self.sp_vnp_size:# [100,2,36,9]-->[100,2,36,10]
                        xx = torch.cat((x, torch.zeros([x.size()[0], x.size()[1], x.size()[2], self.sp_vnp_size - x.size()[3]],device=self.device)), 3)
                y, _ = sp_flow.f(xx) # [100,2,36,10]
                s = y[:, :, :, :self.effect_size]  # [100,2,36,9]
                s = s.transpose(1,2)      # [100,36,2,9]
                s = s.transpose(2,3) # [100,36,9,2]    
                tm_flow = self.time_flows[i]
                s, _ = tm_flow.f(s)               # [100,36,2,9]        
                s = s[:,:,:,:self.time_scale].transpose(2,3)
                s = s.transpose(1,2)     # [100,2,36,9]
                ys.append(s)
            else:
                xx = s
                if len(x.size()) > 1:
                    if x.size()[3] < self.sp_vnp_size:# [100,2,36,9]-->[100,2,36,10]
                        xx = torch.cat((x, torch.zeros([x.size()[0], x.size()[1], x.size()[2], self.sp_vnp_size - x.size()[3]],device=self.device)), 3)
                y, _ = sp_flow.f(xx) # [100,2,36,10]
                s = y[:, :, :, :self.dim]  # [100,2,36,1]                                # [100,1,2,36]-->[100,1,36,2]
                s = s.reshape(s.size()[0], s.size()[1] // self.time_scale, self.time_scale, s.size()[2]).transpose(2, 3) 
                state_size = int(math.sqrt(s.size()[2]))  # 6
                out_s = torch.zeros((s.size()[0], s.size()[1], state_size, state_size), device=self.device) #[100,1,6,6]
                tm_flow = self.time_flows[i]
                if self.time_scale %2 != 0: # [100,1,36,2]
                    s = torch.cat((s, torch.zeros([s.size()[0], s.size()[1], s.size()[2], self.tm_vnp_size - s.size()[3]], device=self.device)), 3)
                s, _ = tm_flow.f(s)
                out_s = s[:,:,:,:self.dim].reshape(s.size()[0], state_size * state_size) # [100,1,36,1]-->[100,6*6]
                ys.append(out_s)
        return ys
    
    def loss(self, predictions, real, loss_f):
        losses = []
        sum_loss = 0
        for i, predict in enumerate(predictions):
            loss = loss_f(predict, real[:,i,:,:])
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(predictions)
    
    def loss_red(self, predictions, real, w, loss_f):
        losses = []
        sum_loss = 0
        for i, predict in enumerate(predictions):
            loss = (loss_f(predict, real, reduction='none').mean(axis=[1,2]) * w).mean()
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(predictions)

    def loss_inverse(self, latent1, latent2, loss_f):
        losses = []
        sum_loss = 0
        for i in range(len(latent1)):
            loss = loss_f(latent1[i], latent2[i])
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(latent1)
    
    def loss_inverse_red(self, latent1, latent2, w, loss_f):
        losses = []
        sum_loss = 0
        for i in range(len(latent1)):
            loss = (loss_f(latent1[i], latent2[i]).mean(axis=[1]) * w).mean()
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(latent1)
    
    def calc_EIs(self, real, latent, latent_ps):
        sp = []
        sp.append(self.encoding(real)[-1])
        eis = []
        for i, state in enumerate(sp):
            dynamics = self.dynamics_modules[i]
             #  state [50, 36]
            ei = approx_ei_nn(dynamics, state, latent[i], latent_ps[i], device=self.device)
            eis.append(ei)
        return eis 