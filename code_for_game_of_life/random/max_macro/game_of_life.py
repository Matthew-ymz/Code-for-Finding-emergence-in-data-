import sys
sys.path.append("../../..")
import torch
from torch import nn
import models
import numpy as np
from models import Parellel_Renorm_Dynamic
from sklearn.neighbors import KernelDensity
import argparse
parser = argparse.ArgumentParser(description='Causal Emergence of Game of life')
# general settings
parser.add_argument('--number_of_random', type=int, default=0)
parser.add_argument('--number_of_perturb_glider', type=int, default=0)
parser.add_argument('--number_of_glider', type=int, default=2)
parser.add_argument('--Sample_Iter', type=int, default=500)
parser.add_argument('--T', type=int, default=500)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:10') if use_cuda else torch.device('cpu')
class Game_of_Life(object):
    def __init__(self, cell_size, sub_cell_size, time_size, level, number_of_random , number_of_perturb_glider, number_of_glider):
        self.cell_size = cell_size
        self.sub_cell_size = sub_cell_size
        self.time_size = time_size
        self.level = level
        self.sub_size = int(self.cell_size * self.cell_size / self.sub_cell_size / self.sub_cell_size)
        self.number_of_glider = number_of_glider
        self.number_of_perturb_glider = number_of_perturb_glider
        self.number_of_random = number_of_random
        self.glider = {0:torch.tensor([[0, 1, 0], [0,0, 1],[1,1,1]], device=device), 
                       1:torch.tensor([[1, 0, 1], [0,1, 1],[0,1,0]], device=device),
                       2:torch.tensor([[0, 0, 1], [1,0, 1],[0,1,1]], device=device), 
                       3:torch.tensor([[1, 0, 0], [0,1, 1],[1,1,0]], device=device)}
        assert self.number_of_random + self.number_of_glider + self.number_of_perturb_glider  <= self.sub_size

    def update_state(self, N, state, state_next):
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                neighbor_num = int((state[i, (j - 1) % N] + state[i, (j + 1) % N] +
                            state[(i - 1) % N, j] + state[(i + 1) % N, j] +
                            state[(i - 1) % N, (j - 1) % N] + state[(i - 1) % N, (j + 1) % N] +
                            state[(i + 1) % N, (j - 1) % N] + state[(i + 1) % N, (j + 1) % N]))
                if state[i, j] == 0:
                    if neighbor_num == 3:
                        state_next[i, j] = 1
                else:
                    if neighbor_num == 2 or neighbor_num == 3:
                        state_next[i, j] = 1
        return state_next

    def init_state(self):
        state = torch.zeros((self.cell_size, self.cell_size), device=device)
        state_temp_glider = torch.zeros((self.cell_size, self.cell_size), device=device)
        index = torch.tensor(np.array(range(0, self.cell_size-self.sub_cell_size)), device=device)
        number_glider = 0
        N = state.size()[0]
        while number_glider < self.number_of_glider:
            column, row = index[torch.multinomial(torch.tensor([1/index.size()[0] for _ in range(index.size()[0])]),1,replacement=False).item()].item(), index[torch.multinomial(torch.Tensor([1/index.size()[0] for _ in range(index.size()[0])]),1,replacement=False).item()].item()
            if state_temp_glider[(column-1)%N][(row-1)%N] == 0 and state_temp_glider[(column+3)%N][(row-1)%N] == 0  and state_temp_glider[(column+3)%N][(row+3)%N] == 0 and state_temp_glider[(column-1)%N][(row+3)%N] == 0:
                number_glider += 1
                state[column:column + self.sub_cell_size, row:row + self.sub_cell_size] = self.glider[torch.multinomial(torch.Tensor([1/4 for _ in range(4)]),1,replacement=False).item()]
                if column == 0 and row == 0:
                    state_temp_glider[column: column + self.sub_cell_size+1, row :row + self.sub_cell_size+1] = -1
                elif column == 0:
                    state_temp_glider[column: column + self.sub_cell_size+1, row-1 :row + self.sub_cell_size+1] = -1
                elif row == 0:
                    state_temp_glider[column-1: column + self.sub_cell_size+1, row:row + self.sub_cell_size+1] = -1
                else:
                    state_temp_glider[column-1: column + self.sub_cell_size+1, row-1:row + self.sub_cell_size+1] = -1
        return state
    
    def generate_data(self, level, iter_number, iter=120): 
        state_all = torch.zeros((args.Sample_Iter*20, iter_number, self.cell_size, self.cell_size))
        state_next_all = torch.zeros((args.Sample_Iter*20, level, iter_number, self.cell_size, self.cell_size))
        temp_state = torch.zeros((iter+3, self.cell_size, self.cell_size))
        number_id = 0
        for _ in range(args.Sample_Iter):
            state = torch.multinomial(torch.Tensor([0.5, 0.5]), self.cell_size*self.cell_size, replacement=True).reshape(self.cell_size, self.cell_size)
            N = state.shape[0]
            temp_state[0] = state
            for j in range(1, temp_state.size()[0]):
                state_next = torch.zeros(state.shape, device=device)
                state_next = self.update_state(N, state, state_next)
                state = state_next.clone()
                temp_state[j] = state
            for s in range(100, iter):
                state_all[number_id] = temp_state[s:s+iter_number]
                state_next_all[number_id][0] = temp_state[s+1:s+iter_number+1]
                state_next_all[number_id][1] = temp_state[s+iter_number:s+2*iter_number]
                number_id += 1
        return state_all, state_next_all
    

    def generate_data_test_uniform(self, number, level, iter_number):
        state_all = torch.zeros((number, iter_number, self.cell_size, self.cell_size))
        state_next_all = torch.zeros((number, level, iter_number, self.cell_size, self.cell_size))
        temp_state = torch.zeros((2*iter_number, self.cell_size, self.cell_size))
        for i in range(number):
            state = torch.multinomial(torch.Tensor([0.5, 0.5]), self.cell_size*self.cell_size, replacement=True).reshape(self.cell_size, self.cell_size)
            N = state.shape[0]
            temp_state[0] = state
            for j in range(1, 2*iter_number):
                state_next = torch.zeros(state.shape, device=device)
                state_next = self.update_state(N, state, state_next)
                state = state_next.clone()
                temp_state[j] = state
            state_all[i] = temp_state[:iter_number]
            state_next_all[i][0] = temp_state[1:iter_number+1]
            state_next_all[i][1] = temp_state[iter_number:]
        return state_all, state_next_all

    def convert_to_realvnp_data(self, data):
        sub_state = torch.zeros((data.size()[0], data.size()[1], self.sub_size, self.sub_cell_size * self.sub_cell_size))
        number = 0
        for i in range(0, self.cell_size, self.sub_cell_size):
            for j in range(0, self.cell_size, self.sub_cell_size):
                sub_state[:,:,number] = data[:,:,i:i + self.sub_cell_size, j:j + self.sub_cell_size].reshape(data.size()[0], data.size()[1], -1)
                number += 1
        return sub_state
    
    def sample(self):
        temp_state, temp_state_next = self.generate_data(self.level, self.time_size)
        state = torch.zeros([temp_state.size()[0], temp_state.size()[1], self.cell_size * self.cell_size])
        state_next = torch.zeros([temp_state.size()[0], self.level, temp_state.size()[1], self.cell_size * self.cell_size])
        sub_state = torch.zeros([temp_state.size()[0], temp_state.size()[1], self.sub_size, self.sub_cell_size * self.sub_cell_size])
        sub_state_next = torch.zeros([temp_state.size()[0], self.level, temp_state.size()[1], self.sub_size, self.sub_cell_size * self.sub_cell_size])
        state = temp_state.reshape(temp_state.size()[0], temp_state.size()[1], -1)
        sub_state = self.convert_to_realvnp_data(temp_state)
        for i in range(1, self.level):
            state_next[:,i] = temp_state_next[:,i].reshape(temp_state_next.size()[0], temp_state_next.size()[2], -1)
            sub_state_next[:,i] = self.convert_to_realvnp_data(temp_state_next[:,i])
        return state, state_next, sub_state, sub_state_next
    
    def sample_uniform(self, number):
        temp_state, temp_state_next = self.generate_data_test_uniform(number, self.level, self.time_size)
        state = torch.zeros([temp_state.size()[0], temp_state.size()[1], self.cell_size * self.cell_size])
        state_next = torch.zeros([temp_state.size()[0], self.level, temp_state.size()[1], self.cell_size * self.cell_size])
        sub_state = torch.zeros([temp_state.size()[0], temp_state.size()[1], self.sub_size, self.sub_cell_size * self.sub_cell_size])
        sub_state_next = torch.zeros([temp_state.size()[0], self.level, temp_state.size()[1], self.sub_size, self.sub_cell_size * self.sub_cell_size])
        state = temp_state.reshape(temp_state.size()[0], temp_state.size()[1], -1)
        sub_state = self.convert_to_realvnp_data(temp_state)
        for i in range(1, self.level):
            state_next[:,i] = temp_state_next[:,i].reshape(temp_state_next.size()[0], temp_state_next.size()[2], -1)
            sub_state_next[:,i] = self.convert_to_realvnp_data(temp_state_next[:,i])
        return state, state_next, sub_state, sub_state_next
    

def to_weights(log_w, temperature=1):
    logsoft = nn.LogSoftmax(dim = 0)
    weights = torch.exp(logsoft(log_w/temperature))
    return weights

def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.numpy())
    log_density = kde.score_samples(X.numpy())
    return log_density, kde

if __name__ == "__main__":
    hidden_units = 64
    # torch.manual_seed(1024)
    spatial_scale = 1
    batch_size = 50
    nll = nn.functional.binary_cross_entropy_with_logits
    L1 = torch.nn.L1Loss()
    nll_red = nn.functional.binary_cross_entropy_with_logits
    L1_red = torch.nn.L1Loss(reduction='none')
    sub_grid_size = 3
    grid_size = 18
    n_of_nods = 9
    time_scale = 2
    time_size = 2
    level = 2
    L = 100
    net = Parellel_Renorm_Dynamic(spatial_size=n_of_nods, spatial_scale=spatial_scale, effect_size=n_of_nods, time_size=time_size, time_scale=time_scale, grid_size=grid_size, sub_grid_size=sub_grid_size, hidden_units=hidden_units, normalized_state=False, device=device)
    net = net.to(device) if use_cuda else net
    optimizer1 = torch.optim.Adam([p for p in net.parameters() if p.requires_grad == True], lr=1e-4)
    optimizer2 = torch.optim.Adam([p for p in net.parameters() if p.requires_grad == True], lr=1e-4)
    game_of_life = Game_of_Life(cell_size=grid_size, sub_cell_size=sub_grid_size, time_size=time_size, level=level, number_of_random=args.number_of_random, number_of_perturb_glider=args.number_of_perturb_glider, number_of_glider=args.number_of_glider)    
    state, state_next, sub_state, sub_state_next = game_of_life.sample()
    weights = torch.ones(state.size()[0], device=device) 
    T = args.T
    
    for t in range(T+1):
        net.train()
        if (t+1) % 1000 == 0:
            idx = torch.randperm(state.shape[0])
            state, state_next, sub_state, sub_state_next, weights = state[idx,:], state_next[idx,:], sub_state[idx,:], sub_state_next[idx,:], weights[idx]
        start_range = state.size()[0]-batch_size
        start = torch.multinomial(torch.Tensor([1/start_range for _ in range(start_range)]),1,replacement=False).item()
        end = start + batch_size
        state_batch,state_next_batch,sub_state_batch,sub_state_next_batch, weight = state[start:end], state_next[start:end], sub_state[start:end], sub_state_next[start:end], weights[start:end]
        if use_cuda:
            state_batch = state_batch.to(device)
            state_next_batch = state_next_batch.to(device)
            sub_state_batch = sub_state_batch.to(device)
            sub_state_next_batch = sub_state_next_batch.to(device)
        
        predicts1, latents1, latent_ps1 = net(sub_state_batch)
        losses_returned1, loss1 = net.loss_red(predicts1, state_next_batch[:, 1], weight, nll_red)
        predicts2, latent2, latentp2 = net.back_forward(sub_state_next_batch[:,1]) 
        losses_returned2, loss2 = net.loss_inverse_red(latentp2, latents1[-1:], weight, L1_red)
        loss = loss1 + 0.01 * loss2
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        
        if t % 10000 == 0:
            torch.save(net.state_dict(), './model/model2/model_parameter_%s_macro_%s_%s_%s.pkl'%(t,args.number_of_random, args.number_of_perturb_glider, args.number_of_glider))
        
        if t > 0 and t % 10000 == 0:
            net_temp = Parellel_Renorm_Dynamic(spatial_size=n_of_nods, spatial_scale=spatial_scale, effect_size=n_of_nods, time_size=time_size, time_scale=time_scale, grid_size=grid_size, sub_grid_size=sub_grid_size, hidden_units=hidden_units, normalized_state=False, device=device)
            net_temp.load_state_dict(torch.load('./model/model2/model_parameter_%s_macro_%s_%s_%s.pkl'%(t,args.number_of_random, args.number_of_perturb_glider, args.number_of_glider)))
            net_temp = net_temp.to(device)
            encoding = torch.zeros_like(latents1[-1]).cpu().data
            sample_size = 1000
            for i in range(0, sub_state.size()[0], sample_size):
                temp_encoding = net_temp.encoding(sub_state[i:i+sample_size].to(device))
                encoding = torch.cat((encoding, temp_encoding[-1].cpu().data), 0)
            latent = encoding[batch_size:]
            scale = latent.size()[1]
            log_density, _ = kde_density(latent)
            log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))
            logp = log_rho - log_density
            weights = to_weights(logp) * state.size()[0]
            weights = weights.to(device)

        if  t % 10000 == 0:
            net.eval()
            state_temp, state_next_temp, sub_state_temp, sub_state_next_temp = game_of_life.sample_uniform(batch_size)
            if use_cuda:
                state_temp = state_temp.to(device)
                state_next_temp = state_next_temp.to(device)
                sub_state_temp = sub_state_temp.to(device)
                sub_state_next_temp = sub_state_next_temp.to(device)
            predicts, latents, latent_ps = net(sub_state_temp)
            eis = net.calc_EIs(sub_state_next_temp[:,1], latents[-1:], latent_ps)
            with open('./result1/macro_%s_%s_%s_1.txt'%(args.number_of_random, args.number_of_perturb_glider, args.number_of_glider), 'a') as file1:
                for loss_item in losses_returned1:
                    file1.write(str(loss_item.item())+ ' ')
                file1.write(str(loss1.item())+ ' ')
                for _ in range(len(eis)):
                    file1.write(str(eis[_]) + ' ')
                file1.write('\n')
            with open('./result1/macro_%s_%s_%s_2.txt'%(args.number_of_random, args.number_of_perturb_glider, args.number_of_glider), 'a') as file2:
                for loss_item in losses_returned2:
                    file2.write(str(loss_item.item())+ ' ')
                file2.write(str(loss2.item())+ ' ')
                for _ in range(len(eis)):
                    file2.write(str(eis[_]) + ' ')
                file2.write('\n')    
    
    