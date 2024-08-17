import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from new_ei import EI
from new_ei import to_weights
from new_ei import kde_density
from datetime import datetime
t0 = datetime.now()


def cpt(s):
    #Timing function
    global t0
    t = datetime.now()
    print(f'check point{s:->10}-> {t.time()}; lasting {t - t0} seconds')
    t0 = t

    
class train_nis():
    def __init__(self, net, data, data_test, device):
        super().__init__()
        self.net = net.to(device=device)
        self.param_counts = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.x_t_all = data[0]
        self.x_t1_all = data[1]
        self.samp_num = self.x_t_all.size(0)
        self.weights = torch.ones(self.samp_num, device=device)
        self.MAE = nn.L1Loss()
        self.MAE_raw = nn.L1Loss(reduction='none')
        self.optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
        self.ei = EI().to(device=device)
        self.x_test = data_test[0]
        self.x_test_t1 = data_test[1]
        self.eis, self.term1s, self.term2s, self.train_losses, self.test_losses = [], [], [], [], []
        self.train_loss, self.test_loss = 0, 0
        
    def train_step(self, batch_size):
        self.net.train()
        start = np.random.randint(self.samp_num - batch_size)
        end = start + batch_size
        x_t, x_t1, w = self.x_t_all[start:end], self.x_t1_all[start:end], self.weights[start:end]
        x_t1_hat, _ = self.net(x_t, x_t1)
        loss = self.MAE(x_t1, x_t1_hat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        
    def test_step(self):
        self.net.eval()
        x_t1_hat, ei_items = self.net(self.x_test, self.x_test_t1)
        self.ei.update(ei_items) 
        dei, term1, term2 = self.ei.compute()
        loss = self.MAE(self.x_test_t1, x_t1_hat)
        return loss, dei, term1, term2

    def log(self, dei, term1, term2, epoch):
        cpt('step')
        print('Epoch:', epoch)
        print('Train loss: %.4f' %  self.train_loss.item())
        print('Train loss: %.4f' %  self.test_loss.item())
        print('dEI: %.4f' % dei.item())
        print('term1: %.4f'% term1.item())
        print('term2: %.4f'% term2.item())
        print(120*'-')
        
        self.eis.append(dei.item())
        self.term1s.append(term1.item())
        self.term2s.append(term2.item())
        self.train_losses.append(self.train_loss.item())
        self.test_losses.append(self.test_loss.item())
       
    def training(self, T_all, batch_size, clip=200):
        for epoch in range(T_all):
            self.train_loss += self.train_step(batch_size)
            if epoch % clip == 0:
                self.test_loss, dei, term1, term2 = self.test_step()
                self.train_loss /= clip
                self.log(dei, term1, term2, epoch)
                self.train_loss = 0

    def return_log(self):
        return self.eis, self.term1s, self.term2s, self.train_losses, self.test_losses

class train_nisp_rnis(train_nis):
    def __init__(self, net, data, data_test, device):
        super().__init__(net, data, data_test, device)

        self.temperature = None
        self.scale = None
        self.L = None

    def update_weight(self, h_t, L=1):
        samples = h_t.size(0)
        scale = h_t.size(1)
        n = h_t.shape[0]
        log_density = kde_density(h_t)
        log_rho = - scale * np.log(2.0 * L) 
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(torch.tensor(logp))
        weights = weights * samples
        self.weights = weights.cuda()

    def reweight(self):
        self.net.eval()
        h_t_all = self.net.encoding(self.x_t_all)
        self.update_weight(h_t_all)


    def train_step2(self, mae2_w, batch_size):
        self.net.train()
        start = np.random.randint(self.samp_num - batch_size)
        end = start + batch_size
        x_t, x_t1, w = self.x_t_all[start:end], self.x_t1_all[start:end], self.weights[start:end]
        x_t1_hat, ei_items = self.net(x_t, x_t1)
        h_t_hat = self.net.back_forward(x_t1)
        mae1 = (self.MAE_raw(x_t1, x_t1_hat).mean(axis=1) * w).mean() 
        mae2 = (self.MAE_raw(h_t_hat, ei_items['h_t']).mean(axis=1) * w).mean() 
        loss = mae1 + mae2_w * mae2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def training(self, T1, T_all, mae2_w, batch_size, clip=200):
        for epoch in range(T_all):
            if epoch < T1:
                self.train_loss += self.train_step(batch_size)
            else:
                self.train_loss += self.train_step2(mae2_w, batch_size)
            if epoch % clip == 0:
                self.test_loss, dei, term1, term2 = self.test_step()
                self.train_loss /= clip
                self.log(dei, term1, term2, epoch)
                self.train_loss = 0
            if epoch > T1 and epoch % 1000 == 0:
                self.reweight()
                print(self.weights[:10])