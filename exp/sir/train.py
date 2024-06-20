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
from dynamic_models_sis_new import calculate_multistep_predict
from ei.EI_calculation import test_model_causal_multi_sis
from ei.EI_calculation import to_weights
from ei.EI_calculation import kde_density
from models.models_new import Renorm_Dynamic
from datetime import datetime
t0 = datetime.now()
def cpt(s):
    #Timing function
    global t0
    t = datetime.now()
    print(f'check point{s:->10}-> {t.time()}; lasting {t - t0} seconds')
    t0 = t


def train(train_data, test_data, sz, scale, mae2_w, T2, T1 = 3001, encoder_interval = 1000, temperature=1, m_step = 10, test_start = 0, test_end = 0.3, sigma=0.03, rou=-0.5, L=1, hidden_units = 64, batch_size = 700, framework = 'nis'):
    MAE = torch.nn.L1Loss()
    MAE_raw = torch.nn.L1Loss(reduction='none')
    ss,sps,ls,lps = train_data
    sample_num = ss.size()[0] # batch_size * (steps * (k1 + k2) -1)
    weights = torch.ones(sample_num, device=device) 
    net = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
                         hidden_units = hidden_units, normalized_state=True, device = device)
    
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
            print('Train loss: %.4f' %  loss.item())
            print('dEI: %.4f' % ei1[0])
            print('term1: %.4f'% ei1[3])
            print('term2: %.4f'% ei1[4])
            print('Test multistep loss: %.4f'% mae_mstep)
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
            loss = (MAE_raw(sp, predicts1).mean(axis=1) * w).mean() 
            mae2 = (MAE_raw(latent1,latentp0).mean(axis=1) * w).mean() 
            loss_total = loss + mae2_w*mae2
            optimizer.zero_grad()
            loss_total.backward()
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
            print('Train loss: %.4f' %  loss.item())
            print('dEI: %.4f' % ei1[0])
            print('term1: %.4f'% ei1[3])
            print('term2: %.4f'% ei1[4])
            print('Test multistep loss: %.4f'% mae_mstep)
            print(120*'-')
  
            eis.append(ei1[0])
            term1s.append(ei1[3].item())
            term2s.append(ei1[4].item())
            losses.append(loss.item())
            MAEs_mstep.append(mae_mstep) 
            cpt('o_1')
            
    return eis, term1s, term2s, losses, MAEs_mstep, net