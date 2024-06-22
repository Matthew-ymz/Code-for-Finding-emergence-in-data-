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
from models_new import Renorm_Dynamic
from datetime import datetime
t0 = datetime.now()
def cpt(s):
    #Timing function
    global t0
    t = datetime.now()
    print(f'check point{s:->10}-> {t.time()}; lasting {t - t0} seconds')
    t0 = t


def train(train_data, test_data, sz, scale, mae2_w, T2, T1 = 3001, encoder_interval = 1000, temperature=1, m_step = 10, test_start = 0, test_end = 0.3, sigma=0.03, rou=-0.5, dt=0.01, L=1, hidden_units = 64, batch_size = 700, framework = 'nis'):
    MAE = torch.nn.L1Loss()
    MAE_raw = torch.nn.L1Loss(reduction='none')
    ss,sps,ls,lps = train_data
    sample_num = ss.size()[0] # batch_size * (steps * (k1 + k2) -1)
    weights = torch.ones(sample_num, device=device) 
    net = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
                         hidden_units = hidden_units, normalized_state=True, device = device)
    net.load_state_dict(torch.load('netwn_init_trnorm0.1+zero_seed=4.mdl').state_dict())
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
                mae_mstep += calculate_multistep_predict(net,s,i,steps=m_step, sigma=sigma, rou=rou, dt=dt)
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
                mae_mstep += calculate_multistep_predict(net,s,i,steps=m_step,sigma=sigma, rou=rou, dt=dt)
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

# def train_plus():
#     results = []
#     results_nn = []
#     encoder_results = []
#     temperature = 1
#     encoder_interval = 1000
#     experiments = 4
#     dt=0.01
#     sz = 4
#     scale = 2
#     L = 1
#     hidden_units = 64
#     sigma = 0.03
#     steps = 7
    
#     batch_size =  700 #20
#     mae2_w= 3
#     rou=-0.5
#     interval=1
#     cpt('begin')
#     z_scale=sz-scale
#     prior = distributions.MultivariateNormal(torch.zeros(z_scale), torch.eye(z_scale))
#     MAE = torch.nn.L1Loss()
#     MAE_raw = torch.nn.L1Loss(reduction='none')
#     test_start=0
#     test_end=0.3
#     m_step=10
#     steps2= 1
#     noise2=0
#     w_steps2=torch.exp(-torch.tensor(range(steps2),device=device)/5)
#     w_steps2=w_steps2.unsqueeze(1).expand(-1, batch_size).reshape(-1)
#     for experiment in range(experiments):
#         noise2=0.1+(experiment+1)*0.01
#         mul_batch_size = [4500,4500]
#         seed = 1+experiment
#         mae3_w = 0 
        
#         T  = 3001
#         T2  = 30001 
#         T3  = 150001
    
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
    
#         # Generating Data
#         spring = Simple_Spring_Model(device=device)
#         spring_data = spring.generate_multistep_sir(size_list=[500,500], steps=10*interval, sigma=sigma, rou=rou,lam=1,miu=0.5,dt=dt,noise2=noise2,interval=interval) #sir
#         ss,sps,ls,lps = spring.generate_multistep_sir(size_list=mul_batch_size, steps=steps, sigma=sigma,rou=rou,lam=1,miu=0.5,dt=dt,noise2=noise2,interval=interval) #sir
    
#         sample_num = ss.size()[0] 
#         weights = torch.ones(sample_num, device=device) 
    
#         net = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
#                              hidden_units = hidden_units, normalized_state=True, device = device)
    
    
#         #net.load_state_dict(torch.load('netwn_init_trnorm0.1+zero_seed=4.mdl').state_dict())
       
#         net.to(device=device)
#         optimizer1 = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)  
#         optimizer3 = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
        
#         result = []
#         ei_net=[]
#         term1_net=[]
#         term2_net=[]
#         loss_net=[]
#         MAEs_net=[]
#         data_models = []
#         spring = Simple_Spring_Model(device=device)
#         cpt('out')
#         for epoch in range(T):
            
#             start = np.random.randint(ss.size()[0]-batch_size)
#             end = start+batch_size
#             s,sp,l,lp, w = ss[start:end], sps[start:end], ls[start:end], lps[start:end], weights[start:end]
#             s_hist, z_hist= net.multi_step_forward(s, steps2)
#             rs_hist, rsn_hist = spring.multi_steps_sir(l, interval*steps2 , sigma,rou=rou,dt=dt,interval=interval) #sir
#             #predicts1, latent1, latentp1,z1 = net.forward_train(s,sp)
#             mae1 =  (MAE_raw(rsn_hist, s_hist).mean(axis=1) * w_steps2).mean()   #Forward error
#             loss = mae1 
#             optimizer1.zero_grad()
#             loss.backward()
#             optimizer1.step()
#             #Parallel training of inverse dynamics in the first stage.
#             for p in net.flow.parameters():
#                 p.requires_grad = False
                
#             #rs_hist, rsn_hist = spring.multi_steps_sir(l, interval*steps2 , sigma,rou=rou,dt=dt,interval=interval) #sir
#             latent1=net.encoding(rsn_hist[:-batch_size,:])
#             latent1=torch.cat((net.encoding(s),latent1),0)
#             batches = torch.split(latent1, batch_size)
#             latent1= torch.cat(batches[::-1])
#             predicts0, latentp0 = net.multi_back_forward(rsn_hist[-batch_size:,:],steps2) 
#             w=w.repeat(steps2)
#             mae2 = (MAE_raw(latent1, latentp0).mean(axis=1) * w * w_steps2).mean()  #
#             optimizer3.zero_grad()
#             mae2.backward()
#             optimizer3.step()
            
#             for p in net.flow.parameters():
#                 p.requires_grad = True
                
        
           
            
#             if epoch % 500 == 0:
#                 #fire=0.99**(epoch/20)
#                 #w_steps2=torch.exp(-torch.tensor(range(steps2))*fire)
#                 #w_steps2=w_steps2.unsqueeze(1).expand(-1, batch_size).reshape(-1)
#                 cpt('o_0')
#                 print(experiment, '/', experiments, 'Epoch:', epoch)
            
    
#                 ei0, sigmas0,weightsnet=test_model_causal_multi_sis(spring_data,10000,MAE_raw,net,sigma,scale, L=L,num_samples = 1000, bigL = L)
            
#                 print('dEI: BiNet= %.4f' % ei0[0])  
#                 print('term1: BiNet= %.4f ' % ei0[3])
#                 print('term2: BiNet= %.4f '% ei0[4])
#                 print(120*'-')
#                 ei_net.append(ei0[0])
#                 term1_net.append(ei0[3].item())
#                 term2_net.append(ei0[4].item())
#                 loss_net.append(mae1.item())
    
                
#                 result.append([ei0[0],ei0[3].item(), ei0[4].item(),loss.item()]) #
#                 cpt('o_1')
                
#         for epoch in range(T,T2):
#             start = np.random.randint(ss.size()[0]-batch_size)
#             end = start+batch_size
#             s,sp,l,lp, w = ss[start:end], sps[start:end], ls[start:end], lps[start:end], weights[start:end]
#             #w=w.repeat(steps2)
            
#             s_hist, z_hist = net.multi_step_forward(s, steps2)
#             rs_hist, rsn_hist = spring.multi_steps_sir(l, interval*steps2 , sigma,rou=rou,dt=dt,interval=interval) 
#             mae1 =  (MAE_raw(rsn_hist, s_hist).mean(axis=1)* w * w_steps2).mean()   #Forward error
            
#             latent1=net.encoding(rsn_hist[:-batch_size,:])
#             latent1=torch.cat((net.encoding(s),latent1),0)
#             batches = torch.split(latent1, batch_size)
#             latent1= torch.cat(batches[::-1])
#             predicts0, latentp0 = net.multi_back_forward(rsn_hist[-batch_size:,:],steps2) 
#             mae2 = (MAE_raw(latent1, latentp0).mean(axis=1) * w * w_steps2).mean()  #
#             loss = mae1+mae2_w*mae2 #Inverse dynamics learner
#             optimizer1.zero_grad()
#             loss.backward()
#             optimizer1.step()
    
#             #Inverse probability weighting
#             if epoch > 0 and epoch % encoder_interval == 0:
#                 cpt('w_0')
#                 #Resampling of training data is performed here, which varies according to the changes in the encoder.
#                 net_temp = Renorm_Dynamic(sym_size = sz, latent_size = scale, effect_size = sz, 
#                              hidden_units = hidden_units, normalized_state=False, device = device)
#                 net_temp.load_state_dict(net.state_dict())
#                 net_temp.to(device=device)
#                 encodings = net_temp.encoding(ss)  #Encoding of data collected by Spring.
#                 cpt('w_1')
#                 log_density, k_model_n = kde_density(encodings)  #Probability distribution of the encoded raw data.
#                 cpt('w_2')
#                 log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  
#                 logp = log_rho - log_density 
#                 weights = to_weights(logp, temperature) * sample_num
#                 if use_cuda:
#                     weights = weights.cuda(device=device)
#                 weights=torch.where(weights<10,weights,10.)
#                 cpt('w_3')
           
    
#             if epoch % 500 == 0:
#                 cpt('o_0')
#                 print(experiment, '/', experiments, 'Epoch:', epoch)
#                 #fire=0.99**(epoch/20)
#                 #w_steps2=torch.exp(-torch.tensor(range(steps2))*fire)
#                 #w_steps2=w_steps2.unsqueeze(1).expand(-1, batch_size).reshape(-1)
#                 ei0, sigmas0,weightsnet=test_model_causal_multi_sis(spring_data,10000,MAE_raw,net,sigma,scale, L=L,num_samples = 1000, bigL = L)
    
#                 print('dEI: BiNet= %.4f' % ei0[0])  
#                 print('term1: BiNet= %.4f ' % ei0[3])
#                 print('term2: BiNet= %.4f '% ei0[4])
#                 print(120*'-')
#                 result.append([ei0[0],ei0[3].item(), ei0[4].item(),mae1.item()])
#                 ei_net.append(ei0[0])
#                 term1_net.append(ei0[3].item())
#                 term2_net.append(ei0[4].item())
#                 loss_net.append(mae1.item())
    
#                 cpt('o_1')
             
#         results.append(result)  
#         torch.save(net.state_dict(), 'noise='+str(noise2)+' original.mdl')
#     #Save Result
#     import pickle
#     with open('noise fourth scale2.pkl', 'wb') as f:
#          pickle.dump(results, f)   

#     return