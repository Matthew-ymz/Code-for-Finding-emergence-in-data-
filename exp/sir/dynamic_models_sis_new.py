import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

class Simple_Spring_Model():
    
    def __init__(self, device, R=3):
        self.device = device
        self.R = R
    
    def one_step(self, x, v, dt=0.01):
        r = (x*x + v*v) ** 0.5
        r -= (r - self.R) * dt / 12.56
        t = torch.atan2(v, x) + dt
        x_, v_ = r * torch.cos(t), r * torch.sin(t)
        return x_, v_
    
    def multi_steps_sir(self, s, steps, sigma,lam=1,miu=0.5,rou=-0.5,dt=0.01,interval=1): 
        #One sample point runs multiple time steps.
        batch_size = s.size()[0]
        s_hist = s
        sn_hist = self.perturb(s, 0.001,rou)
        for t in range(1,steps+1):
            s_next,i_next = self.SIR_step(s[:,0],s[:,1],lam,miu,dt=dt)
            #s_next = torch.Tensor(s_next).unsqueeze(0)
            s_next = torch.cat((s_next.unsqueeze(1),i_next.unsqueeze(1)),1)#s_next.to(self.device)
            if t%interval==0:
                s_hist = torch.cat((s_hist, s_next), 0)
                rand_next = self.perturb(s_next, sigma,rou)
                sn_hist = torch.cat((sn_hist, rand_next), 0)
            s = s_next
        return s_hist[batch_size:,:], sn_hist[batch_size:,:]
    
    def perturb(self, s, sigma,rou):
        prior = distributions.MultivariateNormal(torch.zeros(2,device=self.device), torch.tensor([[1,rou],[rou,1]],device=self.device)*sigma*sigma)
        rand1=prior.sample([s.size()[0]])
        rand2=prior.sample([s.size()[0]])
        s1 = s[:,0].unsqueeze(1) + rand1
        s2 = s[:,1].unsqueeze(1) + rand2
        s1_= torch.cat((s1[:,[0]], s2[:,[0]]), 1)
        s2_= torch.cat((s1[:,[1]], s2[:,[1]]), 1)
        sr = torch.cat((s1_, s2_), 1)
        return sr
    
   
    
    def generate_onestep(self,s,i,r,lam,miu, sigma,rou=-0.5):
        splus, iplus,rplus = self.SIR_step(s,i,r,lam,miu)
        s_p = self.perturb(torch.cat((s,i,r), 1), sigma,rou)
        splus_p = self.perturb(torch.cat((splus, iplus,rplus), 1), sigma,rou)
        return s_p, splus_p, torch.cat((s,i,r),1),torch.cat((splus,iplus,rplus),1)
    
    def SIR_step(self,s,i,lam,miu,noise2=0,dt=1):
        
        if noise2==0:
            s=s-dt* lam * s * i
            i=i+dt*(lam * s * i - miu * i)
        else:
            prior = distributions.MultivariateNormal(torch.zeros(2,device=self.device).float(), torch.tensor([[1,0],[0,1]],device=self.device).float()*noise2*noise2)
            rand=prior.sample([1])
 
            s=s-dt* lam * s * i+rand[0,0]
            i=i+dt*(lam * s * i - miu * i)+rand[0,1]
        return s,i
    
    def SIS_step(self,s,i,lam,miu,dt=0.01):
        s=s-dt* lam * s * i + dt * miu * i
        i=i+dt* lam * s * i - dt * miu * i
        return s,i

  
    def generate_multistep_sir(self,size_list,steps,lam=1,miu=0.5,sigma=0.03,rou=-0.5,dt=0.01,noise2=0,interval=1):
        #Be able to build a sample set with missing values.
        s=torch.tensor([0],device=self.device)
        i=torch.tensor([0],device=self.device)
        while s.size()[0]<size_list[0]:
            s_=torch.rand([1],device=self.device)
            i_=torch.rand([1],device=self.device) 
            if (s_+i_)<=1:
                s = torch.cat((s,s_),0)
                i = torch.cat((i,i_),0)
        frac=1/(len(size_list)-1)
        for j,size in enumerate(size_list[1:]):
            s1=torch.tensor([frac * j],device=self.device)
            i1=torch.tensor([0],device=self.device)
            while s1.size()[0]<size:
                s_=(torch.rand([1],device=self.device) * frac) + frac * j
                i_=torch.rand([1],device=self.device) 
                if (s_+i_)<=1:
                    s1 = torch.cat((s1,s_),0)
                    i1 = torch.cat((i1,i_),0)
            s = torch.cat((s,s1[1:]),0)
            i = torch.cat((i,i1[1:]),0)
        
        s=s[1:].unsqueeze(1)
        i=i[1:].unsqueeze(1)
        size=s.size()[0]
        idx=torch.randperm(s.nelement())
        s=s.view(-1)[idx].view(s.size())
        i=i.view(-1)[idx].view(i.size())
        history = torch.cat((s,i),1)
        observ_hist = self.perturb(history,sigma,rou)
        for k in range(1,steps+1):
            s,i=self.SIR_step(s,i,lam,miu,noise2=noise2,dt=dt)
            sir = torch.cat((s,i),1)
            sir_p = self.perturb(sir, sigma,rou)
            if k%interval==0:
                history = torch.cat((history, sir),0)
                observ_hist = torch.cat((observ_hist, sir_p),0)
        return observ_hist[:-size,:],observ_hist[size:,:], history[:-size,:],history[size:,:]

def calculate_multistep_predict(model,s,i,steps = 10,stochastic=False,sigma=0.03,rou=-0.5,dt=0.01):
    #Out-of-distribution generalization testing function
    spring = Simple_Spring_Model(device=device)

    if stochastic:
        z = torch.randn([1, 2], device=device)*L/2 
    else:
        z=torch.tensor([[s,i]],device=device) 
    s = spring.perturb(z, sigma,rou)

    s_hist, z_hist = model.multi_step_prediction(s, steps)
    if use_cuda:
        s_hist = s_hist.cpu()
        z_hist = z_hist.cpu()

    rs_hist, rsn_hist = spring.multi_steps_sir(z, steps, sigma,rou=rou,dt=dt) #sir
    if use_cuda:
        rs_hist = rs_hist.cpu()
        rsn_hist = rsn_hist.cpu()

    means=torch.mean(torch.abs(rsn_hist-s_hist[1:,:]))
    #cums=torch.cumsum(means, 0)[-1] / steps
    return means.item()
    