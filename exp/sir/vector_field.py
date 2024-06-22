import sys
sys.path.append("../..")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def mesh(density):
    p1 = np.array([0, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 0])

    #Generate an array of coordinates for grid points.
    x_min, x_max = -1, 6
    y_min, y_max = -1, 6
    x_range = np.arange(x_min, x_max, density)
    y_range = np.arange(y_min, y_max, density)
    X, Y = np.meshgrid(x_range, y_range)

    # Determine if each grid point is inside the right-angled triangle.
    in_triangle = np.zeros(X.shape, dtype=bool)
    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.array([X[i][j], Y[i][j]])
            if p[0] >= 0 and p[1] >= 0 and p[0]+p[1] <= 1 :
                in_triangle[i, j] = True

    # Grid points marked within a right-angled triangle.
    x_in = X[in_triangle]
    y_in = Y[in_triangle]

    return x_in, y_in

def transit(ss,sps,density):
    whole=torch.cat((ss,sps),0)
    xmax=whole[:,0].max()
    xmin=whole[:,0].min()
    ymax=whole[:,1].max()
    ymin=whole[:,1].min()
    numberx=(xmax-xmin)/density
    numberx=int(numberx.item())+1
    numbery=(ymax-ymin)/density
    numbery=int(numbery.item())+1
    number=numberx*numbery
    count = torch.zeros(numberx, numbery, numberx, numbery,device=device)
    h1=np.arange(xmin.cpu().detach().numpy(), xmax.cpu().detach().numpy(), density)
    h2=np.arange(ymin.cpu().detach().numpy(), ymax.cpu().detach().numpy(), density)
   
    cpt('start')
    for i in range(numberx):
        for j in range(numbery):
            ss0=(ss[:,0] >= xmin + i*density) & (ss[:,0] < xmin + (i+1)*density) & (ss[:,1] >= ymin + j*density) & (ss[:,1] < ymin + (j+1)*density)
            indices = torch.nonzero(ss0).squeeze()
            indices = torch.flatten(indices).cpu().numpy()
            for i2 in range(numberx):
                for j2 in range(numbery):
                    for k in indices:
                        if (sps[k,0] >= xmin + i2*density) & (sps[k,0] < xmin + (i2+1)*density) & (sps[k,1] >= ymin + j2*density) & (sps[k,1] < ymin + (j2+1)*density):
                            count[i,j,i2,j2]+=1


    count_mean=torch.zeros(numberx, numbery, 2)
    for i in range(numberx):
        for j in range(numbery):
            x_m=[]
            y_m=[]
            for i2 in range(numberx):
                for j2 in range(numbery):
                    x=count[i,j,i2,j2]*(i2-i)*density
                    y=count[i,j,i2,j2]*(j2-j)*density
                    x_m.append(x.item()) 
                    y_m.append(y.item())
            count_mean[i,j,0]=np.sum(x_m)/np.sum(count[i,j,:,:].cpu().detach().numpy())
            count_mean[i,j,1]=np.sum(y_m)/np.sum(count[i,j,:,:].cpu().detach().numpy())

    return count_mean,h1,h2

def vector_func(net,jac_bool=True,density=0.06,density2=0.02,density3=0.08):
    S, I = mesh(density)
    plt.figure()
    plt.scatter(S,I,color=colorlabel[2])
    dSdt,dIdt = SIR(S,I)
    S2=S+dSdt
    I2=I+dIdt
    dSdt=torch.tensor([dSdt],device=device,dtype=torch.float).t()
    dIdt=torch.tensor([dIdt],device=device,dtype=torch.float).t()
    ddSI=torch.cat((dSdt,dIdt,dSdt,dIdt),1)
    SI=torch.tensor([S,I],dtype=torch.float,device=device)
    SI=SI.t()
    spring = Simple_Spring_Model(device=device)
    if jac_bool:
        SISI=torch.cat((SI,SI),1)
        
        jac = torch.autograd.functional.jacobian(net.encoding, SISI)
        SI=spring.perturb(SI, sigma=sigma,rou=rou)
        SI=net.encoding(SI)
        for i in range(len(S)):
            dSI_i=ddSI[i,:].double()@jac[i,:,i,:].t().double()
            #jac_mean.append(dSI_i)
            dSdt[i]=dSI_i[0]
            dIdt[i]=dSI_i[1]
        SI2=SI+torch.cat((dSdt,dIdt),1)
        dd,h1,h2=transit(SI,SI2,density2)
        
    else:
        SI=spring.perturb(SI, sigma=sigma,rou=rou)
        SI=net.encoding(SI)
        plt.figure()
        plt.scatter(SI[:,0].cpu().detach().numpy(),SI[:,1].cpu().detach().numpy(),color=colorlabel[2])
        plt.xticks([i/100 for i in range(44, 58, 2)])
        # 设置y轴的刻度
        plt.yticks([i/100 for i in range(25, 42, 2)])
        # 绘制水平的栅格线
        plt.grid(True, which='major', axis='both', linestyle='-', linewidth=1, color=colorlabel[4])

        SI2=torch.tensor([S2,I2],dtype=torch.float,device=device)
        SI2=SI2.t()
        SI2=spring.perturb(SI2, sigma=sigma,rou=rou)
        SI2=net.encoding(SI2)
        dd,h1,h2=transit(SI,SI2,density2)
        
    
    h1_,h2_=np.meshgrid(h1,h2)
    h1_tensor=torch.tensor(h1_.T,device=device)
    h2_tensor=torch.tensor(h2_.T,device=device)
    hh=torch.cat((h1_tensor.reshape(-1).unsqueeze(1),h2_tensor.reshape(-1).unsqueeze(1)),1)
    S, I = mesh(density3)
    SI=torch.tensor([S,I],dtype=torch.float,device=device)
    SI=SI.t()
    spring = Simple_Spring_Model(device=device)
    SI=spring.perturb(SI, sigma=sigma,rou=rou)
    xx=net.encoding(SI)
    distances = torch.cdist(xx.float(), hh.float())
    nearest_indices = torch.argmin(distances, dim=1)
    nearest_indices=nearest_indices.cpu().detach().numpy()
    
    hh1=hh[:,0].cpu().detach().numpy()
    hh2=hh[:,1].cpu().detach().numpy()
    dd1=dd[:,:,0].cpu().detach().numpy().reshape(-1)
    dd2=dd[:,:,1].cpu().detach().numpy().reshape(-1)
    fig,ax = plt.subplots(figsize=(5,4),dpi=150)
    quiver = ax.quiver(hh1, hh2, dd1, dd2, width=0.005,color=colorlabel[1])
    return dd1[nearest_indices],dd2[nearest_indices]