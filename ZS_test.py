import torch
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

N = 1000

def fine_tune_loss(B, v, w_opt, sigma):
    error_vector = torch.mm(v, B)-w_opt.T #(1,d)
    return torch.mm(torch.mm(error_vector,sigma), error_vector.T).squeeze()

args = argparse.Namespace()

args.sigmau = 1
args.sigmav = 0.1
args.sigma_Vaux = 2
args.v_aux_iterations = 7
args.d = 300
args.r = 10
args.w_opt_sigma = 1
args.sigma_v0 = 0.25

def main():
    for sigmav in tqdm([0,0.1,0.5,1,2,5,10,50]):
        ID_avg = 0
        OOD_avg = 0
        for i in range(N):
            U = torch.normal(0,args.sigmau,(args.d, args.r))
            V = U + torch.normal(0, sigmav, (args.d, args.r))
            
            U,_ = torch.linalg.qr(U)
            

            V_aux = torch.tensor([])
            for i in range(args.v_aux_iterations):
                V_aux = torch.cat((V_aux, U + torch.normal(0, args.sigma_Vaux, U.size())), 1)
                V_aux = torch.cat((V_aux, V + torch.normal(0, args.sigma_Vaux, V.size())), 1)
            
            V,_ = torch.linalg.qr(V)
            V_aux,_ = torch.linalg.qr(V_aux)
            
            sigmaU = torch.mm(U,U.T)
            sigmaV = torch.mm(V,V.T)
            sigmaV_aux = torch.mm(V_aux,V_aux.T)

            w_opt = torch.normal(0, args.w_opt_sigma, (args.d,1)) #(d,1)

            B0 = V_aux.T #(k,d)
            v0 = torch.mm(torch.linalg.pinv(B0).T, w_opt+torch.normal(0, args.sigma_v0, w_opt.size())) #(k,1)

            ID = fine_tune_loss(B0, v0.T, w_opt, sigmaU).item()
            OOD = fine_tune_loss(B0, v0.T, w_opt, sigmaV).item()

            ID_avg += ID
            OOD_avg += OOD
                
        ID_avg /= N
        OOD_avg /= N
        
        plt.scatter(ID_avg, OOD_avg, label=sigmav)
        
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.xlabel('ID loss')       
    plt.ylabel('OOD loss')
    plt.title(f'ZS test for sigmva')
    plt.grid(visible=True, which='both')
    figure = plt.gcf()
    figure.set_size_inches(18, 12)
    plt.savefig(f'ZS_Upre.jpg')
    plt.clf()
    
if __name__ == '__main__':
    main()
