import torch
import logging
import sys
import argparse
import os
import shutil
import csv
import numpy as np
import itertools
from tqdm import tqdm
from scipy.linalg import null_space

import utils
import model
import create_graphs

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

# --- CONSTS ---
CSV_FIELDNAMES = ['loss', 'ID loss', 'OOD loss']
CSV_INT_FIELDNAMES = ['alpha', 'ID loss', 'OOD loss']
plots_dir = 'plots'
alpha_step = 0.01
alphas_interpolate = torch.arange(0,1+alpha_step,alpha_step)

#consts list to be saved as args.txt
consts_list = ["LOSSES","d", "r", "sigmau", "sigmav", "v_aux_iterations",\
    "sigma_VauxU", "sigma_VauxV", "w_opt_sigma", "sigma_v0", "L2_norm_loss_lambda_losses", "cos_reg_loss_lambda_losses", "lr", "epochs"]

# --- END CONSTS ---

# --- LOSSES ---
def BV_L2_norm_loss(B, v, w_opt, B0, v0, sigma_id, lambda_loss):
    return (fine_tune_loss(B, v, w_opt, sigma_id) + lambda_loss[0] *(((B-B0)**2).mean()) + lambda_loss[1] * (((v-v0)**2).mean()))

def L2_norm_loss(B, v, w_opt, B0, sigma_id, lambda_loss):
    return fine_tune_loss(B, v, w_opt, sigma_id) + lambda_loss*(((B-B0)**2).mean())

def cos_reg_loss(B, v, w_opt, B0, sigma_id, sigma_aux, lambda_loss):
    error_vector = B-B0 #(k,d)
    #return ((fine_tune_loss(B, v, w_opt, sigma_id) + lambda_loss * torch.trace(torch.mm(error_vector, error_vector.T))))
    #print(sigma_aux)
    return ((fine_tune_loss(B, v, w_opt, sigma_id) + \
        lambda_loss * torch.trace(torch.mm(torch.mm(error_vector, sigma_aux), error_vector.T))))

def vwrong_cos_reg_loss(B, v, w_opt, B0, sigma_id, sigma_v, lambda_loss):
    error_vector = B-B0 #(k,d)
    
    return fine_tune_loss(B, v, w_opt, sigma_id) + \
        lambda_loss * torch.trace(torch.mm(torch.mm(error_vector, sigma_v), error_vector.T))

def wrong_cos_reg_loss(B, v, w_opt, B0, sigma_id, lambda_loss):
    error_vector = B-B0 #(k,d)
    
    return fine_tune_loss(B, v, w_opt, sigma_id) + \
        lambda_loss * torch.trace(torch.mm(torch.mm(error_vector, sigma_id), error_vector.T))

def fine_tune_loss(B, v, w_opt, sigma):
    error_vector = torch.mm(v, B)-w_opt.T #(1,d)
    return torch.mm(torch.mm(error_vector,sigma), error_vector.T).squeeze()

# --- END LOSSES ---
def parse_argv():
    assert len(sys.argv) > 2
    
    ret_vars = ['out_folder','config','overwrite','preserve']
    overwrite = False
    preserve = []
    out_folder = sys.argv[2]
    
    config = sys.argv[1]
    
    if len(sys.argv) > 3 and sys.argv[3] == 'ow':
        overwrite = True
    
    locals_dict = locals()
    ret = argparse.Namespace()
    for key in locals_dict:
        setattr(ret, key, locals_dict[key])
        
    return ret
    
def parse_config(config_file):
    ret = argparse.Namespace()
    with open(config_file,'r') as config_handler:
        for l in config_handler.readlines():
            l = utils.parse_config_line(l)
            if not l:
                continue
                
            _locals = locals()
            exec(f'value={l[1]}', globals(), _locals)
            value = _locals['value']
                    
            setattr(ret, l[0], value)
            
    return ret
    
def parse_args():
    args = parse_argv()
    config_args = parse_config(args.config)
    args = argparse.Namespace(**vars(args), **vars(config_args)) #merge
    return args

def load_preserved_var(var, args):
    if not var in args.preserve and not args.load_all:
        return False
        
    filename = f'{args.parent_folder}/{var}.pt'
    if not os.path.exists(filename):
        return False

    #print(f'loaded {var}')
    return torch.load(filename)
    
def save_preserved_var(var, args, obj):
    if not var in args.preserve and not args.save_all:
        return
        
    filename = f'{args.parent_folder}/{var}.pt'
    if os.path.exists(filename):
        return
        
    #print(f'saved {var}')
    torch.save(obj, filename)

def main(args):   
    out_folder = args.out_folder
    out_folder = f'outs/{out_folder}'
    
    if args.preserve:
        args.parent_folder = os.path.dirname(out_folder)
    
    if args.overwrite and os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    elif os.path.exists(out_folder):
        pass
        #print("folder exists. Continue?")
        #if input().lower() != 'y':
        #    raise Exception(f'out folder {out_folder} exists')
    
    
    os.makedirs(out_folder, exist_ok=True)
    
    logging.basicConfig(filename=f'{out_folder}/log.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%d-%H:%M:%S',
                        level=logging.INFO)
                        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    #logging.getLogger('').addHandler(console)
    
    with open(f'{out_folder}/args.txt','w') as consts_handler:
        for const in consts_list:
            if const not in args:
                print(args)
                raise Exception(f'{const} not found in config file')
                
            consts_handler.write(f'{const}:{getattr(args, const)}\n')
        
    U = load_preserved_var("U", args)
    if U is False:
        U = torch.normal(0,args.sigmau,(args.d, args.r)).double()
        save_preserved_var("U", args, U)
        
    #print("U", U.mean(), U.max(), U.size())
        
    V = load_preserved_var("V", args)
    if V is False:
        V = U + torch.normal(0, args.sigmav, (args.d, args.r)).double()
        save_preserved_var("V", args, V)
       
    #create Vaux
    V_aux = load_preserved_var("V_aux", args)
    if V_aux is False:
        V_aux = torch.tensor([])
        for i in range(args.v_aux_iterations):
            if args.first_vaux_include and i == 0:
                if args.vaux_includeU:
                    V_aux = torch.cat((V_aux, U), 1)
                if args.vaux_includeV:
                    V_aux = torch.cat((V_aux, V), 1)
            else:
                if args.vaux_includeU:
                    V_aux = torch.cat((V_aux, U + torch.normal(0, args.sigma_VauxU, U.size())), 1)
                if args.vaux_includeV:
                    V_aux = torch.cat((V_aux, V + torch.normal(0, args.sigma_VauxV, V.size())), 1)

        save_preserved_var("V_aux", args, V_aux)
    
    if args.v_aux_rand_iterations:
        for i in range(args.v_aux_rand_iterations):
            V_aux = torch.cat((V_aux, torch.normal(0, args.sigmau, U.size())), 1)

    #orthonormalize
    V_aux,_ = torch.linalg.qr(V_aux)
    U,_ = torch.linalg.qr(U)
    V,_ = torch.linalg.qr(V)
    U,V,V_aux = U.double(), V.double(), V_aux.double()
    
    
    if args.v_aux_post_noise > 0:
        V_aux = torch.normal(0,args.v_aux_post_noise,V_aux.size())
        
    if args.u_post_noise > 0:
        U = torch.normal(0,args.u_post_noise,V_aux.size())
    
    if args.v_post_noise > 0:
        V = torch.normal(0,args.v_post_noise,V_aux.size())
        
    if args.v_aux_column_noise > 0:
        for c in range(V_aux.size(1)):
            V_aux[:,c] *= torch.normal(1,args.v_aux_column_noise,(1,1)).item()
    
    #create sigmas
    sigmaU = torch.mm(U,U.T).double()
    sigmaV = torch.mm(V,V.T).double()
    sigmaV_aux = torch.mm(V_aux,V_aux.T)
    sigmaV_aux = (1-args.sigmaV_aux_alpha)*sigmaV_aux + args.sigmaV_aux_alpha*torch.eye(args.d)
    print(sigmaV_aux)
    
    #create w_opt and v_0 = w_opt^T * B0
    
    w_opt = load_preserved_var("w_opt", args)
    if w_opt is False:
        w_opt = torch.normal(0, args.w_opt_sigma, (args.d,1)).double() #(d,1)
        save_preserved_var("w_opt", args, w_opt)
    
    #B0 = torch.tensor(null_space(U.T).T)
    '''rot = torch.zeros(args.d, args.d).double()
    rot[0][1] = -1
    rot[1][0] = 1
    B0 = torch.mm(rot, U).T
    B0 = B0#+torch.normal(0,1,(args.r,args.d)).double() #V_aux.T #(k,d)
    print(B0.size())
    #B0,_ = torch.linalg.qr(B0.T)
    #B0 = B0.T.double()
    print(B0.size())'''
    
    B0 = load_preserved_var("B0", args)
    if B0 is False:
        B0 = torch.DoubleTensor([])
        for i in range(args.k):
            B0 = torch.cat((B0,w_opt+torch.normal(0,args.B0_sigma,(w_opt.size()))),1)
            
        B0,_ = torch.linalg.qr(B0)
        B0 = B0.T.double()
        save_preserved_var("B0", args, B0)
    
    #B0 = torch.rand(args.k,args.d).double()
    
    v0 = load_preserved_var("v0", args)
    if v0 is False:
        #v0 = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(torch.linalg.pinv(B0).T,torch.linalg.inv(sigmaU)),torch.linalg.pinv(B0)), B0), sigmaU), w_opt+torch.normal(0, args.sigma_v0, w_opt.size())) #(k,1)
        #v0 = torch.mm(torch.mm(torch.mm(torch.linalg.inv(torch.mm(torch.mm(B0,sigmaU),B0.T)).T, B0), sigmaU), w_opt+torch.normal(0, args.sigma_v0, w_opt.size()))
        
        #v0 = torch.mm(torch.mm(torch.mm(torch.linalg.inv(torch.mm(torch.mm(torch.mm(B0,U),U.T),B0.T)),B0),U),w_opt)
        #v0 = torch.mm(torch.linalg.pinv(B0).T,w_opt)
        A = torch.mm(B0,U)
        B = torch.linalg.inv(torch.mm(torch.mm(torch.mm(U.T,B0.T),B0),U))
        #v0 = torch.rand(900,1).double()
        v0_LP = torch.mm(torch.mm(torch.mm(A, B), U.T), w_opt)
        v0 = v0_LP
        
        '''m = model.Model(B0.clone(),v0.clone())
        list(m.parameters())[0].requires_grad = False
        optimizer = torch.optim.SGD(m.parameters(), lr=args.lr)
        id_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaU).item()
        print(f'initial loss: {id_loss}')
        for epoch in tqdm(range(30000)):
            loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaU)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        id_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaU).item()
        print(f'LP loss: {id_loss} {(m.B.weight == B0).all()}')
        
        v0 = m.v.weight.T'''
        
        save_preserved_var("v0", args, v0)
        
    v0_LP = v0
    
    loss_fn_values_dict = {"w_opt": w_opt, "B0":B0, "v0":v0, "sigma_id":sigmaU, "sigma_aux":sigmaV_aux, "sigma_v":sigmaV}
    
    B_filename = f'{out_folder}/B0.pt'
    if not os.path.exists(B_filename):
        torch.save(B0, B_filename)
        torch.save(v0, f'{out_folder}/v0.pt')
        torch.save(w_opt, f'{out_folder}/w_opt.pt')
        torch.save(sigmaU, f'{out_folder}/sigmaU.pt')
        torch.save(sigmaV, f'{out_folder}/sigmaV.pt')
    
    #training
    csvs, csvs_int = {}, {}
    track_BV_files = []
    for loss_fn in tqdm(args.LOSSES, leave=False):
        loss_fn_name = loss_fn.__name__
        loss_fn_argnames = loss_fn.__code__.co_varnames[:loss_fn.__code__.co_argcount]
        investigate_finetune = getattr(args, f'{loss_fn_name}_investigate_finetune')
        
        if loss_fn == fine_tune_loss:
            iterate = [1]
            loss_fn_args = {"w_opt": w_opt, 'sigma':sigmaU}
        else:
            loss_fn_args = {}
            iterate = getattr(args, f"{loss_fn_name}_lambda_losses")
            for var in loss_fn_argnames:
                if var == "B" or var == "v" or var == "lambda_loss":
                    pass
                else:
                    if var not in loss_fn_values_dict:
                        raise Exception(f'var {var} not in {loss_fn_values_dict.keys()}')
                        
                    loss_fn_args[var] = loss_fn_values_dict[var]
            
        for lambda_loss in tqdm(iterate, leave=False):
            if not args.preserve:
                print(f"starting method {loss_fn_name} for lambda {lambda_loss}")
                
            csv_filename = f'{out_folder}/{loss_fn_name}{"" if loss_fn == fine_tune_loss else f"-lambda{lambda_loss}"}.csv'
            csv_file = open(csv_filename,'w')
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            
            if args.track_BV_dist:
                track_BV_dist_filename = f'{out_folder}/{loss_fn_name}{"" if loss_fn == fine_tune_loss else f"-lambda{lambda_loss}"}_track_BV.csv'
                track_BV_dist_csv_file = open(track_BV_dist_filename,'w')
                track_BV_dist_writer = csv.DictWriter(track_BV_dist_csv_file, fieldnames=['index','B mean','B max','v mean','v max'])
                track_BV_dist_writer.writeheader()
            
            #create model with initialized B0=V_aux and v_0
            m = model.Model(B0.clone(),v0.clone())
            m.double()
            optimizer = torch.optim.SGD(m.parameters(), lr=args.lr)
            
            if investigate_finetune:
                B_save = torch.tensor([])
                v_save = torch.tensor([])

            min_ood = np.inf
            min_epoch = 0
            
            if "lambda_loss" in loss_fn_argnames:
                loss_fn_args['lambda_loss'] = lambda_loss
            
            with torch.no_grad():
                id_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaU).item()
                ood_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaV).item()
                
            loss = loss_fn(m.B.weight, m.v.weight, **loss_fn_args)
            was_loss = loss.item()
            if args.include_LP and loss_fn == fine_tune_loss:
                with torch.no_grad():
                    LP_id_loss = fine_tune_loss(m.B.weight, v0_LP.T, w_opt, sigmaU).item()
                    LP_ood_loss = fine_tune_loss(m.B.weight, v0_LP.T, w_opt, sigmaV).item()
                    
                    print('LP', LP_id_loss, LP_ood_loss)
                    
                writer.writerow({'loss':loss.item(), 'ID loss':LP_id_loss, 'OOD loss': LP_ood_loss})
                
            writer.writerow({'loss':loss.item(), 'ID loss':id_loss, 'OOD loss': ood_loss})
            print(f'initial loss: {id_loss} {ood_loss}')
            mBM, mBA, mvM, mvA = [], [], [], []
            for epoch in tqdm(range(args.epochs), leave=False):
                if investigate_finetune:
                    B_save = torch.cat((B_save, m.B.weight.unsqueeze(0)), 0)
                    v_save = torch.cat((v_save, m.v.weight.unsqueeze(0)), 0)
                    
                if args.track_BV_dist:
                    #['index','B mean','B max','v mean','v max']
                    track_BV_dist_writer.writerow({
                        'index':epoch,
                        'B mean': torch.abs(m.B.weight - B0).mean().item(),
                        'B max': torch.abs(m.B.weight - B0).max().item(),
                        'v mean': torch.abs(m.v.weight - v0).mean().item(),
                        'v max': torch.abs(m.v.weight - v0).max().item()
                    })
                
                optimizer.zero_grad()
                    
                
                if epoch % 100 == 0:
                    logging.info(f'epoch {epoch}: {loss}')
                    
                loss.backward()
                optimizer.step()
                
                mBM.append(m.B.weight.grad.data.max().item())
                mBA.append(m.B.weight.grad.data.mean().item())
                mvM.append(m.v.weight.grad.data.max().item())
                mvA.append(m.v.weight.grad.data.mean().item())
                
                loss = loss_fn(m.B.weight, m.v.weight, **loss_fn_args)
                
                with torch.no_grad():
                    id_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaU).item()
                    ood_loss = fine_tune_loss(m.B.weight, m.v.weight, w_opt, sigmaV).item()
                    
                    check_id_losses = [
                        fine_tune_loss(m.B.weight, v0.T, w_opt, sigmaU).item(),
                        fine_tune_loss(B0, m.v.weight, w_opt, sigmaU).item()
                    ]
                    check_ood_losses = [
                        fine_tune_loss(m.B.weight, v0.T, w_opt, sigmaV).item(),
                        fine_tune_loss(B0, m.v.weight, w_opt, sigmaV).item()
                    ]
                
                if ood_loss < min_ood:
                    min_ood = ood_loss
                    min_epoch = epoch
                writer.writerow({'loss':loss.item(), 'ID loss':id_loss, 'OOD loss': ood_loss})
                
                if loss.item() > was_loss:
                    break
                was_loss = loss.item()
            
            plt.plot(range(len(mBM)), mBM, label='max B grad')
            plt.plot(range(len(mBA)), mBA, label='mean B grad')
            plt.plot(range(len(mvM)), mvM, label='max v grad')
            plt.plot(range(len(mvA)), mvA, label='mean v grad')
            plt.title('grad (epochs)')
            plt.yscale('log')
            plt.grid(visible=True, which='both')
            figure = plt.gcf()
            figure.set_size_inches(18, 12)
            plt.legend()
            plt.savefig(f'{out_folder}/grad.jpg')
            
            print(f'finished after {epoch+1}/{args.epochs} epochs')
            print(f'final loss: {id_loss} {ood_loss}')
            
            print(f'check losses: id: {check_id_losses} ood: {check_ood_losses}')
            
            print(f"\n\n\nmin ood {min_ood} epoch: {min_epoch}/{epoch}")
            csv_file.close()
            csvs[(loss_fn_name, lambda_loss)] = csv_filename
            
            csv_filename_int = f'{out_folder}/{loss_fn_name}{"" if loss_fn==fine_tune_loss else f"-lambda{lambda_loss}"}_int.csv'
            csv_file = open(csv_filename_int,'w')
            writer = csv.DictWriter(csv_file, fieldnames=CSV_INT_FIELDNAMES)
            #['alpha', 'ID loss', 'OOD loss']
            writer.writeheader()
            #wise-ft
            if investigate_finetune:
                alphas_res = {}
            
            for alpha in alphas_interpolate:
                res = {}
                
                B_alpha = m.B.weight*alpha+B0*(1-alpha)
                v_alpha = m.v.weight*alpha+v0.T*(1-alpha)
                res['alpha'] = alpha.item()
                res['ID loss'] = fine_tune_loss(B_alpha, v_alpha, w_opt, sigmaU).item()
                res['OOD loss'] = fine_tune_loss(B_alpha, v_alpha, w_opt, sigmaV).item()
                
                if investigate_finetune:
                    alphas_res[alpha] = res
                
                writer.writerow(res)
            csv_file.close()
            csvs_int[(loss_fn_name, lambda_loss)] = csv_filename_int
            
            B_filename = f'{out_folder}/{loss_fn_name}{"" if loss_fn == fine_tune_loss else f"-lambda{lambda_loss}"}_B.pt'
            torch.save(m.B.weight, B_filename)
            
            v_filename = f'{out_folder}/{loss_fn_name}{"" if loss_fn == fine_tune_loss else f"-lambda{lambda_loss}"}_v.pt'
            torch.save(m.v.weight, v_filename)
            
            if investigate_finetune:
                investigate_csv_filename = f'{out_folder}/{loss_fn_name}{"" if loss_fn == fine_tune_loss else f"-lambda{lambda_loss}"}_investigate_finetune.csv'
                csv_file = open(investigate_csv_filename,'w')
                writer = csv.DictWriter(csv_file, fieldnames=['index','(B1-B2)**2.mean','(v1-v2)**2.mean'])
                writer.writeheader()
                Bdiff_mean = 0
                Bdiff_max = 0
                vdiff_mean = 0
                vdiff_max = 0
                minv_max = 0
                minv_avg = 0
                with open(csv_filename,'r') as csv_handler:
                    reader = csv.DictReader(csv_handler)
                    for li, l in enumerate(reader):
                        minv = np.inf
                        id, ood = float(l['ID loss']), float(l['OOD loss'])
                        for alpha in alphas_res:
                            alpha_id, alpha_ood = alphas_res[alpha]['ID loss'], alphas_res[alpha]['OOD loss']
                            val = (id-alpha_id)**2+(ood-alpha_ood)**2
                            if val < minv:
                                minv = val
                                alpha_min = alpha
                        
                        if minv_max < minv:
                            minv_max = minv
                            
                        minv_avg += minv
                        B_alpha = B0*alpha_min+m.B.weight*(1-alpha_min)
                        v_alpha = v0.T*alpha_min+m.v.weight*(1-alpha_min)
                        B = B_save[li]
                        v = v_save[li]
                        Bdiff = ((B-B_alpha)**2).mean().item()
                        vdiff = ((v-v_alpha)**2).mean().item()
                        Bdiff_mean += Bdiff
                        vdiff_mean += vdiff
                        
                        if Bdiff_max < Bdiff:
                            Bdiff_max = Bdiff
                            
                        if vdiff_max < vdiff:
                            vdiff_max = vdiff
                        
                        writer.writerow({
                            'index': li,
                            '(B1-B2)**2.mean':Bdiff,
                            '(v1-v2)**2.mean':vdiff
                        })
                        
                csv_file.close()
                Bdiff_mean /= args.epochs
                vdiff_mean /= args.epochs
                minv_avg /= args.epochs
                
                print(f'Bdiff mean: {Bdiff_mean}')
                print(f'Bdiff max: {Bdiff_max}')
                print(f'vdiff mean: {vdiff_mean}')
                print(f'vdiff max: {vdiff_max}')
                print(f'val max: {minv_max}')
                print(f'val avg: {minv_avg}')
    
            if args.track_BV_dist:
                track_BV_dist_csv_file.close()
                track_BV_files.append(track_BV_dist_filename)
           
    plot_dir = f'{out_folder}/{plots_dir}'
    os.makedirs(plot_dir, exist_ok=True)
    
    csvs_values = list(csvs.values())
    csvs_int_values = list(csvs_int.values())
    
    create_graphs.create_graphs(csvs_values, plot_dir,
        values = ['ID loss','OOD loss'], ylabel='loss', title='training', include_first=True)   
    
    create_graphs.create_graphs_2d(csvs_int_values, plot_dir, "weight interpolation")
    create_graphs.create_graphs_2d(csvs_values, plot_dir, "tuning")
    
    if args.create_graph_weight_vs_tuning:
        create_graphs.create_graph_weight_vs_tuning(csvs_values, csvs_int_values, plot_dir)
        
    if args.track_BV_dist:
        create_graphs.create_graphs(track_BV_files, plot_dir,
            values = ['B mean','B max','v mean','v max'], ylabel='value', title='trackBV', include_first=False)
        
    return csvs, csvs_int
    
if __name__ == '__main__':
    args = parse_args()
    main(args)