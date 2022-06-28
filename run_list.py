import torch
import itertools
import os
import sys
import shutil
from tqdm import tqdm
import pickle
import csv
import numpy as np

import utils
import create_graphs
import train

orig_config = 'configs/config.ini'

#whether to keep objects that aren't affected by params in changes.
#Ex. if iterating over sigma_Vaux, then preserve U,V,w_opt,v_0 between runs and change V_aux, B0
preservce_objects = True

#mega_folder = "ZS"
#big_changes = {'sigmav':[0.1,0.5,1,2,5]}
'''big_changes = {'sigma_v0':[0,0.1,0.25,0.5]}
big_changes = {'sigmav':[2,5]}
big_changes = {'sigma_Vaux':[0.1,1,2,5]}'''
#big_changes = {'v_aux_rand_iterations':[0,1,2,5,10]}
changes = {'rand_vaux':range(10), 'd':1000, 'epochs':1000, 'v_aux_iterations':1, 'sigmav':1, 'sigma_Vaux':1, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]','v_aux_rand_iterations':0}
changes = {'rand_bla':range(10), 'd':1000, 'epochs':1000, 'v_aux_iterations':1, 'sigmav':1, 'sigma_Vaux':1, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]','v_aux_rand_iterations':0}
changes = {'v_aux_post_noise':[0,0.1,0.2,0.5,1,1.5], 'lr':1e-4, 'd':100, 'epochs': 1000, 'create_graph_weight_vs_tuning':True, 'fine_tune_loss_investigate_finetune':True, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]'}
#changes = {'u_post_noise':[0,1], 'd':1000, 'epochs': 1000, 'create_graph_weight_vs_tuning':True, 'fine_tune_loss_investigate_finetune':True, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]'}
changes = {'save_all':True, 'load_all':True, 'd':1000, 'v_aux_iterations':7, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,1000]'}
changes = {'save_all':True, 'load_all':True, 'd':1000, 'v_aux_iterations':7, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[4,5,8,10,100,200,500,1000,2000,3000,4000,5000,6000,7000,8000,9000]'}
#changes = {'save_all':True, 'load_all':True, 'd':1000, 'v_aux_iterations':7, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[]','weight_interpolations':"[('cos_reg_loss-lambda4','cos_reg_loss-lambda5000')]"}
#changes = {'plot_avg':True, 'rand_vaux':range(100), 'd':1000, 'v_aux_iterations':7, 'sigmav':1, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1, 'LOSSES':'[fine_tune_loss]','cos_reg_loss_lambda_losses':'[]'}
changes = {'save_all':True, 'load_all':True, 'd':1000, 'r':100, 'v_aux_iterations':7, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[fine_tune_loss]','BV_L2_norm_loss_lambda_losses':'[(10,10),(100,100),(1000,1000)]','cos_reg_loss_lambda_losses':'[4,5,8,10,100,200,500,1000,2000,3000,4000,5000,6000,7000,8000,9000]'}
changes = {'text_lambdas':True, 'save_all':True, 'load_all':True, 'd':300, 'r':10, 'v_aux_iterations':1, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 500, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','BV_L2_norm_loss_lambda_losses':'[]','cos_reg_loss_lambda_losses':'[15]'}
#changes = {'text_lambdas':True, 'save_all':True, 'load_all':True, 'd':300, 'r':10, 'v_aux_iterations':1, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-3, 'epochs': 100000, 'LOSSES':'[ BV_L2_norm_loss]','BV_L2_norm_loss_lambda_losses':'[(1000,1000)]','cos_reg_loss_lambda_losses':'[]'}
#changes = {'rand_vaux':range(10), 'd':3000, 'r':100, 'v_aux_iterations':7, 'sigmav':1, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[fine_tune_loss]'}
#changes = {'rand_vaux':range(10), 'd':300, 'r':10, 'v_aux_iterations':7, 'sigmav':1, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[fine_tune_loss]'}
#changes = {'sigma_v0':[0.1,0.25,0.5,1], 'd':1000, 'v_aux_iterations':7, 'sigmav':2, 'sigma_Vaux':2, 'lr':1e-4, 'epochs': 1000, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,100,500,1000]'}
#changes = {'v_aux_column_noise':[0,1,2,5,10,20,50,100], 'd':1000, 'sigmav':1, 'sigma_Vaux':2, 'lr':1e-6, 'epochs': 1000, 'create_graph_weight_vs_tuning':True, 'fine_tune_loss_investigate_finetune':True, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]'}
#changes = {'v_post_noise':[0.1,0.2,0.5,1], 'd':100, 'lr':1e-5, 'epochs': 1000, 'create_graph_weight_vs_tuning':True, 'fine_tune_loss_investigate_finetune':True, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]'}
#changes = {'v_post_noise':[0,0.1,0.2,0.5,1,1.5], 'd':1000, 'epochs': 1000, 'create_graph_weight_vs_tuning':True, 'fine_tune_loss_investigate_finetune':True, 'LOSSES':'[fine_tune_loss, cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,50,100,1000,5000,10000]'}

#changes = {'LOSSES':'[fine_tune_loss]', 'fine_tune_loss_investigate_finetune':True, 'create_graph_weight_vs_tuning':True}


changes = {'rand_vaux':range(5), 'save_all':True, 'load_all':True, 'd':1000, 'r':100, 'v_aux_iterations':2, 'sigmav':1, 'sigma_VauxU':2, 'sigma_VauxV':0.5, 'lr':1e-4, 'epochs': 1000, 'sigma_v0':0, 'LOSSES':'[fine_tune_loss]'}
changes = {'save_all':True, 'load_all':True, 'd':1000, 'r':100, 'v_aux_iterations':4, 'sigmav':0.5, 'sigma_VauxU':1, 'sigma_VauxV':1, 'lr':1e-4, 'epochs': 1000, 'sigma_v0':0, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[5,10,50,100,1000]'}
changes = {'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':1000, 'r':100, 'v_aux_iterations':4, 'v_aux_rand_iterations':0, 'sigmav':1, 'sigma_VauxU':1, 'sigma_VauxV':0.25, 'lr':1e-4, 'epochs': 1000, 'sigma_v0':0, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[10,100,1000]'}
changes = {'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':1000, 'r':100, 'v_aux_iterations':1, 'v_aux_rand_iterations':0, 'sigmav':0.5, 'sigma_VauxU':2, 'sigma_VauxV':2, 'lr':5e-5, 'epochs': 1000, 'sigma_v0':0, 'LOSSES':'[fine_tune_loss]','cos_reg_loss_lambda_losses':'[10,100,1000]'}
changes = {'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':10000, 'r':1000, 'v_aux_iterations':1, 'v_aux_rand_iterations':0, 'sigmav':0.5, 'sigma_VauxU':2, 'sigma_VauxV':1, 'lr':5e-5, 'epochs': 5000, 'sigma_v0':0, 'LOSSES':'[fine_tune_loss]','cos_reg_loss_lambda_losses':'[250,300,400,500]','BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
changes = {'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':1000, 'r':100, 'v_aux_iterations':1, 'v_aux_rand_iterations':0, 'sigmav':1, 'sigma_VauxU':3, 'sigma_VauxV':1, 'lr':1e-5, 'epochs': 100000, 'sigma_v0':0, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[0.1,1,1000]','BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
changes = {'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':1000, 'r':10, 'k':5, 'v_aux_iterations':1, 'v_aux_rand_iterations':0, 'sigmav':2, 'sigma_VauxU':3, 'sigma_VauxV':1, 'lr':1e-6, 'epochs': 10000, 'sigma_v0':0, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[1500,2000,3000,3500,4000,6000,8000,12500,15000,17500]','BV_L2_norm_loss_lambda_losses':'[(0,0)]'}

with open('a','r') as handler:
    sigmaV_aux_alpha = int(handler.read().strip())

print('sigmaV_aux_alpha', sigmaV_aux_alpha)
#0,10,100,1000,2500,5000,7500,10000,25000,50000,75000,100000
#500000, 1000000,2500000, 5000000, 7500000, 10000000
changes = {'sigma_VauxV':sigmaV_aux_alpha, 'sigmaV_aux_alpha':0, 'lr':1e-5, 'epochs': int(1e4), 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[0.25,0.5,0.75,1,2,5]', 'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'B0_sigma':0.5, 'd':1000, 'r':10, 'k':5, 'v_aux_iterations':1, 'vaux_includeU':False ,'v_aux_rand_iterations':0, 'sigmav':2, 'sigma_VauxU':0, 'sigma_v0':0, 'BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
#changes = {'sigma_VauxV':2, 'sigmaV_aux_alpha':0, 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[10000000]', 'lr':1e-7, 'epochs': int(1e6), 'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'B0_sigma':0.5, 'd':1000, 'r':10, 'k':5, 'v_aux_iterations':1, 'vaux_includeU':False ,'v_aux_rand_iterations':0, 'sigmav':2, 'sigma_VauxU':0, 'sigma_v0':0, 'BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
changes = {'v_aux_rand_iterations':sigmaV_aux_alpha, 'sigmaV_aux_alpha':0, 'sigma_VauxV':0, 'lr':1e-5, 'epochs': int(1e4), 'LOSSES':'[fine_tune_loss,cos_reg_loss]','cos_reg_loss_lambda_losses':'[0,0.25,0.5,0.75,1,1.25,1.5,10,100,1000,10000]', 'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'B0_sigma':0.5, 'd':1000, 'r':10, 'k':5, 'v_aux_iterations':1, 'vaux_includeU':False , 'sigmav':2, 'sigma_VauxU':0, 'sigma_v0':0, 'BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
changes = {'v_aux_rand_iterations':0, 'sigmaV_aux_alpha':0, 'lr':1e-6, 'epochs': int(1e5), 'LOSSES':'[cos_reg_loss]','cos_reg_loss_lambda_losses':'[20000,40000,50000,70000,90000]', 'include_LP':True,'save_all':True, 'load_all':True, 'text_lambdas':True, 'B0_sigma':0.5, 'd':1000, 'r':10, 'k':5, 'v_aux_iterations':1, 'vaux_includeU':True , 'sigmav':2, 'sigma_VauxU':0, 'sigma_VauxV':0, 'sigma_v0':0, 'BV_L2_norm_loss_lambda_losses':'[(0,0)]'}
#changes = {'save_all':True, 'load_all':True, 'text_lambdas':True, 'd':1000, 'r':100, 'v_aux_iterations':1, 'v_aux_rand_iterations':0, 'sigmav':1, 'sigma_VauxU':3, 'sigma_VauxV':1, 'lr':1e-5, 'epochs': 10000, 'sigma_v0':0, 'LOSSES':'[fine_tune_loss]','cos_reg_loss_lambda_losses':'[10,100,200,250,300,400,500]','BV_L2_norm_loss_lambda_losses':'[(0,0)]'}

objects_affected_params = {
    "U": ["d","r","sigmau"],
    "V": ["d","r","sigmau","sigmav"],
    "V_aux":["d","r","sigmau","sigmav","v_aux_iterations","sigma_VauxU","sigma_VauxV", "rand_vaux"],
    "B0":["w_opt","k"],
    "w_opt":["d","w_opt_sigma"],
    "v0":["d","r","sigmau","sigmav","v_aux_iterations","v_aux_rand_iterations","sigma_VauxU","sigmaVauxV", "w_opt_sigma","sigma_v0","rand_vaux"],
}

for c in changes:
    if isinstance(changes[c], range):
        changes[c] = list(changes[c])

def main_run(changes):
    if len(sys.argv)>1:
        parent_folder = sys.argv[1]
    else:
        parent_folder = "-".join([f'{key}={str(value).replace(" ","").replace("[","").replace("]","")}' for key,value in changes.items() if not any([key.endswith(i) for i in ["LOSSES", "_lambda_losses"]])])
        orig_parent_folder = parent_folder
        ext = 1
        while os.path.exists(f'outs/{parent_folder}'):
            ext += 1
            parent_folder = f'{orig_parent_folder}_{ext}'       
            
    mega = False
    if 'mega_folder' in globals() and 'big_changes' in globals():
        mega = True
        parent_folder = "-".join([f'{key}={str(value).replace(" ","").replace("[","").replace("]","")}' for key,value in changes.items() if not any([key.endswith(i) for i in ["LOSSES", "_lambda_losses"]]) and not (isinstance(value, list) and len(value)>1)])
        parent_folder = f'{mega_folder}/{parent_folder}'

    overwrite = (len(sys.argv)>2 and sys.argv[2] == 'ow')

    if overwrite:
        shutil.rmtree(f'outs/{parent_folder}', ignore_errors=True)
        
        
    with open(orig_config,'r') as orig_handler:
        orig = orig_handler.readlines()

    orig = [l.strip() for l in orig]
    orig_args = [utils.parse_config_line(l) for l in orig]
    orig_args = [l for l in orig_args if l]
    orig_args = [l[0] for l in orig_args]
    
    for c in changes:
        if not isinstance(changes[c],list):
                
            changes[c] = [changes[c]]
            
    print(f'changes: {changes}')
    
    preserve = []
    changes_iter_keys = changes.copy()
    changes_iter_keys = [key for key in changes_iter_keys if len(changes_iter_keys[key]) > 1]
    if preservce_objects:
        for obj in objects_affected_params:
            found = False
            for change_var in changes_iter_keys:
                if change_var in objects_affected_params[obj]:
                    found = True
                    continue
                    
            if found == False:
                preserve.append(obj)
                    
        print(f'preserving objects: {preserve}')
    
    csvs_dict = []
    changes_product = list(dict(zip(changes.keys(), values)) for values in itertools.product(*changes.values())) #list of dictionaries
    for args_values in tqdm(changes_product): #args_values = dictionary
        #print(f'Starting run for args: {args_values}')
        
        new = orig.copy()
        for arg in args_values: #arg = key
            if not arg in orig_args and not arg.startswith("rand"):
                raise Exception(f'arg {arg} not in {orig_config}')
                
            for li, l in enumerate(orig):
                if l.startswith(f'{arg} ') or l.startswith(f'{arg}='):
                    new[li] = f'{arg} = {args_values[arg]}'
                    continue
                    
        args_values = {key:value for key,value in args_values.items() if len(changes[key])>1}
        
        folder = "_".join([f'{key}-{str(args_values[key]).replace(" ","")}' for key in args_values])
            
        new_folder = f'{parent_folder}/{folder}'
        new_config_filename = f'configs/{new_folder}.ini'
        os.makedirs(os.path.dirname(new_config_filename), exist_ok=True)
        with open(new_config_filename,'w') as new_handler:
            for l in new:
                new_handler.write(f'{l}\n')
        
        args = train.parse_config(new_config_filename)
        args.out_folder = new_folder
        args.overwrite = overwrite
        args.preserve = preserve

        csvs, csvs_int = train.main(args)
        
        if csvs or csvs_int:
            csvs_dict.append((args_values, csvs, csvs_int))

    csvs_dict_file = f'outs/{parent_folder}/csvs_dict.pkl'
    
    if os.path.exists(csvs_dict_file):
        with open(csvs_dict_file,'rb') as cdh:
            new_dict = pickle.load(cdh)
            for new_item in new_dict:
                found = False
                for item in csvs_dict:
                    if item[0] == new_item[0]:
                        found = True
                        item[1].update(new_item[1])
                        item[2].update(new_item[2])
                        break
                        
                if not found:
                    csvs_dict.append(new_item)
    
    with open(csvs_dict_file,'wb') as cdh:
        pickle.dump(csvs_dict, cdh)

    with open(f'outs/{parent_folder}/changes.txt','w') as changesh:
        changesh.write(str(changes))   

    if len(args.weight_interpolations) > 0:
        csvs = {}
        w_opt = torch.load(f'outs/{parent_folder}/w_opt.pt')
        sigmaU = torch.load(f'outs/{parent_folder}/sigmaU.pt')
        sigmaV = torch.load(f'outs/{parent_folder}/sigmaV.pt')
        for pair in args.weight_interpolations:
            pair_csv_filename = f'outs/{parent_folder}/interpolate_{"_".join(pair)}.csv'
            pair_csv_handler = open(pair_csv_filename,'w')
            writer = csv.DictWriter(pair_csv_handler, fieldnames=['alpha','ID loss','OOD loss'])
            writer.writeheader()
            
            Bs = [f'outs/{parent_folder}/{p+"_B" if p != "ZS" else "B0"}.pt' for p in pair]
            vs = [f'outs/{parent_folder}/{p+"_v" if p != "ZS" else "v0"}.pt' for p in pair]
            Bs = [torch.load(B) for B in Bs]
            vs = [torch.load(v) for v in vs]
            for i, p in enumerate(pair):
                if p == 'ZS':
                    vs[i] = vs[i].T
                    
            alpha_step = 0.001
            alphas_interpolate = torch.arange(0,1+alpha_step,alpha_step)
            
            for alpha in alphas_interpolate:
                res = {}
                
                B_alpha = Bs[1]*alpha+(1-alpha)*Bs[0]
                v_alpha = vs[1]*alpha+(1-alpha)*vs[0]
                res['alpha'] = alpha.item()
                res['ID loss'] = train.fine_tune_loss(B_alpha, v_alpha, w_opt, sigmaU).item()
                res['OOD loss'] = train.fine_tune_loss(B_alpha, v_alpha, w_opt, sigmaV).item()
                
                writer.writerow(res)
                
            csvs[("_".join(pair),)] = pair_csv_filename
            pair_csv_handler.close()
            
        found = False
        for item in csvs_dict:
            if item[0] == {}:
                item[1].update(csvs)
                found = True
        
        if not found:
            csvs_dict.append(({}, csvs, {}))
    
    print(csvs_dict)

    #return
    if True:#len(changes_product) > 1:
        cross_dir = f'outs/{parent_folder}/cross'
        os.makedirs(cross_dir, exist_ok=True)
            
        if False:
            for change in changes:
                if len(changes[change]) > 1:
                    for change_value in changes[change]:
                        csvs_for_graph = []
                        for csvs_dict_iter in csvs_dict:
                            csv_args_values = csvs_dict_iter[0]
                            if csv_args_values[change] == change_value:
                                csvs_for_graph.append(csvs_dict_iter)
                                
                        print(csvs_for_graph, change, change_value)
                        if len(csvs_for_graph) > 1:
                            create_graphs.create_graphs_2d([i[1] for i in csvs_dict_iter], cross_dir, f'tuning {change}={change_value}')
                            create_graphs.create_graphs_2d([i[2] for i in csvs_dict_iter], cross_dir, f'weight interpolation {change}={change_value}')
        
        loss_lambda_keys = list(csvs_dict[0][1].keys())
        
        lambdas_folder = f'{cross_dir}/lambdas'
        os.makedirs(lambdas_folder, exist_ok=True)
        for key in loss_lambda_keys:
            csvs, csvs_int = {}, {}
            for csvs_dict_iter in csvs_dict:
                csvs.update({csvs_dict_iter[1][csv_key]:" ".join([f'{key}={value}' for key,value in csvs_dict_iter[0].items()]) for csv_key in csvs_dict_iter[1] if csv_key == key})
                csvs_int.update({csvs_dict_iter[2][csv_key]:" ".join([f'{key}={value}' for key,value in csvs_dict_iter[0].items()]) for csv_key in csvs_dict_iter[2] if csv_key == key})
                
            #print(f'{key}:{len(csvs)}')
            
            create_graphs.create_graphs_2d(csvs, lambdas_folder, f'tuning for {key}')
            create_graphs.create_graphs_2d(csvs_int, lambdas_folder, f'weight interpolation for {key}')
        
        args_folder = f'{cross_dir}/args'
        os.makedirs(args_folder, exist_ok=True)
        
        loss_lambda_keys = list(csvs_dict[0][1].keys())
        
        for csvs_dict_iter in csvs_dict:
            key = " ".join([f"{key}={value}" for key,value in csvs_dict_iter[0].items()])
            csvs = {value:key for key,value in csvs_dict_iter[1].items()}
            #csvs['outs/BV_cos_reg_early_3/cos_reg_loss-lambda100.csv'] = ('100_early',100,'all')
            #csvs['outs/BV_cos_reg_early_2/cos_reg_loss-lambda1000000.csv'] = ('1000000_early',1000000,'all')

            create_graphs.create_graphs_2d(csvs, args_folder, f'tuning for {key} {args.weight_interpolations if len(args.weight_interpolations)>0 else ""}', fine_tune_compare=True, prints=True, text_lambdas=args.text_lambdas, first_label='ZS', final_label='FT', include_LP='include_LP' in changes)
            #create_graphs.create_graphs_2d(csvs_int, args_folder, f'weight interpolation for {key}')
                
        
    print(f'Done. Saved to {parent_folder}')
    print('Run:')
    run = f'scp -r maorkehati@c-004.cs.tau.ac.il:/home/ycarmon/users/maorkehati/texp/outs/{parent_folder} texpN'
    print(run)
    return run
    
def main():
    changes_local = changes.copy()
    if 'big_changes' in globals() and big_changes and 'mega_folder' in globals():
        assert len(big_changes) == 1
        
        runs = []
        
        key = list(big_changes.keys())[0]
        for value in big_changes[key]:
            changes_local[key] = value
            runs.append(main_run(changes_local))
            
        print(f'scp -r maorkehati@c-004.cs.tau.ac.il:/home/ycarmon/users/maorkehati/texp/outs/{mega_folder} texpN')
        
    else:
        main_run(changes_local)
    
if __name__ == '__main__':
    main()
    open('blabla','w').write('bla')