import itertools
import random
import csv
import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

MARKER_CYCLE = ['o', '+', '.', 'v', '*','X','s','p','P','1','2','H','D']

def create_graph_weight_vs_tuning(csvs, csvs_int, out_dir):
    marker_cycle = MARKER_CYCLE.copy()
    res_max = {}
    for csv_file in csvs:
        random.shuffle(marker_cycle)
        marker = itertools.cycle(marker_cycle)
        splitext = os.path.splitext(csv_file)
        csv_int = f"{splitext[0]}_int{splitext[1]}"
        assert csv_int in csvs_int
        
        alphas, ID, OOD = [], [], []
        with open(csv_int, 'r', newline='') as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for l in reader:
                #alphas.append(float(l['alpha']))
                ID.append(float(l['ID loss']))
                OOD.append(float(l['OOD loss']))
                
        label = os.path.splitext(os.path.basename(csv_file))[0]
            
        plt.plot(ID, OOD, label=label+" weight interpolation", marker=next(marker))

        ID, OOD = [], []
        with open(csv_file, 'r', newline='') as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for l in reader:
                ID.append(float(l['ID loss']))
                OOD.append(float(l['OOD loss']))
                
        plt.plot(ID, OOD, label=f'{label} training', marker=next(marker))
        
        plt.scatter(ID[0], OOD[0], label='ZS', c='g', s=225)
        
        
        plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.xlabel('ID loss')       
        plt.ylabel('OOD loss')
        
        if True:
            plt.xscale('log')
        else:
            plt.xlim(left=0)
            
        plt.title(f'{label}: weight interpolation vs. early stopping')
        plt.grid(visible=True, which='both')
        figure = plt.gcf()
        figure.set_size_inches(18, 12)
        plt.savefig(f'{out_dir}/{label}_weight_training.jpg')
        plt.clf()
    
def create_graphs_2d(csvs, out_dir, title, fine_tune_compare = False, prints=False, text_lambdas=False, first_label=False, final_label=False, include_LP=False):
    if prints:
        print(f'creating 2d graph {title}')
        
    marker_cycle = MARKER_CYCLE.copy()
    random.shuffle(marker_cycle)
    marker = itertools.cycle(marker_cycle)
    res_max = {}
    curve = defaultdict(list)
    for csv_file in csvs:
        ID, OOD = [], []
        include_LP_flag = True
        with open(csv_file, 'r', newline='') as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for l in reader:
                #alphas.append(float(l['alpha']))
                if include_LP and include_LP_flag:
                    include_LP_flag = False
                    LP_id = float(l['ID loss'])
                    LP_ood = float(l['OOD loss'])
                else:
                    id, ood = float(l['ID loss']), float(l['OOD loss'])
                    if id > 0 and ood > 0:
                        ID.append(id)
                        OOD.append(ood)
        
        if isinstance(csvs, dict):
            label = csvs[csv_file]
        else:
            label = os.path.splitext(os.path.basename(csv_file))[0]
            if label.endswith("_int"):
                label = label[:-4]
        if fine_tune_compare and\
            not any(os.path.basename(csv_file).startswith(s) for s in ["fine_tune", "interpolate"])\
            and (len(label)<=2 or label[2] != 'all'):
            curve[label[0]].append((label[1],ID[-1],OOD[-1]))
                       
            if prints:
                print(label[1],ID[-1],OOD[-1])
                
        else:
            if os.path.basename(csv_file).startswith("fine_tune"):
                #label = label[0]
                fine_tune_res = (ID, OOD)
                
                ms = 15
                if first_label:
                    plt.plot(ID[0], OOD[0], label=f'{first_label} {label}', marker=next(marker), ms=ms)
                    print(ID[0], OOD[0])
                    
                if final_label:
                    plt.plot(ID[-1], OOD[-1], label=f'{final_label} {label}', marker=next(marker), ms=ms)
                    print(ID[-1], OOD[-1])
                    
                if include_LP:
                    plt.plot(LP_id, LP_ood, label=f'LP {label}', marker=next(marker), ms=ms)
                
            plt.plot(ID, OOD, label=label, marker=",")
                    
        res_max[label] = []
        ind = np.argmin(ID)
        res_max[label].append((ID[ind], OOD[ind]))
        ind = np.argmin(OOD)
        res_max[label].append((ID[ind], OOD[ind]))
           
    if fine_tune_compare:
        ID_total = np.array([])
        OOD_total = np.array([])
        for label in curve:
            key = curve[label][0][0]
            if isinstance(key, float) or isinstance(key,int):
                key0 = 0
                keyinf = np.inf
            else:
                key0 = tuple(0 for i in key)
                keyinf = tuple(np.inf for i in key)
                
            #curve[label].append((keyinf, fine_tune_res[0][0], fine_tune_res[1][0]))
            #curve[label].append((key0, fine_tune_res[0][-1], fine_tune_res[1][-1]))
            
            curve[label].sort(key = lambda x: x[0])
            
            p = plt.plot([i[1] for i in curve[label]], [i[2] for i in curve[label]], label=label, marker=next(marker))
            color = p[0].get_color()
            
            if text_lambdas:
                [plt.text(i[1],i[2],i[0], color=color) for i in curve[label]]
                
            '''if first_label:
                plt.scatter([i[1][0] for i in curve[label]], [i[2][0] for i in curve[label]], label=f'{first_label} {label}', marker=next(marker))

            if final_label:
                plt.scatter([i[1][-1] for i in curve[label]], [i[2][-1] for i in curve[label]], label=f'{final_label} {label}', marker=next(marker))
            '''
    #plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.xlabel('ID loss')       
    plt.ylabel('OOD loss')
    if False:
        plt.xscale('log')
    else:
        plt.xlim(left=0)
    plt.title(title)
    plt.grid(visible=True, which='both')
    figure = plt.gcf()
    figure.set_size_inches(18, 12)
    plt.savefig(f'{out_dir}/{title.replace(" ","_")}.jpg')
    plt.clf()
    
    return
    
    random.shuffle(marker_cycle)
    marker = itertools.cycle(marker_cycle) 
    
    for label in res_max:
        plt.plot([i[0] for i in res_max[label]], [i[1] for i in res_max[label]], label=label, marker=next(marker), linestyle="None")
            
    plt.legend(loc='best')
    plt.xlabel('ID loss')       
    plt.ylabel('OOD loss')
    plt.title('weight interpolation - best')
    plt.grid(visible=True, which='both')
    figure = plt.gcf()
    figure.set_size_inches(18, 12)
    plt.savefig(f'{out_dir}/{title.replace(" ","_")}_best.jpg')
    plt.clf()
    
def create_graphs(csvs, out_dir, values, ylabel, title, include_first):
    first = True
    for csv_file in csvs:
        values_list = defaultdict(list)
        X, Y = [], []
        with open(csv_file, 'r', newline='') as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for l in reader:
                for value in values:
                    values_list[value].append(float(l[value]))
                
        label = os.path.splitext(os.path.basename(csv_file))[0]
        for value in values_list:
            plt.plot(range(len(values_list[value])), values_list[value], '--', label=f'{label} {value}')
        
        if include_first:
            if first:
                first = False
                first_label = label
                first_values_list = values_list.copy()
                
            else:
                for value in first_values_list:
                    plt.plot(range(len(first_values_list[value])), first_values_list[value], '--', label=f'{first_label} {value}')
        
        plt.legend(loc='best')
        plt.xlabel('epochs')       
        plt.ylabel(ylabel)
        plt.title(f'{label} {title}')
        plt.grid(visible=True, which='both')
        figure = plt.gcf()
        figure.set_size_inches(18, 12)
        plt.savefig(f'{out_dir}/{label}_{title}.jpg')
        plt.clf()
        
    for csv_file in csvs:
        values_list = defaultdict(list)
        with open(csv_file, 'r', newline='') as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for l in reader:
                for value in values:
                    values_list[value].append(float(l[value]))
                
        label = os.path.splitext(os.path.basename(csv_file))[0]
        for value in values_list:
            plt.plot(range(len(values_list[value])), values_list[value], '--', label=f'{label} {value}')
        
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc='best')
    plt.xlabel('epochs')       
    plt.ylabel(ylabel)
    plt.title(f'{title} {ylabel} (epochs)')
    plt.grid(visible=True, which='both')
    figure = plt.gcf()
    figure.set_size_inches(18, 12)
    plt.savefig(f'{out_dir}/res_{title}.jpg')
    plt.clf()