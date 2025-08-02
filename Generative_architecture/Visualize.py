import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def plot_to_DBG_for_easy_inspection(exp, method):    
    source = os.path.join(dir,exp)
    os.chdir(source)
    if len(method) == 2:
        LD = method[1]
        method = method[0]
    else: 
        LD = 0
    n =1
    if LD:
        data = torch.load(os.path.join(LD,"data_"+method+".pt"), weights_only=False, map_location=torch.device('cpu'))
    else:
        data = torch.load(os.path.join("ETE","data_"+method+".pt"), weights_only=False, map_location=torch.device('cpu'))
    
    data_to_plot = torch.stack([data[i] for i in range(n*n)]).view(45*n,32*n).detach().numpy()
    plt.imshow(data_to_plot, cmap="gray")

    dst = os.path.join(savedir,exp)
    if LD:
        dst = os.path.join(dst,LD)
        if not os.path.isdir(dst) : os.mkdir(dst)
    save_path = os.path.join(dst, method)
    plt.savefig(save_path+".png")
    plt.close()

def exp_plot_to_dbg(exp):
    source = os.path.join(dir,exp)
    os.chdir(source)
    dst = os.path.join(savedir,exp)
    if not os.path.isdir(dst):os.mkdir(dst)
    
    for LD in ["10","20","30","40","50"]:
        for method in ["NF","wgan","rae"]:
            plot_to_DBG_for_easy_inspection(exp,(method, LD))
    
    for method in ["ete_NF", "ete_Wgan"]:
        plot_to_DBG_for_easy_inspection(exp,method)

dir = "/home/jcolombini/Purpose/Labeler/Results/Generative_results"
savedir = "/home/jcolombini/Purpose/Labeler/DBG/IMG/INSPECTION"
exp_plot_to_dbg("2025-06-22-10-36")
exit()
for exp in os.listdir(dir):
    exp_plot_to_dbg(exp)