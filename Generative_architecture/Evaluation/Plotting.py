import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Evaluation.Distances import Distances
import json

plt.rcParams['font.family'] = 'serif'

def dst_to_sim(x, sigma):
    return np.e**(-np.clip(x,-1,10000)/sigma)

def load_data(data,function_names):
    for key in function_names:
        if key not in data:
            raise Exception("Function name not found in data: " + key)
        if key == "MMD" or key == "KID" or key == "FID":
            data[key] = [np.mean([dst_to_sim(a,100) for a in data[key]]), np.sqrt(np.var([dst_to_sim(a,100) for a in data[key]]))]
        elif key == "Precision" or key == "Recall" or key == "Density" or key == "Coverage" or key == "F1_PR":
            data[key] = [np.mean([a for a in data[key]]), np.sqrt(np.var([a for a in data[key]]))]
        else:
            data[key] = [np.mean([dst_to_sim(a,1) for a in data[key]]), np.sqrt(np.var([dst_to_sim(a,1) for a in data[key]]))]
    return data

def compared_plot(data_a,data_b,data_c, method, Latent_space, function_name,size):
    plt.hist(np.clip(data_a, -1000, 1000), bins=20, density=True, alpha=0.4, label='TT')
    plt.hist(np.clip(data_b, -1000, 1000), bins=20, density=True, alpha=0.4, label='TF')
    plt.hist(np.clip(data_c, -1000, 1000), bins=20, density=True, alpha=0.4, label='FF')
    plt.legend()
    plt.savefig( str(Latent_space)+"/distributions/"+method+function_name+"size"+str(size)+".png")
    plt.close()

def select_runs(name):
    if name == "NHTS":
        with open("config.py","r") as file:
            for line in file.readlines():
                if line [:len("DATASET")] == "DATASET":
                    if line[len("DATASET")+4:len("DATASET")+7] == "TMD":
                        return 1
        return 0    
    elif name == "TMD":
        with open("config.py","r") as file:
            for line in file.readlines():
                if line [:len("DATASET")] == "DATASET":
                    if line[len("DATASET")+4:len("DATASET")+8] == "NHTS":
                        return 1
        return 0   
    else:
        raise Exception("Not Implemented Dataset")  


def read_config(name):
    with open("config.py","r") as file:
        for line in file.readlines():
            if line[:len("EPOCHS")]=="EPOCHS":
                i = line.index("(")+1
                j = line.index(")")
                EPOCHS = line[i:j]
            if line[:len("LEARNING_RATE")]=="LEARNING_RATE":
                i = line.index("(")
                j = line.index(")")+1
                LEARNING_RATE = line[i:j]
    return EPOCHS, LEARNING_RATE

def plot_cross_epoch(a_x,a_y, measure_name, method):
    fig, ax = plt.subplots(3,2)
    for j in range(5):
        ax[j%3][j//3].scatter(a_x[j::5], a_y[j::5], c = "black")
        ax[j%3][j//3].set_title(measure_name[j])
    fig.savefig("/home/jcolombini/Purpose/Labeler/DBG/IMG/LR_EPOCH/"+method+".png")
    plt.close()


def epoch_LR_exam(resdir, dataset):
    methods = ["rae" , "wgan", "NF"]
    for j, method in enumerate(methods):
        totalx = []
        totaly = []
        runningruns = []        
        for name in os.listdir(resdir):
            if name in runningruns:
                continue
            os.chdir(os.path.join(resdir, name))
            if select_runs(dataset):
                continue
            EPOCHS, LEARNING_RATE = read_config(name)
            EPOCHS = EPOCHS.split(',')
            EPOCHS = (int(EPOCHS[0]), int(EPOCHS[0]),int(EPOCHS[0]))

            Matrix_functions = Distances.Graph_functions_tasks
            Graph_functions = Distances.Matrix_functions_tasks
            function_names = [f.__name__ for f in Matrix_functions + Graph_functions]

            L_D = []
            for Latent_dimension in [10,20,30,40,50]:      ######
                with open(str(Latent_dimension)+'/results_'+method+'.json', 'r') as file:
                    data = json.load(file)   
                                    ###############
                data = [ 1-np.mean(data[key]) for key in function_names]
                L_D.append(data)
            for e in np.max(np.array(L_D), axis = 0):
                totaly.append(e)
            for _ in range(len(np.max(np.array(L_D), axis = 0))):
                totalx.append(EPOCHS[j])
        plot_cross_epoch(totalx,totaly,["Centr","Dens","Deg","Frob","JSD"],method)



def plot(name):
    os.chdir(name)
    Matrix_functions = Distances.Graph_functions_tasks
    Graph_functions = Distances.Matrix_functions_tasks
    function_names = [f.__name__ for f in Matrix_functions + Graph_functions]
    function_names += ["MMD","KID","FID","Precision","Recall","Density","Coverage","F1_PR"]
    methods = ["NF", "wgan", "rae"]
    total = []
    for method in methods:
        L_D = []
        for Latent_dimension in [10,20,30,40,50]:      ######
            if not os.path.isfile(str(Latent_dimension)+'/was_results_'+method+'.json'):
                return None
            with open(str(Latent_dimension)+'/was_results_'+method+'.json', 'r') as file:
                data = json.load(file)  
            data = load_data(data, function_names)
            L_D.append(data)
        total.append(L_D) 
        
    for method in ['ete_Wgan','ete_NF']:
        L_D=[]
        with open('ETE/was_results_'+method+'.json', 'r') as file:
            data = json.load(file)   
                            ###############
            data = load_data(data, function_names)
        L_D.append(data)
        total.append(L_D) 

    # total = [
    #    NF   [10{f1:[m,v],f2:[],f3:[],f4:[],f5:[]},20{},30{-},40{},50{}],
    #    wgan [10{},20{},30{},40{},50{}],
    #    rae  [10{},20{},30{},40{},50{}]
    # ]

    for function_name in function_names:
        NF =    [total[0][a][function_name] for a in range(len(total[0]))]
        wgan =  [total[1][a][function_name] for a in range(len(total[1]))]
        Rae =   [total[2][a][function_name] for a in range(len(total[2]))]
        Wgan_ete = [total[3][a][function_name] for a in range(len(total[3]))]
        NF_ete = [total[4][a][function_name] for a in range(len(total[4]))]

        all = [NF,wgan,Rae]
        ete = [Wgan_ete,NF_ete]

        x = np.arange(7)  # the label locations
        x_ax = np.arange(5)
        fig, ax = plt.subplots()
        colo = ['red','black','blue']
        t=0
        offset = [-0.04, 0 , +0.04]
        for t, serie in enumerate(all):  
            media =[a[0] for a in serie]
            errore = [a[1] for a in serie]  
            ax.errorbar(x_ax + offset[t], media, yerr=errore, ls = '-.', capsize=3, color = colo[t],markersize=15,marker='.' )

        for t, serie in enumerate(ete):
            media =[a[0] for a in serie]
            errore = [a[1] for a in serie]  
            ax.errorbar(t+len(x_ax), media, yerr=errore, ls = '-.', capsize=3, color = colo[t],markersize=15,marker='.' )
        
        ax.set_title(function_name.replace("_distance",""), fontweight='bold', fontsize="25")
        ax.set_ylabel('Similarity', fontweight='bold', fontsize = "20")
        ax.set_xlabel('Latent dimension', fontweight='bold', fontsize = "20")
        ax.set_yticks(ax.get_yticks()[1:-1],[str(a//(0.01)/100) for a in ax.get_yticks()[1:-1]], fontsize = "15", fontweight='bold')
        ax.set_xticks(x, [str(10),str(20),str(30),str(40),str(50),"Wgan", "NF"], fontsize = "15", fontweight='bold')
        ax.legend(["NF","Wgan","Rae"])
        s=2.65
        inces = (3*s,2*s)
        fig.set_size_inches(*inces) 
        plt.savefig(function_name+".pdf")
        plt.close()
    return None

if __name__ == "__main__":
    Res_dir = "/home/jcolombini/Purpose/Labeler/Results/Generative_results"
    experiments = os.listdir(Res_dir)
    for j, experiment in enumerate(experiments):
        dir = os.path.join(Res_dir, experiment)
        plot(dir)
    # pass
    # make_figures_trad("/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-17-EPOCHS(1000, 1500, 200)-LR(0.0002, 0.0003, 0.0001)-ALPHA0.0003NUM_CLSS13")