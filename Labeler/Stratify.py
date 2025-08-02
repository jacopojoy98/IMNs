import os, sys
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Generative_architecture.Evaluation.DataManipulation import Number_of_nodes_of_graph_from_matrix
from Loader import Load
from Folds import group_kfold_with_distribution, extract_subdatasets

Generative_results_directory = "/home/jcolombini/Purpose/Labeler/Results/Generative_results"



def stratify(args):
    sgkf = StratifiedGroupKFold(n_splits=10)
    working_directory = os.path.join(Generative_results_directory, args.name)

    best_model_data = os.path.join(args.Dimension, "data_"+args.method+".pt")

    bmd = best_model_data.replace("/","")[:-3]
    saving_directory = "/home/jcolombini/Purpose/Labeler/Labeler/Stratified_data/" + args.name 
    if not os.path.isdir(saving_directory):
        os.mkdir(saving_directory)
    # for Num_labels in Num_labelss:
    DATA_r = Load(os.path.join(working_directory,"testset.pt"), args.Num_classes)
    DATA_g = Load(os.path.join(working_directory, best_model_data), args.Num_classes)
    collective_labels=[]
    collective_r_groups=[]

    for j, DATA in enumerate(DATA_r):
        for a in DATA[0]:
            collective_r_groups.append(j)
            collective_labels.append(a)

    s = sgkf.split(X=np.zeros((len(collective_labels),args.Num_classes)), y=collective_labels, groups=collective_r_groups)

    for t,b in enumerate(s):
        fold = list(set([collective_r_groups[i] for i in b[1] ]))
        torch.save([DATA_r[b] for b in fold], saving_directory+"/stratum-"+str(t)+"real_"+str(args.Num_classes)+".pt")
    
    distrib = dict([(i,collective_labels.count(i)/len(collective_labels)) for i in range(13)])

    generated_data_groupings = [{"group":j, "label": a} for j,Data in enumerate(DATA_g) for a in Data[0] ]
    for j, DATA in enumerate(DATA_g):
        for a in DATA[0]:
            generated_data_groupings.append({"group":j, "label": a})
    s = extract_subdatasets(generated_data_groupings,"group","label", 140 ,distrib)
    
    for t, stratum in enumerate(s[:10]):
        torch.save([DATA_g[i] for i in stratum], saving_directory+"/stratum-"+str(t)+"gen_"+str(args.Num_classes)+".pt")


if __name__ == "__main__":
    name = "2025-08-01-12-25"
    stratify(name, "20/data_NF.pt", 13)
