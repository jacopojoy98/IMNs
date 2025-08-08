import torch
import dgl
import DataManipulation as DM
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import approximation
import os
import json
import numpy as np
from evaluation.evaluator import Evaluator
from evaluation.gin_evaluation import load_feature_extractor, MMDEvaluation, KIDEvaluation, FIDEvaluation, prdcEvaluation
import warnings
warnings.filterwarnings("ignore")

def mode(matrix):
    flat_list = [item for sublist in matrix for item in sublist]
    element_count = {}
    for element in flat_list:
        element_count[element] = element_count.get(element, 0) + 1
    max_count = max(element_count.values())
    mode = [element for element, count in element_count.items() if count == max_count]
    return mode[0]

def calculate_sample_variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Sample variance requires at least 2 data points.")

    # Step 2: Calculate the sample mean
    sample_mean = sum(data) / n

    # Step 3: Calculate the sum of squared differences
    squared_diff_sum = sum((x - sample_mean) ** 2 for x in data)

    # Step 4: Divide the sum of squared differences by (n-1) to get the sample variance
    sample_variance = squared_diff_sum / (n - 1)

    return sample_mean, np.sqrt(sample_variance)

def cropped_matrix(matrix):
    maxi = 0
    maxj = 0
    for i, row in enumerate(matrix[1:]):
        for j, element in enumerate(row):
            if element > 0.05:
                if i > maxi:
                    maxi = i
                if j > maxj:
                    maxj = j
    new_matrix = matrix[:(maxj+1),:maxj]
    for i, row in enumerate(new_matrix):
        for j, element in enumerate(row):
            if element <= max(mode(matrix),0.05):
                new_matrix[i][j] = 0
    return new_matrix


def get_max(list):
    max=0
    max_i=0
    for i in range(13):
        if list[i]>max:
            max = list[i]
            max_i =i
    return max_i

def erase_ones(list):
    j=0
    for i in range(1,31):
        if list[-i]<0.8:
            break
        list[-i]=0
        j+=1
    return list, 32-j

def Create_Graph(GraphMatrix):
    NN = Number_of_nodes_of_graph_from_matrix(GraphMatrix)
    purpose = np.argmax(GraphMatrix[:13], axis=0)
    Omega = GraphMatrix[13:13+NN,:NN]
    threshold = np.percentile(Omega, 96) / 2  ### VIP
    G = nx.DiGraph()
    NX_G_R = []
    for i,row in enumerate(Omega):
        for j, el in enumerate(row):
            if (el>threshold):
                NX_G_R.append((i,j,el))
    for i in range(NN):
        G.add_node(i,x=1, y = purpose[i])
    G.add_weighted_edges_from(NX_G_R)
    return G

def Number_of_nodes_of_graph_from_matrix(matrix):
    if isinstance(matrix, torch.Tensor):
        tmppurp =  np.sum(matrix[:13].detach().numpy(), axis=0)
    else:
        # print(matrix)
        tmppurp = np.sum(matrix[:13], axis=0)
    purpTresh = np.percentile(tmppurp, 90) / 2  ###VIP
    estimated_nodes = np.sum(tmppurp > purpTresh)
    return estimated_nodes

def Create_Graph_List(path):
    list_generated = os.listdir(path)
    List_Graphs = []
    for graphname in list_generated:
        path_to_graph = os.path.join(path,graphname)
        GraphMatrix = np.loadtxt(path_to_graph)
        Omega = GraphMatrix[2:]
        tau = GraphMatrix[1]
        Purpose = GraphMatrix[0]
        Graph = Create_Graph(Omega,tau,Purpose)
        List_Graphs.append(Graph)
    return List_Graphs

def dotheeval(name):
    device = torch.device('cpu')
    gin = load_feature_extractor(device=torch.device('cpu'))
    evaluator = Evaluator(device=device)
    os.chdir(str(name))
    Real_graphs = []
    R = DM.Load_graphs("testset.pt",10000)
    for s in R:
        Real_graphs.append(DM.Create_Graph(s))
    lobsters = [dgl.from_networkx(g) for g in Real_graphs]
    iterations=20

    for Latent_space in [10,20,30,40,50]:       
        for method in ["wgan","rae","NF"]:
            iter = "data_"+method
            path_to_generated = os.path.join(str(Latent_space), iter+".pt")
            list_generated  = DM.Load_graphs(path_to_generated, iterations*len(Real_graphs)) 
            mmd_array = []
            kid_array = []
            fid_array = []
            p_array = []
            r_array = []
            d_array = []
            c_array = []
            f1_pr_array = []
            for s in range(iterations):  
                print(f"Percentuale completata: {s*100//iterations}%")
                Gen_graphs  = [] 
                for graph in list_generated[s*len(list_generated)//iterations:(s+1)*len(list_generated)//iterations]:
                    Gen_graphs.append(DM.Create_Graph(graph.detach().numpy()))
                grids = [dgl.from_networkx(g) for g in Gen_graphs]
                result = evaluator.evaluate_all(generated_dataset=grids, reference_dataset=lobsters)
                mmd_array.append(result['mmd_linear'].item())
                kid_array.append(result['kid'].item())
                fid_array.append(result['fid'].item())
                p_array.append(result['precision'].item())
                r_array.append(result['recall'].item())
                d_array.append(result['density'].item())
                c_array.append(result['coverage'].item())
                f1_pr_array.append(result['f1_pr'].item())

            with open(os.path.join(str(Latent_space),"was_results_"+method+".json"), "r") as file:
                final_results = json.load(file)  
            final_results["MMD"] = mmd_array 
            final_results["KID"] = kid_array
            final_results["FID"] = fid_array
            final_results["Precision"] = p_array
            final_results["Recall"] = r_array
            final_results["Density"] = d_array
            final_results["Coverage"] = c_array
            final_results["F1_PR"] = f1_pr_array
            with open(os.path.join(str(Latent_space),"was_results_"+method+".json"), "w") as file:
                json.dump(final_results, file)

    for method in ['ete_Wgan','ete_NF']:
        mmd_array = []
        list_generated = DM.Load_graphs('ETE/data_'+method+'.pt', iterations*len(Real_graphs))
        for s in range(iterations):
            print(f"Percentuale completata: {s*100//iterations}%")
            Gen_graphs  = [] 
            for graph in list_generated[s*len(list_generated)//iterations:(s+1)*len(list_generated)//iterations]:
                Gen_graphs.append(DM.Create_Graph(graph.detach().numpy()))
            grids = [dgl.from_networkx(g) for g in Gen_graphs]
            result = evaluator.evaluate_all(generated_dataset=grids, reference_dataset=lobsters)
            mmd_array.append(result['mmd_linear'].item())
            kid_array.append(result['kid'].item())
            fid_array.append(result['fid'].item())
            p_array.append(result['precision'].item())
            r_array.append(result['recall'].item())
            d_array.append(result['density'].item())
            c_array.append(result['coverage'].item())
            f1_pr_array.append(result['f1_pr'].item())
        with open(os.path.join("ETE","was_results_"+method+".json"), "r") as file:
            final_results = json.load(file)  
        final_results["MMD"] = mmd_array 
        final_results["KID"] = kid_array
        final_results["FID"] = fid_array
        final_results["Precision"] = p_array
        final_results["Recall"] = r_array
        final_results["Density"] = d_array
        final_results["Coverage"] = c_array
        final_results["F1_PR"] = f1_pr_array
        with open(os.path.join("ETE","was_results_"+method+".json"), "w") as file:
            json.dump(final_results, file)

if __name__ == "__main__":
    r_dir = "/home/jcolombini/Purpose/Labeler/Results/Generative_results"
    name = os.path.join(r_dir, "2025-08-07-11-23")
    dotheeval(name)
            # dotheeval(name+"_"+str(p)+"_"+str(s))