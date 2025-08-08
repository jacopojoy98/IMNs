import pickle
import numpy as np
import networkx as nx
import torch
import os, sys
from operator import itemgetter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import  Generative_architecture.Data_pulling
from GLOBAL_FUNCTIONS import plots

def fileread(filename):

    ## Simple function that unpickles the input file

    with open(filename , "rb") as f:
        matrix= pickle.load(f)
    return matrix

def Create_Graph_from_tuple(data_tuple):

    # input : tuple(purposes, matrix) -
    # output: networkx_graph
    # 
    # The purposes are give in categoric format, as for the output
    # The omega matrix is assumed to be already cleaned from spurious elements

    purpose = data_tuple[0]
    Omega = data_tuple[1]
    G = nx.DiGraph()
    NX_G_R = []
    for i,row in enumerate(Omega):
        for j, el in enumerate(row):
            if el >0:
                NX_G_R.append((i,j,el))
    s = 1
    for i in range(len(purpose)):
        G.add_node(i, x= [float(1),float(1)], y = purpose[i])  ##### 
    G.add_weighted_edges_from(NX_G_R)
    return G

def Create_Graph(GraphMatrix):
    NN = Number_of_nodes_of_graph_from_matrix(GraphMatrix)
    purpose = np.argmax(GraphMatrix[:10], axis=0)
    Omega = GraphMatrix[10:10+NN,:NN]
    # threshold = np.percentile(Omega, 90) / 2  ### VIP
    threshold = 0.05
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
        tmppurp =  np.sum(matrix[:10].detach().numpy(), axis=0)
    else:
        tmppurp = np.sum(matrix[:10], axis=0)
    purpTresh = np.percentile(tmppurp, 90) / 2  ###VIP
    estimated_nodes = np.sum(tmppurp > purpTresh)
    return estimated_nodes

def Load_graphs(path, quantity):   ### Spostare il load fuori per maggiore efficenza
    graphs = []
    matrices = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
    if quantity>len(matrices): quantity=len(matrices)
    for index in range(quantity):
        matrix = matrices[index]
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().numpy()
        if np.isnan(matrix).any():
            continue
        graphs.append(matrix)
    return graphs

def split_on_size(matrices, split_size):
    matrices_size = len(matrices)
    matrices_with_size = [[Number_of_nodes_of_graph_from_matrix(matrix), matrix] for matrix in matrices]
    matrices_with_size.sort(key = itemgetter(0))
    matrixes = [matrix[1] for matrix in matrices_with_size]
    out =  [matrixes[int(s*matrices_size/split_size): int((s+1)*matrices_size/split_size) ] for s in range(split_size)]
    return out

def create_unordered_couples(Groups1,Groups2, split_size):
    return [[(n,m) for j, n in enumerate(Groups1[i]) for m in Groups2[i][(j+1):]] for i in range(split_size)]

def create_tiping_couples(Groups1, Groups2, split_size):
    couple_1 = create_unordered_couples(Groups1, Groups1, split_size)
    couple_2 = create_unordered_couples(Groups1, Groups2, split_size)
    couple_3 = create_unordered_couples(Groups2, Groups2, split_size)
    return couple_1, couple_2, couple_3

def Create_Groups(Real_matrices, Synt_matrices):
    Sized_real_matrices = split_on_size(Real_matrices, 4)
    Sized_synt_matrices = split_on_size(Synt_matrices, 4)
    TT, FT, FF = create_tiping_couples(Sized_real_matrices, Sized_synt_matrices, 4)
    Groups = [TT,FT,FF]
    return Groups

def discard_matrix(matrix):
    ## Possible function used to discard some particularly ugly
    #  generated matrices
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().numpy()
    c1 = 0
    c2 = 0
    purposes = matrix[:10]
    adjacency = matrix[10:]
    sum_p = np.sum(purposes , axis = 0)
    diff_p = [min(abs(x-0),abs(x-1)) for x in sum_p]
    if np.sum(diff_p) > 2:
        c1 =1
    sum_a = np.sum(adjacency, axis = 1)
    diff_a = [min(abs(x-0),abs(x-1)) for x in sum_a]
    if np.sum(diff_a) > 4:
        c2 =1
    return  np.sum(diff_a)


if __name__ == "__main__":
    file = "/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-17-EPOCHS(1000, 1500, 200)-LR(0.0002, 0.0003, 0.0001)-ALPHA0.0003NUM_CLSS13/50/data_NF.pt"
    matrices = torch.load(file, weights_only=True)
    for matrix in matrices:
        discard_matrix(matrix.detach().numpy())
        # matrix = matrix.detach().numpy()
        # NN = Number_of_nodes_of_graph_from_matrix(matrix)
        # print(f"Numero di nodi calcolato = {NN}")
        # Omega = matrix[13:13+NN,:NN]
        # plots(Omega, "matrix")
        # threshold = np.percentile(Omega, 96) / 2
        # Omega[Omega<threshold]=0
        # plots(Omega,"cleared_matrix")
        # input()