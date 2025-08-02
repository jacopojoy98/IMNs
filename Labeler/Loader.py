import os,sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Generative_architecture.Evaluation.DataManipulation import Number_of_nodes_of_graph_from_matrix, Create_Graph_from_tuple
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

def Load(file, Num_classes):
    matrices = torch.load(file, weights_only=True)
    out = []
    for matrix in matrices:
        matrix = matrix.detach().numpy()
        NN = Number_of_nodes_of_graph_from_matrix(matrix)  ### Modificare su torch.quantile e usare torch invece che numpy
        purpose = np.argmax(matrix[:13], axis=0)[:NN]       #  torch.quantile(tensor, 0.9, "linear")
        purpose = [min(p, Num_classes - 1) for p in purpose]
        Omega = matrix[13:13+NN,:NN]
        threshold = np.percentile(Omega, 90) / 2
        Omega[Omega<threshold] = 0
        out.append((purpose, Omega))
    return out

# def Load_MLP(file):
#     matrices = torch.load(file, weights_only=False)
#     out = []
#     for matrix in matrices:
#         print(matrix)
#         matrix = matrix.detach().numpy()
#         NN = Number_of_nodes_of_graph_from_matrix(matrix)  ### Modificare su torch.quantile e usare torch invece che numpy
#         purpose = np.argmax(matrix[:13], axis=0)[:NN]       #  torch.quantile(tensor, 0.9, "linear")
#         Omega = matrix[13:13+NN,:NN]
#         threshold = np.percentile(Omega, 90) / 2
#         Omega[Omega<threshold] = 0
#         out.append((torch.tensor(purpose), torch.tensor(Omega)))
#     return torch.stack(out)

def Load_graphs(file):
    matrices = torch.load(file, weights_only=False)
    return [Create_Graph_from_tuple(matrix) for matrix in matrices]

def Graph_data(file):
    graphs = Load_graphs(file)
    Dataset = []
    for graph in graphs:
        pyg_graph = from_networkx(graph)
        Dataset.append(pyg_graph)
    return Dataset

if __name__ == "__main__":
    pass