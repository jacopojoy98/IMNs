from Generative_architecture.Data_pulling import Extract_Dataset
from Generative_architecture.Evaluation.DataManipulation import Create_Graph, Number_of_nodes_of_graph_from_matrix
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
root = "/home/jcolombini/Purpose/Labeler/Data/NHTS"
dataset = Extract_Dataset(root)
purposes = []
Number_of_nodes = [] 
edges = []
density = []
avg_degree = []
min_degree = []
max_degree = []
for data in dataset:
    p = np.argmax(data[:13], axis=0)
    NN = Number_of_nodes_of_graph_from_matrix(data)
    Graph = Create_Graph(data)
    density.append(nx.density(Graph))   
    avg_degree.append(np.mean(list(dict(Graph.degree()).values())))
    min_degree.append(np.min(list(dict(Graph.degree()).values())))
    max_degree.append(np.max(list(dict(Graph.degree()).values())))
    edges.append(Graph.number_of_edges())
    for e in p[:NN]:
        purposes.append(e)
    Number_of_nodes.append(NN)

plt.hist(purposes,bins=np.arange(0, 13), align='left', rwidth=0.8)
plt.title("Purpose distribution")
plt.xlabel("Purpose")
plt.ylabel("Count")
plt.savefig("Purpose_distribution.png")
plt.close()

plt.hist(Number_of_nodes, bins=np.arange(0, max(Number_of_nodes)+1), align='left', rwidth=0.8)
plt.title("Number of nodes distribution")
plt.xlabel("Number of nodes")
plt.savefig("Number_of_nodes_distribution.png")
plt.close()

plt.hist(edges, bins=np.arange(0, max(edges)+1), align='left', rwidth=0.8)
plt.title("Edges distribution")
plt.xlabel("Number of edges")
plt.savefig("Edges_distribution.png")
plt.close() 

plt.hist(avg_degree, bins=15)
plt.title("Average degree distribution")   
plt.xlabel("Average degree")
plt.savefig("Average_degree_distribution.png")
plt.close()

plt.hist(min_degree, bins=15)
plt.title("Minimum degree distribution")
plt.xlabel("Minimum degree")
plt.savefig("Minimum_degree_distribution.png")
plt.close()

plt.hist(max_degree, bins=12)
plt.title("Maximum degree distribution")
plt.xlabel("Maximum degree")
plt.savefig("Maximum_degree_distribution.png")
plt.close()

plt.hist(density, bins=12)
plt.title("Density distribution")
plt.xlabel("Density")
plt.savefig("Density_distribution.png")
plt.close() 