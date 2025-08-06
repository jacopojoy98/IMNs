import numpy as np
import networkx as nx
from scipy.stats import entropy
from numpy.linalg import norm

def pad(list,n):
    while(len(list)<n):
        list.append(0)
    return list

def NRSE(dist_1, dist_2):
    ll  = max([len(dist_1), len(dist_2)])
    dist_1 = pad(dist_1, ll)
    dist_2 = pad(dist_2, ll)
    return np.linalg.norm(subtract(dist_1,dist_2))

def dict_to_lsit(dict):
    list = []
    for i in dict:
        list.append(dict[i])
    return list

def subtract (a,b):
    return [a_i - b_i for a_i, b_i in zip(a, b)]

####

class Distances:
    Matrix_functions_tasks = []
    Graph_functions_tasks = []

    @classmethod
    def Matrix_functions(cls, func):
        cls.Matrix_functions_tasks.append(func)  # Store method name
        return func

    @classmethod
    def Graph_functions(cls, func):
        cls.Graph_functions_tasks.append(func)
        return func
    
# class containing the functions. 

class MyTasks(Distances):
    @Distances.Matrix_functions
    def Frobenius_distance(a,b):
        a_b = [[x_a-x_b for x_a,x_b in zip(rowa,rowb)] for rowa,rowb in zip(a,b)]
        return np.linalg.norm(a_b)

    @Distances.Matrix_functions
    def JSD(fP, fQ):
        P = [item for sublist in fP for item in sublist]
        Q = [item for sublist in fQ for item in sublist]
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    # @Distances.Graph_functions
    # def Laplacian_spectrum_distance(a,b):
    #     a_ = a.to_undirected()
    #     b_ = b.to_undirected()
    #     return NRSE((nx.linalg.laplacian_spectrum(a_, weight=None)).tolist(), nx.linalg.laplacian_spectrum(b_, weight=None).tolist())

    @Distances.Graph_functions
    def Centrality_distance(a,b):
        return NRSE( dict_to_lsit(nx.centrality.betweenness_centrality(a)), dict_to_lsit(nx.centrality.betweenness_centrality(b)))

    @Distances.Graph_functions
    def Density_distance(a,b):
      return np.abs(nx.function.density(a)-nx.function.density(b))

    @Distances.Graph_functions
    def Degree_distance(a,b):
        return NRSE(nx.function.degree_histogram(a),nx.function.degree_histogram(b))









