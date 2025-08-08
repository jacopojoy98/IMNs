# IMPORT PACCHETTI
import os
import numpy as np
import json
import time
# from torch import R
import tqdm 
from .Plotting import compared_plot
from . import DataManipulation as DM 
from .Distances import Distances as Distances
from . import JS_Divergence as JS

Matrix_functions = Distances.Matrix_functions_tasks
Graph_functions  = Distances.Graph_functions_tasks
function_names = [f.__name__ for f in Matrix_functions + Graph_functions]

def Distributions_plots(results_of_method, function_names, method, Latent_space):
    aggregated_results_by_size = []
    num_functions = len(results_of_method[0][0][0])
    num_sizes = len(results_of_method[0])
    for i in range(num_sizes):              
        aggregated_results_by_function = []
        for j in range(num_functions):              
            results_TT = [results_of_method[0][i][s][j] for s in range(len(results_of_method[0][i]))]
            results_FT = [results_of_method[1][i][s][j] for s in range(len(results_of_method[1][i]))]
            results_FF = [results_of_method[2][i][s][j] for s in range(len(results_of_method[2][i]))]
            compared_plot(results_TT,results_FT,results_FF, method, Latent_space, function_names[j],i)
            result = JS.DdD(results_TT,results_FT,results_FF)
            aggregated_results_by_function.append(result)
        aggregated_results_by_size.append(aggregated_results_by_function)
    return np.mean(aggregated_results_by_size, axis=0)

def Evaluate_functions (a,b, matrix_functions, graph_functions):
    results = []
    for function in matrix_functions:
        result = function(a,b)
        results.append(result)
    a = DM.Create_Graph(a)
    b = DM.Create_Graph(b)
    for function in graph_functions:
        result = function(a,b)
        results.append(result)
    return results

def Aggregate_results(results_of_method):
    aggregated_results_by_size = []
    num_functions = len(results_of_method[0][0][0])
    num_sizes = len(results_of_method[0])
    for i in range(num_sizes):              
        aggregated_results_by_function = []
        for j in range(num_functions):              
            results_TT = [results_of_method[0][i][s][j] for s in range(len(results_of_method[0][i]))]
            results_FT = [results_of_method[1][i][s][j] for s in range(len(results_of_method[1][i]))]
            results_FF = [results_of_method[2][i][s][j] for s in range(len(results_of_method[2][i]))]
            # Plotting.compared_plot(results_TT,results_FT,results_FF,"size"+str(i)+"_"+function_names[j])
            result = JS.DdD(np.clip(results_TT,-1000,1000).tolist(),np.clip(results_FT,-1000,1000).tolist(),np.clip(results_FF,-1000,1000).tolist())
            aggregated_results_by_function.append(result)
        aggregated_results_by_size.append(aggregated_results_by_function)
    return np.mean(aggregated_results_by_size, axis=0)

def Evaluate(name):
    Matrix_functions = Distances.Matrix_functions_tasks
    Graph_functions  = Distances.Graph_functions_tasks
    function_names = [f.__name__ for f in Matrix_functions + Graph_functions]
    num_functions = len(function_names)
    os.chdir(str(name))
    iterations = 20
    final_results = dict([(function_names[j],[]) for j in range(num_functions)])
    Real_graphs = DM.Load_graphs("testset.pt",10000)    
    print("length of real graphs", len(Real_graphs))
    distance_between_distributions = "was"  
    ## Modificare, dal calcolo di delle distanze tra le distribuzoni delle distanze tra le propietà
    ## delle matrici generate e quelle reali, si passa al calcolo della distanza tra le distribuzioni delle
    ## proprietà stesse

    for Latent_space in [10,20,30,40,50]:       
        for method in ["NF","wgan","rae"]:
            final_results = dict([(function_names[j],[]) for j in range(num_functions)])
            iter = "data_"+method
            path_to_generated = os.path.join(str(Latent_space), iter+".pt")
            total_Gen_graphs  = DM.Load_graphs(path_to_generated, iterations*len(Real_graphs)) 
            len_dataset = len(total_Gen_graphs)
            for s in tqdm.tqdm(range(iterations), desc = method):
                Gen_graphs = total_Gen_graphs[s*(len_dataset//iterations):(s+1)*(len_dataset//iterations)]
                Groups = DM.Create_Groups(Real_graphs, Gen_graphs)
                results_of_method = []
                for i, Group in enumerate(Groups):      
                    results_of_group = []
                    for k, Group_of_size in enumerate(Group):
                        results_of_size = []
                        for (Matrix_a,Matrix_b) in  Group_of_size:
                            _results = Evaluate_functions(Matrix_a, Matrix_b, Matrix_functions, Graph_functions)
                            results_of_size.append(_results)
                        results_of_group.append(results_of_size)
                    results_of_method.append(results_of_group)
                results = Aggregate_results(results_of_method)
                if not os.path.isdir(os.path.join(str(Latent_space),"distributions")): os.mkdir(os.path.join(str(Latent_space),"distributions"))
                if s == iterations-1:
                    Distributions_plots(results_of_method, function_names, method, Latent_space)
                for j, function_name in enumerate(function_names):
                    final_results[function_name].append(results[j])
            with open(os.path.join(str(Latent_space),distance_between_distributions+"_results_"+method+".json"), "w") as file:
                json.dump(final_results, file)

    for method in ['ete_Wgan','ete_NF']:
        total_Gen_graphs = DM.Load_graphs('ETE/data_'+method+'.pt', iterations*len(Real_graphs))
        len_dataset = len(total_Gen_graphs)
        if not os.path.exists("ETE"):os.mkdir("ETE")
        for s in tqdm.tqdm(range(iterations), desc = method):
            Gen_graphs = total_Gen_graphs[s*(len_dataset//iterations):(s+1)*(len_dataset//iterations)]
            Groups = DM.Create_Groups(Real_graphs, Gen_graphs)
            results_of_method = []
            for i, Group in enumerate(Groups):      
                results_of_group = []
                for k, Group_of_size in enumerate(Group):
                    results_of_size = []
                    for (Matrix_a,Matrix_b) in  Group_of_size:
                        results = Evaluate_functions (Matrix_a, Matrix_b, Matrix_functions, Graph_functions)
                        results_of_size.append(results)
                    results_of_group.append(results_of_size)
                results_of_method.append(results_of_group)
            results = Aggregate_results(results_of_method)
            if not os.path.isdir(os.path.join("ETE","distributions")): os.mkdir(os.path.join("ETE","distributions"))
            if s == iterations-1:
                Distributions_plots(results_of_method, function_names, method, "ETE")
            for j, function_name in enumerate(function_names):
                final_results[function_name].append(results[j])
        with open(os.path.join("ETE",distance_between_distributions+"_results_"+method+".json"), "w") as file:
            json.dump(final_results, file) 
        

if __name__ == "__main__":
    # path = "/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-13-EPOCHS(2000, 2000, 400)-LR(0.0001, 0.0005, 0.0001)-ALPHA0.0003NUM_CLSS13"
    path = "/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-04-01-00-16"
    Evaluate(path)


