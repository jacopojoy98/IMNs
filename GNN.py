import os
import torch
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn.models import GCN
import Labeler.Loader as Loader

def micro_accuracy(matrix):
    micro_acc = 0
    t_classes = 0
    for j,row in enumerate(matrix):
        if np.sum(row):
            micro_acc += row[j]/np.sum(row)
            t_classes +=1
    return micro_acc/t_classes


def printout(out):
    p = torch.zeros((len(out),len(out[0])))
    for i, ou in enumerate(out):
        for j, entry in enumerate(ou):
            if entry > 0.1:
                p[i][j] = entry.item()
            else:
                p[i][j] = 0
    print(p)

def purp(G_list,a):
    matrix = np.zeros((a))
    for G in G_list:
        for p in G:
            matrix += p.detach().numpy()
            # input()
    return matrix

def calculate_statistics(conf_matrix):
    if isinstance(conf_matrix, list):
        conf_matrix = np.array(conf_matrix)
    Recall = []
    for j, row in enumerate(conf_matrix):
        if sum(row) != 0:
            Recall.append(row[j]/sum(row))
    
    Precision = []
    for j, row in enumerate(conf_matrix.T):
        if sum(row) != 0:
            Precision.append(row[j]/sum(row))
        else: 
            Precision.append(0)
    
    Accuracy = np.sum(np.diag(conf_matrix))/(np.sum(conf_matrix))
    return Recall, Precision, Accuracy

def Multiple_plot(Array, name, percentage, savedir):
    errs = np.array([np.std(vals) for vals in Array])
    Array = np.array(Array).T
    for line in Array:
        plt.plot(np.linspace(0,len(line),len(line)),line , alpha=0.05, color = "blue")
    mean = np.mean(Array, axis =0)
    plt.plot(np.linspace(0,len(mean),len(mean)),mean, color = "black")
    plt.fill_between(np.linspace(0,len(mean),len(mean)), mean-errs, mean+errs, alpha=0.05, color = "blue")
    plt.title(name)
    plt.savefig(savedir + "//"+name+str(percentage)+".png")
    plt.close()


def plot(array, name, percentage, savedir): 
    plt.plot(np.linspace(0,len(array),len(array)),array)
    plt.title(name)
    plt.savefig(savedir + "//"+name+str(percentage)+".png")
    plt.close()

def groupappend(a,b):
    for i,e in enumerate(a):
        b[i].append(e)

def groupreset(a):
    for e in a:
        e.reset() 

EPOCHS = 1500
LR = 2e-4
percentages = [0,20,40,60,80,100]
Num_labelss = [13]

def gnn(name, root=None):
    Num_labels = 13
    savedir = "Results/Labeler_results/"+datetime.now().strftime('%Y-%m-%d-%H-%M')+"E="+str(EPOCHS)+"_LR="+str(LR)+"HC=64_NL="+str(Num_labels)
    DATA_DIR = "/home/jcolombini/Purpose/Labeler/Labeler/Stratified_data"
    for percentage in percentages:
        for fold in range(10):
            test = Loader.Graph_data(DATA_DIR+"/"+name+"/stratum-"+str(fold)+"real_"+str(Num_labels)+".pt") 
            train=[]
            for s in range(10):
                if s==fold:
                    continue
                else:
                    for graph in Loader.Graph_data(DATA_DIR+"/"+name+"/stratum-"+str(fold)+"real_"+str(Num_labels)+".pt"):
                        train.append(graph)

            for i in range(percentage//10):
                for graph in Loader.Graph_data(DATA_DIR+"/"+name+"/stratum-"+str(fold)+"gen_"+str(Num_labels)+".pt"):
                    train.append(graph)

            os.makedirs(savedir, exist_ok = True)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(device)
            #device=torch.device('cpu')
            modelmsp = GCN(
                in_channels = 2,
                hidden_channels=64,
                num_layers = 32,
                out_channels=Num_labels,
                aggr = "max"
            ).to(device)

            optimizer = torch.optim.Adam(modelmsp.parameters(), lr=LR)
            los = torch.nn.BCELoss(reduction='sum')

            losses = []
            testlosses = []
            Accuracies = []
            Recalls = []
            Precisions = []

            for epoch in range(EPOCHS):
                square = np.zeros((Num_labels,Num_labels))
                modelmsp.train()
                loss_sum = 0
                for i, data in enumerate(train):
                    optimizer.zero_grad()#data.x,data.edge_index,data.edge_weight
                    out = modelmsp(data.x, data.edge_index, data.weight.type(torch.float32)).sigmoid()
                    target = F.one_hot(data.y, num_classes=Num_labels).type(torch.float32)
                    loss = los(out,target)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()

                modelmsp.eval()
                testloss_sum =0
                for test_ in test:
                    testout = modelmsp(test_.x, test_.edge_index, test_.weight.type(torch.float32)).sigmoid()
                    targettest =  F.one_hot(test_.y, num_classes = Num_labels).type(torch.float32)
                    loss = los(testout,targettest)
                    testloss_sum +=  loss.item()
                    ii = torch.argmax(testout,  dim = 1)
                    jj = torch.argmax(targettest, dim = 1)
                    for node in range(len(ii)):
                        guessed_label = ii[node]
                        real_label = jj[node]
                        square[real_label][guessed_label] +=1
                    Recall, Precision, Accuracy = calculate_statistics(square)

                losses.append(loss_sum/len(train))
                testlosses.append(testloss_sum/len(test))
                Accuracies.append(Accuracy)
                Recalls.append(Recall)
                Precisions.append(Precision)

                print(f"epoch: {epoch}\t- loss:  {loss_sum/len(train)} \n         \t- tloss: {testloss_sum/len(test)} \n         \t- accuracy: {Accuracy}\
                       \n         \t- Recall: {np.mean(Recall)} \n         \t- Precision: {np.mean(Precision)}")

            plot(losses, "trainlossesf"+str(fold), percentage, savedir)
            plot(testlosses, "testlossesf"+str(fold), percentage, savedir)
            plot(Accuracies, "accuracyf"+str(fold), percentage, savedir)
            Multiple_plot(Recalls, "Recallf"+str(fold), percentage, savedir)
            Multiple_plot(Precisions, "Precisionf"+str(fold), percentage, savedir)


            np.savetxt(savedir + "//testlossf"+str(percentage)+str(fold)+".txt", testlosses )
            np.savetxt(savedir + "//trainlossf"+str(percentage)+str(fold)+".txt", losses )
            np.savetxt(savedir + "//Accuracy"+str(percentage)+"f"+str(fold)+".txt", Accuracies )
            np.savetxt(savedir + "//Recall"+str(percentage)+"f"+str(fold)+".txt", Recalls)
            np.savetxt(savedir + "//Precision"+str(percentage)+"f"+str(fold)+".txt", Precisions)
            

if __name__ == "__main__":
    gnn("2025-03-27-09-4250data_NF")
    # dir = os.curdir
    # for x in percentages:
    #     arr = np.loadtxt(dir+"//dgb//precision"+str(x)+".txt")
    #     arr = arr[100:]
    #     mean = np.mean(arr)
    #     var = np.var(arr)
    #     print(f"perc{x}- mean {mean} -var {var}")