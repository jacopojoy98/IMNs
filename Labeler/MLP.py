import os
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import Loader as Loader
import shutil

class MLP(nn.Module):
    def __init__(self,num_labels):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(1024, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, num_labels),
          nn.Sigmoid()
        )

    def forward(self, x):
        x=self.layers(x.flatten())
        denom = x.norm(p=1).clamp_min(1e-13)
        return x/denom


def shift_tensor(tensor):
    return torch.cat((tensor[:, 1:], tensor[:, :1]), dim=1)

def printout(out):
    p = torch.zeros((len(out),len(out[0])))
    for i, ou in enumerate(out):
        for j, entry in enumerate(ou):
            if entry > 0.1:
                p[i][j] = entry.item()
            else:
                p[i][j] = 0
    print(p)

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

def purp(G_list,a):
    matrix = np.zeros((a))
    for G in G_list:
        for p in G:
            matrix += p.detach().numpy()
    return matrix

def groupappend(a,b):
    for i,e in enumerate(a):
        b[i].append(e)

def groupreset(a):
    for e in a:
        e.reset() 


def gnn(args):
    DATA_DIR = "/home/jcolombini/Purpose/Labeler/Labeler/Stratified_data"
    savedir =  os.path.join("/home/jcolombini/Purpose/Labeler/Results/Labeler_results", args.savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
    # shutil.copy("/home/jcolombini/Purpose/Labeler/Labeler/config.py" ,os.path.join(savedir,"config.py"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for percentage in args.percentages:
        for fold in range(10):
            test = []
            for graph in torch.load(DATA_DIR+"/"+args.name+"/stratum-"+str(fold)+"real_"+str(args.Num_labels)+".pt") :
                test.append((torch.tensor(graph[0]).to(device), torch.tensor(graph[1]).to(device)))

            train=[]
            for s in range(10):
                if s==fold:
                    continue
                else:
                    for graph in torch.load(DATA_DIR+"/"+args.name+"/stratum-"+str(fold)+"real_"+str(args.Num_labels)+".pt"):
                        train.append((torch.tensor(graph[0]).to(device), torch.tensor(graph[1]).to(device)))

            for i in range(percentage//10):
                for graph in torch.load(DATA_DIR+"/"+args.name+"/stratum-"+str(fold)+"gen_"+str(args.Num_labels)+".pt"):
                    train.append((torch.tensor(graph[0]).to(device), torch.tensor(graph[1]).to(device)))


            modelmsp = MLP(args.Num_labels).to(device)

            optimizer = torch.optim.Adam(modelmsp.parameters(), lr=args.LR)
            los = torch.nn.BCELoss(reduction='sum')

            losses = []
            testlosses = []
            Accuracies = []
            Recalls = []
            Precisions = []
            square = np.zeros((args.Num_labels,args.Num_labels))

            for epoch in range(args.Epochs):
                modelmsp.train()
                loss_sum = 0
                for i, (label, data) in enumerate(train):
                    out= []
                    optimizer.zero_grad()
                    target = F.one_hot(label, num_classes=args.Num_labels).type(torch.float32)
                    for j in range(len(data[0])):
                        padded_data = F.pad(data,(0,32-len(data), 0, 32-len(data)))
                        tmp_out = modelmsp(padded_data.type(torch.float32)).sigmoid()
                        out.append(tmp_out)
                        data = shift_tensor(data)
                    out = torch.stack(out)
                    loss = los(out,target)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()

                modelmsp.eval()
                testloss_sum =0 
                for (labels, test_) in test:
                    #test.x,test.edge_index,test.edge_weight
                    out=[]
                    for j in range(len(test_[0])):
                        #data.x,data.edge_index,data.edge_weight
                        padded_test_ = F.pad(test_,(0,32-len(test_), 0, 32-len(test_)))
                        tmp_out = modelmsp(padded_test_.type(torch.float32)).sigmoid()
                        out.append(tmp_out)
                        test_ = shift_tensor(test_)
                    out = torch.stack(out)
                    targettest =  F.one_hot(labels, num_classes = args.Num_labels).type(torch.float32)
                    loss = los(out,targettest)
                    testloss_sum +=  loss.item()                    
                    ii = torch.argmax(out,  dim = 1)
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

                print(f"epoch: {epoch}\t- loss:  {loss_sum/len(train)} \n perc:{percentage}   \t- tloss: {testloss_sum/len(test)} \n fold:{fold}   \t- accuracy: {Accuracy}\
                       \n         \t- Recall: {np.mean(Recall)} \n         \t- Precision: {np.mean(Precision)}", end = "\r")

            plot(losses, "trainlosses", percentage, savedir)
            plot(testlosses, "testlosses", percentage, savedir)
            plot(Accuracies, "accuracy", percentage, savedir)
            Multiple_plot(Recalls, "Recall", percentage, savedir)
            Multiple_plot(Precisions, "Precision", percentage, savedir)

            np.savetxt(savedir + "//testloss.txt", testlosses )
            np.savetxt(savedir + "//trainloss.txt", losses )
            np.savetxt(savedir + "//Accuracy"+str(percentage)+"f"+str(fold)+".txt", Accuracies )
            np.savetxt(savedir + "//Recall"+str(percentage)+"f"+str(fold)+".txt", Recalls)
            np.savetxt(savedir + "//Precision"+str(percentage)+"f"+str(fold)+".txt", Precisions)

if __name__ == "__main__":
    pass
    # dir = os.curdir
    # for x in percentages:
    #     arr = np.loadtxt(dir+"//dgb//precision"+str(x)+".txt")
    #     arr = arr[100:]
    #     mean = np.mean(arr)
    #     var = np.var(arr)
    #     print(f"perc{x}- mean {mean} -var {var}")