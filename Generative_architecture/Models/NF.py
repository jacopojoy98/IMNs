#%%
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from Data_pulling import Extract_Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_pulling import Latent_dimension_dataset
from GLOBAL_FUNCTIONS import save_on_report
import config as config

import normflows as nf

def Normalizing_Flow (num_layers, LATENT_DIM):
    base = nf.distributions.base.DiagGaussian(LATENT_DIM)
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([LATENT_DIM//2, config.HIDDEN_FLOW_DIM1, config.HIDDEN_FLOW_DIM2, LATENT_DIM], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(LATENT_DIM, mode='swap'))
    model = nf.NormalizingFlow(base, flows)
    return model

def train_NF(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for l, LATENT_DIM in enumerate(config.LATENT_DIMENSIONS):
        # Define 2D Gaussian base distribution
        # Define list of flows
        model = Normalizing_Flow(config.NUM_FLOW_LAYESRS, LATENT_DIM).to(device)

        dataset =  torch.stack([X.view([LATENT_DIM]).to(device) for X in  Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_train")])
        testset =  torch.stack([X.view([LATENT_DIM]).to(device) for X in  Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_test")]) 
        # TUTTO QUESTO VA SOSTITUITO CON DUE RIGHE
        optimizer = torch.optim.Adam(model.parameters(), lr=args.Learning_rate_NF)
        Savetrainloss= []
        Savetestloss= []
        start_time = time.time()
        for epoch in range(args.Epochs_NF):
            optimizer.zero_grad()
            loss = model.forward_kld(dataset)
            loss.backward()
            optimizer.step()
            Savetrainloss.append(loss.item())
            testloss = model.forward_kld(testset)
            Savetestloss.append(testloss.item())
            if epoch%20 ==0:
                print("[Epoch %d/%d] [loss: %f] "% (epoch, args.Epochs_NF, testloss.item()))
        torch.save(model.state_dict(),os.path.join(str(LATENT_DIM), config.NF_model_file))
        save_on_report(os.path.join(str(LATENT_DIM),"report.json"), "NF_testloss",Savetestloss)
        save_on_report(os.path.join(str(LATENT_DIM),"report.json"), "NF_trainloss",Savetrainloss)
        save_on_report(os.path.join(str(LATENT_DIM),"report.json"), "NF_traintime",time.time()-start_time)

####3   Come fare le cose Learnable

def end_to_end_NF(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    START_DIM = 32*(32+13)
    model = Normalizing_Flow(config.NUM_ETE_FLOW_LAYESRS, START_DIM).to(device)
    dataset = torch.stack([X.view(START_DIM).float().to(device) for X in torch.load('trainset.pt',weights_only=False)])
    testset = torch.stack([X.view(START_DIM).float().to(device) for X in torch.load('testset.pt', weights_only=False)])
    if not os.path.exists("ETE"):
            os.mkdir("ETE")
    os.chdir("ETE")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR_ete_NF)
    Savetrainloss= []
    Savetestloss= []
    start_time = time.time()
    for epoch in range(args.Epochs_ete_NF):
        optimizer.zero_grad()
        loss = model.forward_kld(dataset)
        loss.backward()
        optimizer.step()
        Savetrainloss.append(loss.item())
        testloss = model.forward_kld(testset)
        Savetestloss.append(testloss.item())
        if epoch%20 ==0:
            print("[Epoch %d/%d] [loss: %f] "% (epoch, args.Epochs_ete_NF, testloss.item()))

    torch.save(model.state_dict(), 'NF_end_to_end.pt')
    np.savetxt("NF_testloss.txt",Savetestloss)
    np.savetxt("NF_trainloss.txt",Savetrainloss)
    np.savetxt("NF_traintime.txt",np.array([time.time()-start_time]))
    os.chdir("..")


if __name__ =="__main__":
    pass
    # os.chdir("/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-12-EPOCHS(40, 2000, 400)-LR(0.0001, 0.0005, 0.0001)-ALPHA0.0003NUM_CLSS13")
    # NEW_NF(100,1e-4)

# BACKUP OLD NF
# def train_NF(EPOCHS, LR):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     for l, LATENT_DIM in enumerate(GENconfig.LATENT_DIMENSIONS):
#         start_time = time.time()
#         X = Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_train")
#         TEST = Latent_dimension_dataset(str(LATENT_DIM)+"/encoded_train")
#         dataset = []
#         testset = []
#         for i in range(len(X)):
#             dataset.append(X[i].view([LATENT_DIM]))
#         for i in range(len(TEST)):
#             testset.append(TEST[i].view([LATENT_DIM]))

#         dataset =  torch.tensor(np.array(dataset), dtype=torch.float)
#         testset =  torch.tensor(np.array(testset), dtype=torch.float) 

#         base_dist = dist.Normal(torch.zeros(LATENT_DIM), torch.ones(LATENT_DIM), validate_args=False)
#         spline_transform = T.spline_coupling(LATENT_DIM, count_bins=32)

#         flow_dist = dist.TransformedDistribution(base_dist, [spline_transform], validate_args=False)

#         steps =  EPOCHS
#         Savetrainloss= []
#         Savetestloss= []
#         optimizer = torch.optim.Adam(spline_transform.parameters(), lr=LR)
#         for step in range(steps+1):
#             optimizer.zero_grad()
#             loss = -flow_dist.log_prob(dataset).mean()
#             Savetrainloss.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             flow_dist.clear_cache()

#             if step % 100 == 0:
#                 print('step: {}, loss: {}'.format(step, loss.item()))
#                 testloss = (-flow_dist.log_prob(testset).mean()).item()
#                 print('testloss = {}'.format(testloss))
#                 Savetestloss.append(testloss)

#         X_flow = flow_dist.sample(torch.Size([4000,])).detach().numpy()

#         if not os.path.exists("NF"+str(LATENT_DIM)):
#             os.mkdir(str(LATENT_DIM))
#             # os.mkdir(str(LATENT_DIM)+"/data")
#         np.savetxt(str(LATENT_DIM)+"/NF_trainloss.txt", Savetrainloss)
#         np.savetxt(str(LATENT_DIM)+"/NF_testloss.txt", Savetestloss)
#         np.savetxt(str(LATENT_DIM)+"/NF_traintime.txt", np.array([time.time()-start_time]))
#         # for j, ele in enumerate(X_flow):
#         #     torch.save(ele, "NF"+str(LATENT_DIM)+"/data"+"/n_"+str(j)+".pt")
