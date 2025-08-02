from __future__ import print_function
import time
import os, sys
import torch
import shutil
from torch.utils.data import random_split
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as config
from GLOBAL_FUNCTIONS import save_on_report, save_plots
from Data_pulling import Extract_Dataset
NUMBER_OF_CLASSES = config.NUMBER_OF_CLASSES
TEST_SPLIT = config.RAE_TEST_SPLIT
VAL_SPLIT = config.RAE_VAL_SPLIT


class RAE(nn.Module):
    def __init__(self, LAT_DIM, IMG_H, IMG_W):
        super(RAE, self).__init__()
        self.img_h = IMG_H
        self.img_w = IMG_W
        self.encoder = nn.Sequential(
              nn.Linear(IMG_H*(IMG_W+NUMBER_OF_CLASSES), 400),
              nn.ReLU(),
              nn.Linear(400, 100),
              nn.ReLU(),
              nn.Linear(100, LAT_DIM)
        )

        self.decoder = nn.Sequential(
              nn.Linear(LAT_DIM, 100),
              nn.ReLU(),
              nn.Linear(100, 400),
              nn.ReLU(),
              nn.Linear(400, IMG_H*(IMG_W+NUMBER_OF_CLASSES))
        )

    def encode(self, x):
        x = x.view(-1, self.img_h*(self.img_w+NUMBER_OF_CLASSES))
        return self.encoder(x)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), self.reg(z)

    def reg(self, z):
        l2 = 0
        for p in self.decoder.parameters():
            l2 += p.pow(2).sum()
        return l2 + (0.5 * z.pow(2).sum())

def train(epoch, model, optimizer, train_loader, device, IMG_H, IMG_W, ALPHA):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        recon_batch, reg = model(data)
        loss = F.binary_cross_entropy(recon_batch, data.view(-1, IMG_H*(IMG_W+13)), reduction="sum") + ALPHA * reg
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch%20==0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))    
    return train_loss / len(train_loader.dataset)

def test(model, test_loader, device,IMG_H, IMG_W, ALPHA):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data
            recon_batch, reg = model(data)
            loss = F.binary_cross_entropy(recon_batch, data.view(-1, IMG_H*(IMG_W+13)), reduction="sum") + ALPHA * reg
            test_loss += loss.item()
    print('====> Test set loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))
    return test_loss / len(test_loader.dataset)

def save(Latent_Dimension, model, trainset, testset, train_loss, test_loss, start_time, n, device):
    # Creation of saving directory
    if not os.path.exists(str(Latent_Dimension)):
            os.mkdir(str(Latent_Dimension))
    os.chdir(str(Latent_Dimension))

    # Plotting example of reconstruction
    data_to_plot = torch.stack([testset[i] for i in range(n*n)]).unsqueeze(0)
    recon, reg = model(data_to_plot.float().to(device))
    plot = torch.hstack((recon.view(45*n,32*n),data_to_plot.view(45*n,32*n)))   
    save_plots(plot.detach(),"reconstruction")

    torch.save(model.state_dict(), config.rae_model_file)
    save_on_report("report.json",__name__+"_traintime", [time.time() - start_time])
    save_on_report("report.json",__name__+"_trainloss", train_loss)
    save_on_report("report.json",__name__+"_testloss", test_loss)
    os.makedirs("encoded_train",exist_ok=True)
    os.makedirs("encoded_test",exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(trainset):
            data = data.float().to(device)
            encoded = model.encode(data)
            torch.save(encoded, "encoded_train/"+str(i)+".pt")
        for i, data in enumerate(testset):
            data = data.float().to(device)
            encoded = model.encode(data)
            torch.save(encoded, "encoded_test/"+str(i)+".pt")
    os.chdir("..")

def train_raes(args,
        IMG_H = 32,
        IMG_W = 32,
        ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Extract_Dataset(root_dir = args.data_directory)
    dataset = [data.float().to(device) for data in dataset]
    # Split the dataset into train, test, and validation sets
    if args.Datset == "NHTS":
        VAL_PERCENT = VAL_SPLIT
        TEST_PERCENT =  TEST_SPLIT
        trainset, testset, valset = random_split(dataset,[1-TEST_PERCENT-VAL_PERCENT, TEST_PERCENT, VAL_PERCENT])
        torch.save(torch.stack([traindata for traindata in trainset]), "trainset.pt")
        torch.save(torch.stack([traindata for traindata in testset]), "testset.pt")
        if valset:
            torch.save(torch.stack([traindata for traindata in valset]), "valset.pt")
    if args.Datset == "TMD":
        trainset = dataset
        torch.save(torch.stack([traindata for traindata in dataset]), "trainset.pt")
        testset = Extract_Dataset(root_dir = args.data_directory+"_test")
        testset = [data.float().to(device) for data in testset]
        torch.save(torch.stack([traindata for traindata in testset]), "testset.pt")
    else:
        raise ValueError("Dataset not recognized")
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(args.Batch_size),
        shuffle=True)
    test_loader =  torch.utils.data.DataLoader(
        testset,
        batch_size=int(args.Batch_size),
        shuffle=True)
    
    # device = torch.device("cpu")
    for LATENT_DIM in config.LATENT_DIMENSIONS:
        start_time = time.time()
        train_loss = []
        test_loss = []
        model = RAE(LATENT_DIM, IMG_H, IMG_W).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.Learning_rate_Rae)
        for epoch in range(1, args.Epochs_Rae + 1):
            loss = train(epoch, model, optimizer, train_loader, device,IMG_H, IMG_W, args.Alpha)
            if epoch%10==0 :
                train_loss.append(loss)
            if epoch%40==0:
                loss = test(model, test_loader, device,IMG_H, IMG_W, args.Alpha)
                test_loss.append(loss)
        save(LATENT_DIM, model, trainset, testset, train_loss, test_loss, start_time, 3, device)

if __name__ == "__main__":
    pass