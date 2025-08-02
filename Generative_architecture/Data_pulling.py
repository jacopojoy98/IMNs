import pickle
import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Generative_architecture.GLOBAL_FUNCTIONS import NHTS_purposes_to_TMD

def fileread(filename):
    with open(filename , "rb") as f:
        matrix= pickle.load(f)
    return matrix

def unpack_matrix(purposes, matrix, unpack_method = "tuple"):
    purposes = F.one_hot(torch.tensor(purposes), num_classes = 13).transpose(1,0).type(torch.float32)
    purposes = F.pad(purposes,(0,32-len(purposes[0])))
                                        ### ATTENZIONE A QUESTO 32
    structure = F.pad(torch.tensor(matrix),(0,32-len(matrix[0]),0,32-len(matrix)))
    if unpack_method == "tuple": 
        out = (purposes , structure)
    elif unpack_method == "Tensor":
        out =  torch.cat((purposes , structure), dim=0)
    else:
        raise ValueError("Undefined unpack method")
    return out

def unpack_single_NHTS(username, unpack_method = "tuple"):
    NHTS_IMN = fileread(os.path.join(username, "m2.pkl"))
    NHTS_purposes = [NHTS_purposes_to_TMD(NHTS_purpose) for NHTS_purpose in [p for p in NHTS_IMN[0] if p>0]]
    ABS_FIX = [[abs(l) for l in c ]for c in NHTS_IMN[1:]]
    out = unpack_matrix(NHTS_purposes, ABS_FIX, unpack_method)    
    return out

def unpack_single_TMD(username, unpack_method = "tuple"):
    purposes = [int(p*12) for p in fileread(os.path.join(username, "purposes.pickle"))]
    omega = fileread(os.path.join(username, "omega.pickle"))
    out = unpack_matrix(purposes, omega, unpack_method)
    return out

def unpack_to_list(dirname, unpack_fn):
    list = []
    for user in os.listdir(dirname):
        IMN = unpack_fn(os.path.join(dirname,user))
        list.append(IMN)
    return list

def unpack_to_tensor(dirname, unpack_fn):
    list = []
    for user in os.listdir(dirname):
        IMN = unpack_fn(os.path.join(dirname,user), unpack_method = "Tensor")
        list.append(IMN)
    out = torch.stack(list)
    return out

if __name__ == "__main__":
    pass

###
# Dataset
###

class Extract_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.users = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transfrom = transform

    def __len__(self):
        return(len(self.users))

    def __get_tuple_item__(self, idx):
        IMN_path = os.path.join(self.root_dir, self.users[idx])
        if os.path.basename(os.path.normpath(self.root_dir)) == "TMD" :
            IMN = unpack_single_TMD(IMN_path)
        elif os.path.basename(os.path.normpath(self.root_dir)) == "TMD_test": 
            IMN = unpack_single_TMD(IMN_path)  
        elif os.path.basename(os.path.normpath(self.root_dir)) == "NHTS" :
            IMN = unpack_single_NHTS(IMN_path)
        else:
            raise ValueError("Undefined data directory")
        return IMN
    
    def __getitem__(self, idx):
        IMN_path = os.path.join(self.root_dir, self.users[idx])
        if os.path.basename(os.path.normpath(self.root_dir)) == "TMD" :
            IMN = unpack_single_TMD(IMN_path, unpack_method = "Tensor")
        elif os.path.basename(os.path.normpath(self.root_dir)) == "TMD_test": 
            IMN = unpack_single_TMD(IMN_path, unpack_method = "Tensor")  
        elif os.path.basename(os.path.normpath(self.root_dir)) == "NHTS" :
            IMN = unpack_single_NHTS(IMN_path, unpack_method = "Tensor")
        else:
            raise ValueError("Undefined data directory")
        return IMN

class Latent_dimension_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.file = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transfrom = transform

    def __len__(self):
        return(len(self.file))

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.file[idx])
        data = torch.load(data_path, weights_only=True)
        return data.squeeze()
    
if __name__ == "__main__":

        pass
    # a = torch.load("/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-19-09-28/testset.pt", weights_only=False)
    # torch.save(torch.stack([traindata for traindata in a]), "/home/jcolombini/Purpose/Labeler/Results/Generative_results/2025-02-17-EPOCHS(1000, 1500, 200)-LR(0.0002, 0.0003, 0.0001)-ALPHA0.0003NUM_CLSS13/testsetdbg.pt") 