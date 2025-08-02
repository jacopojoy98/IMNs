import json
import matplotlib.pyplot as plt
import torch 

def NHTS_purposes_to_TMD (NHTS_purpose):
    if NHTS_purpose == 1 or NHTS_purpose == 2:
        return 0
    if NHTS_purpose == 3 or NHTS_purpose == 4 or NHTS_purpose == 5:
        return 1
    if NHTS_purpose == 6:
        return 6
    if NHTS_purpose == 7 or NHTS_purpose == 9 or NHTS_purpose == 97:
        return 3
    if NHTS_purpose == 8 or NHTS_purpose == 14 or NHTS_purpose == 17 or NHTS_purpose == 19:
        return 2
    if NHTS_purpose == 10:
        return 7
    if NHTS_purpose == 11:
        return 8
    if NHTS_purpose == 12:
        return 10
    if NHTS_purpose == 13:
        return 5
    if NHTS_purpose == 15 or NHTS_purpose == 16:
        return 4
    if NHTS_purpose == 18:
        return 9
    else:
        return 0
    
def save_on_report(filename, name, data):
    with open(filename, "a") as file:
        json.dump({name : data}, file)

def plot_with_text(matrix_,name): 
    if matrix_.shape == torch.Size([1, 1, 32, 32]):
        matrix = matrix_[0][0]
    else:
        matrix = matrix_
    plt.imshow(matrix, cmap="gray")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i,j]:.2f}",  # Formattazione con due decimali
                     ha="center", va="center",size=4.5, color="white" if matrix[i, j] < torch.mean(matrix) else "black")
    plt.savefig("/home/jcolombini/Purpose/Labeler/DBG/IMG/"+name+".png")
    plt.close()
    # print("saved "+ name, end ="")
    # input()

def plots(matrix_,name): 
    if torch.is_tensor(matrix_):
        matrix_ = matrix_.detach().numpy()
    if matrix_.shape == torch.Size([1, 1, 32, 32]):
        matrix = matrix_[0][0]
    else:
        matrix = matrix_
    matrix = matrix.squeeze()
    plt.imshow(matrix, cmap="gray")
    
    plt.savefig("/home/jcolombini/Purpose/Labeler/DBG/IMG/"+name+".png")
    plt.close()
    # print("saved "+ name, end ="")
    # input()

def save_plots(matrix,save_path): 
    plt.imshow(matrix.cpu(), cmap="gray")
    plt.savefig(save_path+".png")
    plt.close()