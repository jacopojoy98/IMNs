import torch
import numpy as np
import os
from Models.NF import Normalizing_Flow
from Models.rae import RAE
from Models.wgan import Generator
from torch.autograd import Variable
import config as config
from Evaluation.DataManipulation import discard_matrix

def clean_data(data):
    discard = [discard_matrix(datum) for datum in data]
    treshold = np.quantile(discard, 0.9)
    data[discard<treshold]
    return data

def Generate_Data_wgan(rae_model, wgan_model, size, clean):
    z = Variable(torch.FloatTensor(
                    np.random.normal(0, 1, (size, config.INNER_LATENT_DIM))
                    ))
    data = rae_model.decode(wgan_model(z).detach()).view((size,42,32))
    if clean:
        data = clean_data(data)
    return data

def Generate_Data_rae(rae_model, size, Latent_dimension, args):
    z = Variable(torch.FloatTensor(
                    np.random.normal(0, 1, (size, Latent_dimension))
                    ))
    data = rae_model.decode(z).view((size,42,32))
    if args.clean_data :
        data = clean_data(data)
    return data

def Generate_Data_NF(rae_model, NF_model, size, Latent_dimension, args):

    z = Variable(torch.FloatTensor(
                    np.random.normal(0, 1, (size, Latent_dimension))
                    ))
    data = rae_model.decode(NF_model.forward(z).detach()).view((size,42,32))
    if args.clean_data:
        data = clean_data(data)
    return data

def Syth(args, size=4000):

    LATENT_DIMs = config.LATENT_DIMENSIONS
    IMG_H = config.IMG_SIZE
    IMG_W = config.IMG_SIZE

    for LATENT_DIM in LATENT_DIMs:
        os.chdir(os.path.join(args.name,str(LATENT_DIM)))
        Decoder = RAE(LATENT_DIM, IMG_H, IMG_W)
        Decoder.load_state_dict(torch.load(config.rae_model_file,
                                            map_location=torch.device('cpu'), 
                                            weights_only=True)
                                )
        Decoder.eval()
        Wgan = Generator(LATENT_DIM, config.INNER_LATENT_DIM)
        Wgan.load_state_dict(torch.load(config.wgan_model_file, 
                                        map_location=torch.device('cpu'), 
                                        weights_only=True)
                            )
        Wgan.eval()
        NF = Normalizing_Flow(config.NUM_FLOW_LAYESRS, LATENT_DIM)
        NF.load_state_dict(torch.load(config.NF_model_file, 
                                        map_location=torch.device('cpu'), 
                                        weights_only=True)
                            )
        NF.eval()

        data1 = Generate_Data_rae(Decoder,size, LATENT_DIM, args)
        data2 = Generate_Data_wgan(Decoder, Wgan, size, args)
        data3 = Generate_Data_NF(Decoder, NF, size, LATENT_DIM, args)
        torch.save(data1, "data_rae.pt")
        torch.save(data2, "data_wgan.pt")
        torch.save(data3, "data_NF.pt")
    os.chdir(os.path.join(args.name, "ETE"))
    #ete_NF
    dim = config.IMG_SIZE*(config.IMG_SIZE+config.NUMBER_OF_CLASSES)
    NF = Normalizing_Flow(config.NUM_ETE_FLOW_LAYESRS, config.IMG_SIZE*(config.IMG_SIZE+config.NUMBER_OF_CLASSES))
    NF.load_state_dict(torch.load('NF_end_to_end.pt', 
                                        map_location=torch.device('cpu'), 
                                        weights_only=True))
    z = Variable(torch.FloatTensor(
                    np.random.normal(0, 1, (size, dim))))
    data = NF.forward(z).detach().view((size,args.Img_size+args.Num_classes,args.Img_size))
    if args.clean_data:
        data = clean_data(data)
    torch.save(data, "data_ete_NF.pt")
    # ete_Wgan
    Wgan = Generator(dim, config.INNER_LATENT_DIM)
    Wgan.load_state_dict(torch.load('Wgan_end_to_end.pt', 
                                map_location=torch.device('cpu'), 
                                weights_only=True)
                    )
    z = Variable(torch.FloatTensor(
                    np.random.normal(0, 1, (size, config.INNER_LATENT_DIM))
                    ))
    data = Wgan(z).detach().view((size,args.Img_size+args.Num_classes,args.Img_size))
    if args.clean_data:
        data = clean_data(data)
    torch.save(data, "data_ete_Wgan.pt")
    
if __name__ == "__main__":
    pass