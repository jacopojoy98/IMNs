from dataclasses import dataclass
from datetime import datetime
import os

DATASET = "TMD"
# DATASET = "NHTS"

## Run parameters
RESULTS_DIRECTORY   = "/home/jcolombini/Purpose/Labeler/Results/Generative_results"
LATENT_DIMENSIONS   = (10,20,30,40,50)
NUMBER_OF_CLASSES   = 10
IMG_SIZE            = 32
BATCH_SIZE          = 8
DATA_DIRECTORY      = "/home/jcolombini/Purpose/Labeler/Data"


RUN_NAME            = datetime.now().strftime('%Y-%m-%d-%H-%M')

## RAE parameters
RAE_EPOCHS          = 5000
RAE_LR              = 1e-4
RAE_ALPHA           = 3e-4
RAE_TEST_SPLIT      = 0.2
RAE_VAL_SPLIT       = 0.2
rae_model_file      = "model_rae.pt"

## WGAN parameters
WGAN_EPOCHS         = 5000
WGAN_LR             = 2e-4
N_CPU               = 8
INNER_LATENT_DIM    = 100
CHANNELS            = 1
N_CRITIC            = 5
CLIP_VALUE          = 0.01
SAMPLE_INTERVAL     = 400
wgan_model_file     = "model_wgan.pt"

## NF parameters
NF_EPOCHS           = 400
NF_LR               = 1e-4
HIDDEN_FLOW_DIM1    = 32
HIDDEN_FLOW_DIM2    = 32
NUM_FLOW_LAYESRS    = 8
NF_model_file       = "model_NF.pt"

## NF ete parameters
EPOCHS_ETE_NF = 10000
NUM_ETE_FLOW_LAYESRS = 8
LR_ETE_NF = 1e-5

## Wgan ete Parameters
EPOCHS_ETE_WGAN = 10000
LR_ETE_WGAN = 1e-4

@dataclass
class RunParameters:
    name: str = os.path.join(RESULTS_DIRECTORY, RUN_NAME)
    data_directory: str = os.path.join(DATA_DIRECTORY, DATASET)
    Batch_size: int = BATCH_SIZE
    Datset : str = DATASET
    Num_classes : int = NUMBER_OF_CLASSES
    Img_size : int = IMG_SIZE

    RAE: bool = 1
    Epochs_Rae: int = RAE_EPOCHS
    Learning_rate_Rae: tuple = RAE_LR
    Alpha: float = RAE_ALPHA

    WGAN: bool = 1
    Epochs_Wgan: int = WGAN_EPOCHS
    Learning_rate_Wgan: tuple = WGAN_LR

    NF: bool = 1
    Epochs_NF: int = NF_EPOCHS
    Learning_rate_NF: tuple = NF_LR

    End_to_end_Wgan:bool=1
    Epochs_ete_Wgan:int= EPOCHS_ETE_WGAN
    LR_ete_Wgan :float = LR_ETE_WGAN

    End_to_end_NF: bool =1
    Epochs_ete_NF:int = EPOCHS_ETE_NF
    LR_ete_NF :float = LR_ETE_NF

    clean_data: bool = False
