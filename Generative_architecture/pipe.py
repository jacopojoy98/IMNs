import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Evaluation import Analisis as Analisis
from Evaluation import Plotting as Plotting
from Models.rae import train_raes
from Models.wgan import train_wgans, end_to_end_Wgan
from Models.NF import train_NF, end_to_end_NF
from Syntetic_data import Syth
from config import RunParameters, RESULTS_DIRECTORY

import shutil

def train_models(args):
    if not os.path.isdir(args.name):
        os.mkdir(args.name)
    os.chdir(args.name)
    shutil.copy("/home/jcolombini/Purpose/Labeler/Generative_architecture/config.py","config.py")

    if args.RAE:
        train_raes(args)
    if args.WGAN:
        train_wgans(args)
    if args.NF:
        train_NF(args)

    if args.End_to_end_NF:
        end_to_end_NF(args)
    if args.End_to_end_Wgan:
        end_to_end_Wgan(args)
    return None

args = RunParameters()
args.name = os.path.join(RESULTS_DIRECTORY, "2025-08-01-15-14")
args.Epochs_NF = 300
args.RAE = False
args.NF = False
train_models(args)
Syth(args, 4000)
Analisis.Evaluate(args.name)
Plotting.plot(args.name)
