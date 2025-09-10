from Stratify import stratify
from MLP import gnn
from Cross_Exam import folded_experiment
from config import RunParameters
import os
from datetime import datetime

args = RunParameters()
for N in [10,5,3,2]:
    args.name = "2025-08-07-11-23"
    args.Num_classes = N
    args.Num_labels = N
    args.savedir = datetime.now().strftime('%Y-%m-%d-%H-%M')
    stratify(args)
    gnn(args)
    folded_experiment(args.savedir)
