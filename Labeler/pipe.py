from Stratify import stratify
from MLP import gnn
from Cross_Exam import folded_experiment
from config import RunParameters
import os
from datetime import datetime

args = RunParameters()
for N in [2, 3, 6, 13]:
    args.name = "2025-08-01-12-25"
    args.Num_classes = N
    args.Num_labels = N
    args.savedir = datetime.now().strftime('%Y-%m-%d-%H-%M')
    stratify(args)
    gnn(args)
    folded_experiment(args.savedir)
