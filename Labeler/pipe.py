from Stratify import stratify
from MLP import gnn
from config import RunParameters
import os
from datetime import datetime

args = RunParameters()
stratify(args)
for dim in [40,50]:
    args.Dimension = str(dim)
    args.savedir = os.path.join("/home/jcolombini/Purpose/Labeler/Results/Labeler_results", datetime.now().strftime('%Y-%m-%d-%H-%M'))
    gnn(args)