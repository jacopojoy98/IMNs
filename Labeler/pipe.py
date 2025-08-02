from Stratify import stratify
from MLP import gnn
from config import RunParameters

args = RunParameters()

stratify(args)
gnn(args)