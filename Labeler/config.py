from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class RunParameters:
    name :str = "2025-08-01-12-25"     #The synthetic dataset name
    Dimension: str = str(30)           #The latent dimension of the synthetic dataset
    method: str = "NF"                 #The generative method used to create the synthetic dataset
    
    savedir: str = os.path.join("/home/jcolombini/Purpose/Labeler/Results/Labeler_results", datetime.now().strftime('%Y-%m-%d-%H-%M'))
    
    Num_classes: int = 13       #The number of classes for the foldinn of the synthetic dataset

    Epochs: int = 700
    LR: float = 1e-4
    Num_labels: int = 13
    percentages: tuple = (0, 20, 40, 60, 80, 100)