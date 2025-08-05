from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class RunParameters:
    name :str = "2025-08-01-12-25"     #The synthetic dataset name
    Dimension: str = str(50)           #The latent dimension of the synthetic dataset
    method: str = "wgan"               #The generative method used to create the synthetic dataset
    
    savedir: str = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    Num_classes: int = 6       #The number of classes for the foldinn of the synthetic dataset

    Epochs: int = 700
    LR: float = 1e-4
    Num_labels: int = 6
    percentages: tuple = (0, 20, 40, 60, 80, 100)