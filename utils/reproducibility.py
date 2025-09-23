# E:\gemini\app-analyse\utils\reproducibility.py

import numpy as np
import random
import os

def set_global_seed(seed):
    """
    Sets the global seed for reproducibility across various libraries.
    """
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # CHANGE: Ajout de commentaires pour d'autres bibliothèques populaires
        # Pour TensorFlow/Keras:
        # try:
        #     import tensorflow as tf
        #     tf.random.set_seed(seed)
        # except ImportError:
        #     pass

        # Pour PyTorch:
        # try:
        #     import torch
        #     torch.manual_seed(seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed(seed)
        #         torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        #         torch.backends.cudnn.deterministic = True
        #         torch.backends.cudnn.benchmark = False
        # except ImportError:
        #     pass
            
    else:
        print("Attention: Aucune seed fournie. La reproductibilité pourrait ne pas être garantie.")