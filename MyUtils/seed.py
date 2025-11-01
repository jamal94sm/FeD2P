import os
import random
import numpy as np
import torch

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from transformers import set_seed as transformers_set_seed
except ImportError:
    transformers_set_seed = None

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if tf is not None:
        tf.random.set_seed(seed)

    if transformers_set_seed is not None:
        transformers_set_seed(seed)