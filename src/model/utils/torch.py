
import torch
import numpy as np
import random



def set_seed(seed):
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    



