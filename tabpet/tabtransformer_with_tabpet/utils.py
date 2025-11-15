"""
Utility functions for the FT-Transformer with Positional Encodings experiment
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

def setup_gpu(gpu_id=None):
    """Setup GPU configuration"""
    def show_gpu_info():
        if torch.cuda.is_available():
            print(f"Found {torch.cuda.device_count()} GPUs:")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("No GPU available")

    show_gpu_info()
    
    if gpu_id is None:
        gpu_id = 1
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    return device

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class VerbosePrinter:
    """A printer class that respects verbose settings"""
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def section(self, title):
        if self.verbose:
            print(f"\n{'='*60}")
            print(title)
            print('='*60)