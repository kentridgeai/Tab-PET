"""
Utility functions for the SAINT with Positional Encodings experiment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

def setup_gpu(gpu_id=None):
    """Setup GPU configuration"""
    if gpu_id is None:
        gpu_id = 1
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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

# SAINT Data Utilities
def data_split_saint(X, y, nan_mask, indices):
    """Split data for SAINT format"""
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape:
        raise ValueError('Shape of data not same as that of nan mask!')
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

def prepare_data_for_saint(X_df, categorical_feature_names, verbose=True):
    """Prepare data for SAINT format"""
    printer = VerbosePrinter(verbose)
    
    temp_df = X_df.copy()
    
    # Handle categorical columns by converting to object type first
    for col in temp_df.columns:
        if pd.api.types.is_categorical_dtype(temp_df[col]):
            temp_df[col] = temp_df[col].astype('object')
    
    # Fill missing values to create nan mask
    temp = temp_df.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    # Identify categorical and continuous columns
    if categorical_feature_names is not None:
        categorical_columns = [col for col in categorical_feature_names if col in X_df.columns]
    else:
        categorical_columns = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    cont_columns = list(set(X_df.columns.tolist()) - set(categorical_columns))
    
    # Process categorical columns
    cat_dims = []
    for col in categorical_columns:
        temp_df[col] = temp_df[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        temp_df[col] = l_enc.fit_transform(temp_df[col].values)
        cat_dims.append(len(l_enc.classes_))
    
    # Get indices for categorical and continuous columns
    cat_idxs = [i for i, col in enumerate(X_df.columns) if col in categorical_columns]
    con_idxs = [i for i, col in enumerate(X_df.columns) if col in cont_columns]
    
    return temp_df, cat_dims, cat_idxs, con_idxs, nan_mask

class DataSetCatCon(Dataset):
    """SAINT dataset class for categorical and continuous data"""
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        
        self.X1 = X[:, cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32) #numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64) #numerical columns
        
        if task == 'clf':
            self.y = Y['data']
        else:
            self.y = Y['data'].astype(np.float32)
            
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (np.concatenate((self.cls[idx], self.X1[idx])), 
                self.X2[idx], 
                self.y[idx], 
                np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), 
                self.X2_mask[idx])

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)