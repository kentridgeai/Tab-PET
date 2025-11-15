"""
Configuration file for the FT-Transformer with Positional Encodings experiment
"""

# Dataset configuration
DEFAULT_THRESHOLDS = [0.01, 0.05, 0.1, 0.2]

# LiNGAM configuration
LINGAM_METHOD = 'direct'
LINGAM_RANDOM_STATE = 42

DAG_METHOD = 'spearman'  # Options: 'lingam', 'notears', 'pearson', 'spearman', 'chowliu'

# NOTEARS configuration
NOTEARS_LAMBDA1_VALUES = [0.01, 0.05, 0.1, 0.2]
NOTEARS_LOSS_TYPE = 'l2'
NOTEARS_MAX_ITER = 100
NOTEARS_H_TOL = 1e-8
NOTEARS_RHO_MAX = 1e+16
NOTEARS_W_THRESHOLD = 0.3

# LLPE configuration
TARGET_TOTAL_DIM = 192
K_FIRST_DEFAULT = 3
K_LAST_DEFAULT = 3

# FT-Transformer configuration
MODEL_CONFIG = {
    'n_layers': 3,
    'd_ffn_factor': 4/3,
    'n_heads': 8,
    'attention_dropout': 0.2,
    'ffn_dropout': 0.1,
    'residual_dropout': 0.0,
    'activation': 'reglu',
    'prenormalization': True,
    'initialization': 'kaiming',
}

# TabTransformer configuration
TABTRANSFORMER_CONFIG = {
    'dim': 32,
    'depth': 6,
    'heads': 8,
    'dim_head': 16,
    'attn_dropout': 0.1,
    'ff_dropout': 0.1,
    'mlp_hidden_mults': (4, 2),
    'mlp_act': None,
    'num_special_tokens': 2,
    'use_shared_categ_embed': True,
    'shared_categ_dim_divisor': 8.0,
    'num_residual_streams': 4
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 500,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'early_stopping': True,
    'patience': 50,
    'min_delta': 1e-6,
    'min_epochs': 50,
    'weight_decay': 1e-5
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    'seeds': [1, 2, 3, 4, 5],
    # 'pe_alphas': [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0],
    'pe_alphas': [0.0, 1.0],
    'baseline_type': 'alpha_zero',
    'experiment_type': 'real_pe',
    'pe_integration_type': 'concat_fixed'
}

# Random seeds
RANDOM_SEEDS = {
    'numpy': 42,
    'torch': 42,
    'split': 42
}

# GPU configuration
GPU_CONFIG = {
    'default_gpu_id': 1
}