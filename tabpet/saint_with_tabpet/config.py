"""
Configuration file for the SAINT with Tab-PET experiment
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

# SAINT configuration
SAINT_CONFIG = {
    'target_embedding_dim': 32,
    'transformer_depth': 1,
    'attention_heads': 4,
    'attention_dropout': 0.8,
    'ffn_dropout': 0.8,
    'cont_embeddings': 'MLP',
    'attentiontype': 'colrow',
    'final_mlp_style': 'sep'
}

# Training configuration
TRAINING_CONFIG = {
    'num_epochs': 500,
    'learning_rate': 1e-4,
    'batch_size': 256,
    'early_stopping': True,
    'patience': 50,
    'min_delta': 1e-6,
    'min_epochs': 50,
    'weight_decay': 1e-5
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    'seeds': [1, 2, 3, 4, 5],
    'pe_alphas': [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0],
    # 'pe_alphas': [0.0, 1.0], # for quick start
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
