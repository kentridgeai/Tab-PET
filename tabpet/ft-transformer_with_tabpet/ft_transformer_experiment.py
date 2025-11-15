"""
FT-Transformer experiment module with Tab-PET, corresponding to (d) in Figure 1 of the main paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
import warnings
import time
import copy
import json
import datetime
import os
from pathlib import Path

from dag_construction import load_dataset
from utils import VerbosePrinter, set_random_seeds

warnings.filterwarnings('ignore')

import math
import typing as ty
from torch import Tensor
    
# https://github.com/kalelpark/FT_TransFormer/blob/main/model/fttransformer.py    
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear] = None,
        value_compression: ty.Optional[nn.Linear] = None,
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


def get_activation_fn(activation: str):
    if activation == 'reglu':
        def reglu(x):
            x, gates = x.chunk(2, dim=-1)
            return x * F.relu(gates)
        return reglu
    elif activation == 'geglu':
        def geglu(x):
            x, gates = x.chunk(2, dim=-1)
            return x * F.gelu(gates)
        return geglu
    elif activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise ValueError(f'Unknown activation: {activation}')


def get_nonglu_activation_fn(activation: str):
    if activation.endswith('glu'):
        return F.relu  # Default for GLU variants
    elif activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    else:
        return F.relu


class FTTransformerWithPE(nn.Module):
    """
    Enhanced FT-Transformer with Tab-PET
    """
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        # Tokenizer
        token_bias: bool = True,
        # Transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str = 'reglu',
        prenormalization: bool = True,
        initialization: str = 'kaiming',
        # Linformer (not used, but kept for compatibility)
        kv_compression: ty.Optional[float] = None,
        kv_compression_sharing: ty.Optional[str] = None,
        d_out: int,
        # PE parameters
        positional_encodings: ty.Optional[np.ndarray] = None,
        pe_integration_type: str = 'concat_fixed',
        pe_alpha: float = 0.1,
        use_pe: bool = True
    ) -> None:
        super().__init__()
        
        # Store PE configuration
        self.use_pe = use_pe and positional_encodings is not None
        self.pe_integration_type = pe_integration_type
        self.pe_alpha = pe_alpha
        self.d_numerical = d_numerical
        self.categories = categories
        
        # Process and normalize PE
        if self.use_pe:
            self.positional_encodings = self._process_positional_encodings(positional_encodings)
        else:
            self.positional_encodings = None
        
        # Tokenizer
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        
        self.original_d_token = d_token
        
        # PE integration using fixed concatenate
        if self.use_pe:
            pe_dim = self.positional_encodings.shape[1]
            
            # Use original PE dimension directly without reduction
            self.pe_concat_dim = pe_dim
            
            # Calculate effective token dimension 
            candidate_effective_dim = self.original_d_token + pe_dim
            
            # Ensure divisibility by n_heads using padding if needed
            remainder = candidate_effective_dim % n_heads
            if remainder != 0:
                padding_needed = n_heads - remainder
                self.pe_padding_dim = padding_needed
            else:
                self.pe_padding_dim = 0
            
            self.effective_d_token = candidate_effective_dim + self.pe_padding_dim

        else:
            self.effective_d_token = d_token
        
        def make_normalization():
            return nn.LayerNorm(self.effective_d_token)

        d_hidden = int(self.effective_d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict({
                'attention': MultiheadAttention(
                    self.effective_d_token, n_heads, attention_dropout, initialization
                ),
                'linear0': nn.Linear(
                    self.effective_d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                ),
                'linear1': nn.Linear(d_hidden, self.effective_d_token),
                'norm1': make_normalization(),
            })
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        
        # Output head (use effective_d_token)
        self.head = nn.Linear(self.effective_d_token, d_out)

    def _process_positional_encodings(self, pe_values):
        pe_array = np.array(pe_values, dtype=np.float32)
        pe_clipped = np.clip(pe_array, -10, 10)  # prevent extreme values
        return torch.tensor(pe_clipped, dtype=torch.float32)
    
    def _get_feature_indices(self, n_tokens):
        """Get the indices of feature tokens (excluding [CLS])"""
        n_numerical = self.d_numerical if self.d_numerical > 0 else 0
        n_categorical = len(self.categories) if self.categories else 0
        n_feature_tokens = n_numerical + n_categorical
        
        feature_start_idx = 1  # Skip [CLS] token
        feature_end_idx = min(feature_start_idx + n_feature_tokens, n_tokens)
        
        return feature_start_idx, feature_end_idx, n_feature_tokens

    def _integrate_positional_encodings(self, x, pe_tensor):
        """Integrate positional encodings with tokenized features using concat_fixed method"""
        batch_size, n_tokens, d_token = x.shape
        feature_start_idx, feature_end_idx, n_feature_tokens = self._get_feature_indices(n_tokens)
        
        # Ensure PE tensor matches feature count
        if pe_tensor.shape[0] != n_feature_tokens:
            if pe_tensor.shape[0] > n_feature_tokens:
                pe_tensor = pe_tensor[:n_feature_tokens]
            else:
                padding_size = n_feature_tokens - pe_tensor.shape[0]
                padding = torch.zeros(padding_size, pe_tensor.shape[1], 
                                    device=pe_tensor.device, dtype=pe_tensor.dtype)
                pe_tensor = torch.cat([pe_tensor, padding], dim=0)
        
        # Fixed COconcat method - NO LEARNABLE parameters
        
        # Step 1: Extract different types of tokens  
        cls_tokens = x[:, :feature_start_idx]  # [batch, 1, d_token]
        feature_tokens = x[:, feature_start_idx:feature_end_idx]  # [batch, n_features, d_token]
        remaining_tokens = x[:, feature_end_idx:] if feature_end_idx < n_tokens else None
        
        # Step 2: Apply PE scaling and expand
        pe_scaled = self.pe_alpha * pe_tensor  # [n_features, pe_dim] - direct scaling only
        pe_expanded = pe_scaled.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_features, pe_dim]
        
        # Step 3: Add padding to PE if needed for dimension alignment
        if self.pe_padding_dim > 0:
            pe_padding = torch.zeros(batch_size, pe_expanded.shape[1], self.pe_padding_dim, 
                                   device=x.device, dtype=x.dtype)
            pe_expanded = torch.cat([pe_expanded, pe_padding], dim=-1)  # [batch, n_features, pe_dim + padding]
        
        # Step 4: Concatenate original feature tokens with PE (NO projection, completely fixed)
        enhanced_feature_tokens = torch.cat([feature_tokens, pe_expanded], dim=-1)  # [batch, n_features, effective_d_token]
                  
        # Step 5: Process non-feature tokens - pad them to match effective_d_token
        padding_for_non_features = self.pe_concat_dim + self.pe_padding_dim
        
        cls_padding = torch.zeros(batch_size, cls_tokens.shape[1], padding_for_non_features, 
                                 device=x.device, dtype=x.dtype)
        padded_cls_tokens = torch.cat([cls_tokens, cls_padding], dim=-1)  # [batch, 1, effective_d_token]
        
        # Step 6: Reconstruct the full sequence
        token_list = [padded_cls_tokens, enhanced_feature_tokens]
        
        if remaining_tokens is not None:
            remaining_padding = torch.zeros(batch_size, remaining_tokens.shape[1], padding_for_non_features, 
                                           device=x.device, dtype=x.dtype)
            padded_remaining_tokens = torch.cat([remaining_tokens, remaining_padding], dim=-1)
            token_list.append(padded_remaining_tokens)
        
        x = torch.cat(token_list, dim=1)  # [batch, n_tokens, effective_d_token]
        
        return x

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_attn: bool = False) -> Tensor:

        x = self.tokenizer(x_num, x_cat)
        
        if self.use_pe:
            pe_tensor = self.positional_encodings.to(x.device)
            x = self._integrate_positional_encodings(x, pe_tensor)
        
        attentions = []
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            attn_input = x_residual[:, :1] if is_last_layer else x_residual
            x_residual = layer['attention'](attn_input, x_residual)
            
            if return_attn:
                attentions.append(None)
                
            if is_last_layer:
                x = x[:, :x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        
        if not return_attn:
            return x
        return x, attentions


def prepare_data_for_ft_transformer(X_df, categorical_feature_names, pe_values, verbose=True):

    printer = VerbosePrinter(verbose)
    
    categorical_data = []
    numerical_data = []
    categorical_categories = [] 
    
    categorical_feature_info = []
    numerical_feature_info = []
    
    for col in X_df.columns:
        is_categorical = (
            (categorical_feature_names is not None and col in categorical_feature_names) or 
            pd.api.types.is_categorical_dtype(X_df[col]) or 
            X_df[col].dtype == 'object'
        )
        
        if is_categorical:
            col_data = X_df[col]
            
            if pd.api.types.is_categorical_dtype(col_data):
                col_data = col_data.astype(str)
            
            unique_values = col_data.unique()
            n_categories = len(unique_values)
            
            label_encoder = LabelEncoder()
            encoded_values = label_encoder.fit_transform(col_data)
            
            categorical_data.append(encoded_values)
            categorical_categories.append(n_categories)
            categorical_feature_info.append({
                'name': col,
                'encoder': label_encoder,
                'n_categories': n_categories,
                'original_values': unique_values
            })
            
        else:
            numerical_data.append(X_df[col].values)
            numerical_feature_info.append({'name': col})
    
    if categorical_data:
        X_categorical = np.column_stack(categorical_data).astype(np.int64)
        categories = categorical_categories
    else:
        X_categorical = np.array([]).reshape(len(X_df), 0).astype(np.int64)
        categories = []
    
    if numerical_data:
        X_numerical = np.column_stack(numerical_data).astype(np.float32)
    else:
        X_numerical = np.array([]).reshape(len(X_df), 0).astype(np.float32)
    
    return X_categorical, X_numerical, categories, categorical_feature_info, numerical_feature_info

def get_openml_categorical_features(dataset_name, dataset_version):
    """
    Get categorical features from OpenML
    """
    
    try:
        dataset_variations = [
            (dataset_name, dataset_version),
            (dataset_name.replace('_', '-'), dataset_version),
            (dataset_name.replace('-', '_'), dataset_version)
        ]
        
        dataset = None
        for name_variant, version in dataset_variations:
            try:
                dataset = fetch_openml(name=name_variant, version=version, as_frame=True, parser='auto')
                break
            except:
                continue
        
        if dataset is None:
            return []
            
        X = dataset.data
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                unique_vals = X[col].nunique()
                if unique_vals <= 10 and unique_vals < len(X) * 0.05:  
                    categorical_features.append(col)
        
        return categorical_features
        
    except Exception as e:
        return []
    
def aggregate_positional_encodings_average(pe_feature_names, pe_values, current_features, 
                                          dataset_name, dataset_version, verbose=True):
    """
    Aggregate PEs using averaging strategy for categorical features
    """
    printer = VerbosePrinter(verbose)
    
    categorical_features = get_openml_categorical_features(dataset_name, dataset_version)
    categorical_features_set = set(categorical_features)
    
    pe_name_to_idx = {name: idx for idx, name in enumerate(pe_feature_names)}
    
    aggregated_pe_list = []
    feature_pe_mapping = []
    
    for current_feature in current_features:
        if current_feature in categorical_features_set:

            related_pe_indices = []
            related_pe_names = []
            
            target_prefix = f"{current_feature}_"
            for pe_name in pe_feature_names:
                if pe_name.startswith(target_prefix):
                    related_pe_indices.append(pe_name_to_idx[pe_name])
                    related_pe_names.append(pe_name)
            
            if related_pe_indices:
                related_pe_values = pe_values[related_pe_indices]
                aggregated_pe = related_pe_values.mean(axis=0)
                feature_pe_mapping.append(f"{current_feature} <- {related_pe_names}")
            else:
                aggregated_pe = np.zeros(pe_values.shape[1])
                feature_pe_mapping.append(f"{current_feature} <- [NO_PE_FOUND]")
        
        else:
            # For numerical features: look for exact match
            if current_feature in pe_name_to_idx:
                pe_idx = pe_name_to_idx[current_feature]
                aggregated_pe = pe_values[pe_idx]
                feature_pe_mapping.append(f"{current_feature} <- [{current_feature}]")
            else:
                aggregated_pe = np.zeros(pe_values.shape[1])
                feature_pe_mapping.append(f"{current_feature} <- [NO_PE_FOUND]")
        
        aggregated_pe_list.append(aggregated_pe)
    
    aggregated_pe_array = np.array(aggregated_pe_list)
    
    return aggregated_pe_array, feature_pe_mapping


def detect_task_type_and_prepare_target(y, verbose=True):
    """
    classification or regression task
    """
    printer = VerbosePrinter(verbose)
    
    if hasattr(y, 'dtype'):
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            task_type = 'classification'
        else:
            unique_vals = len(np.unique(y))
            if unique_vals <= 20 and np.allclose(y, np.round(y)):
                task_type = 'classification'
            else:
                task_type = 'regression'
    else:
        try:
            y_numeric = pd.to_numeric(y, errors='coerce')
            if y_numeric.isnull().any():
                task_type = 'classification'
            else:
                unique_vals = len(pd.Series(y).unique())
                if unique_vals <= 20:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
        except:
            task_type = 'classification'
    
    if task_type == 'classification':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        
        return y_encoded, task_type, num_classes, label_encoder
    else:
        y_numeric = pd.to_numeric(y, errors='coerce')
        return y_numeric.values, task_type, 1, None


def train_model(model, train_loader, val_loader, num_epochs=500, learning_rate=1e-4, task_type='auto', 
                early_stopping=False, patience=50, min_delta=1e-6, min_epochs=50, verbose_freq=20, verbose=True):
    """
    Train the FT-Transformer model with Tab-PET
    """
    printer = VerbosePrinter(verbose)
    
    device = next(model.parameters()).device
    
    if task_type == 'auto':
        if hasattr(model, 'head'):
            output_dim = model.head.out_features
        else:
            output_dim = model.d_out
        
        if output_dim == 1:
            task_type = 'regression'
        else:
            task_type = 'classification'
    
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:  # classification
        # Calculate class weights for balanced cross entropy
        unique_labels, counts = torch.unique(train_loader.dataset.tensors[2], return_counts=True)
        num_classes = len(unique_labels)
        
        weights = []
        for i in range(num_classes):
            class_freq = (train_loader.dataset.tensors[2] == i).float().mean()
            weights.append(1.0 / class_freq)
        
        weights = torch.tensor(weights, dtype=torch.float32) / num_classes
        weights = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    val_balanced_accuracies = []  # Store balanced accuracy history
    
    if task_type == 'classification':
        best_metric = -float('inf')  # Balanced accuracy
        metric_name = "Balanced Accuracy"
        if early_stopping:
            patience = max(30, patience) 
            min_delta = max(1e-4, min_delta)
    else:
        best_metric = float('inf')   # Validation loss
        metric_name = "Validation Loss"
    
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss = 0.0
        
        # Training
        for batch_idx, (x_cat, x_num, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            if x_cat.shape[1] == 0:
                x_cat = None
            if x_num.shape[1] == 0:
                x_num = None
            
            outputs = model(x_num, x_cat)
            
            if task_type == 'classification':
                if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                    y_batch = y_batch.squeeze(-1)
                y_batch = y_batch.long()
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        current_metric = 0.0
        
        with torch.no_grad():
            if task_type == 'classification':
                # Calculate balanced accuracy
                test_labels = val_loader.dataset.tensors[2]
                unique_labels = torch.unique(test_labels)
                num_classes = len(unique_labels)
                
                # Calculate class weights
                class_weights = []
                for i in range(num_classes):
                    class_count = (test_labels == i).float().sum()
                    class_weights.append(1.0 / (class_count * num_classes))
                
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
                balanced_accuracy = torch.tensor(0.0).to(device)
                
                for x_cat, x_num, y_batch in val_loader:
                    if x_cat.shape[1] == 0:
                        x_cat = None
                    if x_num.shape[1] == 0:
                        x_num = None
                    
                    if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                        y_batch = y_batch.squeeze(-1)
                    
                    outputs = model(x_num, x_cat)
                    predicted = torch.argmax(outputs, dim=1)
                    
                    # Validation loss
                    val_loss += criterion(outputs, y_batch.long()).item()
                    
                    # Balanced accuracy
                    balanced_accuracy += torch.sum((predicted == y_batch).float() * class_weights[y_batch])
                
                current_metric = balanced_accuracy.cpu().item()  # Use balanced accuracy as evaluation metric
                
            else:  # regression
                for x_cat, x_num, y_batch in val_loader:
                    if x_cat.shape[1] == 0:
                        x_cat = None
                    if x_num.shape[1] == 0:
                        x_num = None
                    
                    outputs = model(x_num, x_cat)
                    val_loss += criterion(outputs, y_batch).item()
                
                current_metric = val_loss / len(val_loader)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if task_type == 'classification':
            val_balanced_accuracies.append(current_metric)
        
        improved = False
        if task_type == 'classification':
            if current_metric > best_metric:
                improved = True
        else:
            if current_metric < best_metric:
                improved = True
        
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
        if early_stopping and epoch >= min_epochs:
            if task_type == 'classification':
                if current_metric > best_metric - min_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if current_metric < best_metric + min_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                early_stopped = True
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        model.eval()
        verified_metric = 0.0
        with torch.no_grad():
            if task_type == 'classification':
                test_labels = val_loader.dataset.tensors[2]
                unique_labels = torch.unique(test_labels)
                num_classes = len(unique_labels)
                
                class_weights = []
                for i in range(num_classes):
                    class_count = (test_labels == i).float().sum()
                    class_weights.append(1.0 / (class_count * num_classes))
                
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
                balanced_accuracy = torch.tensor(0.0).to(device)
                
                for x_cat, x_num, y_batch in val_loader:
                    if x_cat.shape[1] == 0:
                        x_cat = None
                    if x_num.shape[1] == 0:
                        x_num = None
                    
                    if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                        y_batch = y_batch.squeeze(-1)
                    
                    outputs = model(x_num, x_cat)
                    predicted = torch.argmax(outputs, dim=1)
                    balanced_accuracy += torch.sum((predicted == y_batch).float() * class_weights[y_batch])
                
                verified_metric = balanced_accuracy.cpu().item()
            else:
                verified_val_loss = 0.0
                for x_cat, x_num, y_batch in val_loader:
                    if x_cat.shape[1] == 0:
                        x_cat = None
                    if x_num.shape[1] == 0:
                        x_num = None
                    
                    outputs = model(x_num, x_cat)
                    verified_val_loss += criterion(outputs, y_batch).item()
                
                verified_metric = verified_val_loss / len(val_loader)
        
        final_best_metric = verified_metric
    else:
        if task_type == 'classification':
            final_best_metric = val_balanced_accuracies[-1] if val_balanced_accuracies else 0.0
        else:
            final_best_metric = val_losses[-1]
    
    return train_losses, val_losses, final_best_metric, best_epoch


def evaluate_model(model, test_loader, task_type='auto', verbose=True):

    printer = VerbosePrinter(verbose)
    
    if task_type == 'auto':
        if hasattr(model, 'head'):
            output_dim = model.head.out_features
        else:
            output_dim = model.d_out
        
        if output_dim == 1:
            task_type = 'regression'
        else:
            task_type = 'classification'
    
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x_cat, x_num, y_batch in test_loader:
            if x_cat.shape[1] == 0:
                x_cat = None
            if x_num.shape[1] == 0:
                x_num = None
            
            outputs = model(x_num, x_cat)
            
            if task_type == 'classification':
                predicted_classes = torch.argmax(outputs, dim=1)
                predictions.extend(predicted_classes.cpu().numpy())
                
                if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                    y_batch = y_batch.squeeze(-1)
                actuals.extend(y_batch.cpu().numpy())
            else:
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    if task_type == 'regression':
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        return rmse, r2, predictions, actuals
    else:  # classification
        # Calculate balanced accuracy
        model.eval()
        balanced_accuracy = torch.tensor(0.0).to(device)
        
        # Calculate Class Weights
        test_labels = test_loader.dataset.tensors[2]
        unique_labels = torch.unique(test_labels)
        num_classes = len(unique_labels)
        
        weights = []
        for i in range(num_classes):
            class_count = (test_labels == i).float().sum()
            weights.append(1.0 / (class_count * num_classes))
        
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        # Balanced accuracy calculation
        with torch.no_grad():
            for x_cat, x_num, y_batch in test_loader:
                if x_cat.shape[1] == 0:
                    x_cat = None
                if x_num.shape[1] == 0:
                    x_num = None
                
                if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                    y_batch = y_batch.squeeze(-1)
                
                outputs = model(x_num, x_cat)
                predicted = torch.argmax(outputs, dim=1)
                
                # Calculate weighted accuracy
                balanced_accuracy += torch.sum((predicted == y_batch).float() * weights[y_batch])
        
        balanced_accuracy = balanced_accuracy.cpu().item()
        
        f1 = f1_score(actuals, predictions, average='weighted')
        
        return balanced_accuracy, f1, predictions, actuals


def run_multi_seed_experiment(model_creator_func, train_loader, val_loader, test_loader, task_info, 
                             seeds=[1,2,3,4,5], num_epochs=500, learning_rate=1e-4,
                             early_stopping=False, patience=50, min_delta=1e-6, min_epochs=50, verbose=True):
    """
    Run experiment with multiple random seeds for robust results
    """
    printer = VerbosePrinter(verbose)
    
    all_results = []
    task_type = task_info['task_type']
    
    total_epochs_saved = 0
    
    for seed in seeds:
        device = next(iter(train_loader.dataset.tensors)).device
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        set_random_seeds(seed)
        
        model = model_creator_func()
        
        # Train model with early stopping using validation set
        train_losses, val_losses, best_val_loss_verified, best_epoch = train_model(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, learning_rate=learning_rate, task_type=task_type,
            early_stopping=early_stopping, patience=patience, min_delta=min_delta, 
            min_epochs=min_epochs, verbose_freq=20, verbose=verbose
        )
        
        epochs_trained = len(train_losses)
        epochs_saved = num_epochs - epochs_trained
        total_epochs_saved += epochs_saved
        
        # Evaluate model on test set
        metric1, metric2, preds, actuals = evaluate_model(model, test_loader, task_type=task_type, verbose=verbose)
        
        result = {
            'seed': seed,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': best_val_loss_verified,
            'best_val_loss': best_val_loss_verified,
            'best_epoch': best_epoch,
            'epochs_trained': epochs_trained,
            'epochs_saved': epochs_saved,
            'early_stopped': epochs_trained < num_epochs
        }
        
        if task_type == 'regression':
            result['rmse'] = metric1
            result['r2'] = metric2
        else:
            result['accuracy'] = metric1
            result['f1'] = metric2
        
        all_results.append(result)
        
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return all_results


def summarize_multi_seed_results(results, experiment_name="", task_info=None, verbose=True):

    printer = VerbosePrinter(verbose)
    
    if task_info is None:
        if 'rmse' in results[0]:
            task_type = 'regression'
        else:
            task_type = 'classification'
    else:
        task_type = task_info['task_type']
    
    avg_epochs = np.mean([r['epochs_trained'] for r in results])
    early_stopped_count = sum([r['early_stopped'] for r in results])
    
    if task_type == 'regression':
        rmses = [r['rmse'] for r in results]
        r2s = [r['r2'] for r in results]
        
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        r2_mean, r2_std = np.mean(r2s), np.std(r2s)
        
        return {
            'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
            'r2_mean': r2_mean, 'r2_std': r2_std,
            'all_results': results,
            'task_type': task_type,
            'avg_epochs': avg_epochs,
            'early_stopped_count': early_stopped_count
        }
    else:
        accuracies = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]
        
        acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        
        return {
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'f1_mean': f1_mean, 'f1_std': f1_std,
            'all_results': results,
            'task_type': task_type,
            'avg_epochs': avg_epochs,
            'early_stopped_count': early_stopped_count
        }


def run_ft_transformer_experiment(dag_results, llpe_results, dataset_name, dataset_version, 
                                config, device, verbose=True):
    """Main function to run FT-Transformer experiment with Tab-PET"""
    printer = VerbosePrinter(verbose)
    
    data = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)
    
    X_df_features = data.data
    y = data.target
    
    y_processed, task_type, num_classes, label_encoder = detect_task_type_and_prepare_target(y, verbose)
    
    categorical_feature_names = None
    if 'categorical_cols' in dag_results:
        categorical_feature_names = dag_results['categorical_cols']

    X_categorical, X_numerical, categories, categorical_info, numerical_info = prepare_data_for_ft_transformer(
        X_df_features, categorical_feature_names, llpe_results['positional_encodings'], verbose
    )
    
    pe_feature_names = llpe_results['encoding_df'].index.tolist()
    pe_values = llpe_results['positional_encodings']
    ft_feature_order = list(X_df_features.columns)
    
    pe_values_aligned, pe_mapping = aggregate_positional_encodings_average(
        pe_feature_names, pe_values, ft_feature_order, dataset_name, dataset_version, verbose
    )
    
    # Calculate dynamic d_token based on PE dimensions to ensure the total dimensions are fixed
    pe_dim = pe_values_aligned.shape[1] if pe_values_aligned is not None else 0
    target_total_dim = config['target_total_dim']
    dynamic_d_token = target_total_dim - pe_dim
    
    if dynamic_d_token <= 0:
        raise ValueError(f"PE dimensions ({pe_dim}) too large! Must be < {target_total_dim}")

    if target_total_dim % config['model_config']['n_heads'] != 0:
        raise ValueError(f"Target total dimension ({target_total_dim}) must be divisible by n_heads ({config['model_config']['n_heads']})")

    model_config = config['model_config'].copy()
    model_config.update({
        'd_numerical': X_numerical.shape[1],
        'categories': categories if categories else None,
        'd_token': dynamic_d_token,
        'd_out': num_classes if task_type == 'classification' else 1,
    })

    stratify_param = y_processed if task_type == 'classification' else None

    X_cat_temp, X_cat_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
        X_categorical, X_numerical, y_processed, test_size=0.2, random_state=42,
        stratify=stratify_param
    )

    if task_type == 'classification':
        stratify_temp = y_temp
    else:
        stratify_temp = None

    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_cat_temp, X_num_temp, y_temp, test_size=0.25, random_state=42,
        stratify=stratify_temp
    )

    if X_numerical.shape[1] > 0:
        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(X_num_train)
        X_num_val = scaler.transform(X_num_val)
        X_num_test = scaler.transform(X_num_test)

    # Convert to tensors
    X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.long).to(device)
    X_cat_val_tensor = torch.tensor(X_cat_val, dtype=torch.long).to(device)
    X_cat_test_tensor = torch.tensor(X_cat_test, dtype=torch.long).to(device)
    X_num_train_tensor = torch.tensor(X_num_train, dtype=torch.float32).to(device)
    X_num_val_tensor = torch.tensor(X_num_val, dtype=torch.float32).to(device)
    X_num_test_tensor = torch.tensor(X_num_test, dtype=torch.float32).to(device)

    # Prepare target tensors based on task type
    if task_type == 'classification':
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    else:
        y_train_array = np.array(y_train).reshape(-1, 1).astype(np.float32)
        y_val_array = np.array(y_val).reshape(-1, 1).astype(np.float32)
        y_test_array = np.array(y_test).reshape(-1, 1).astype(np.float32)
        y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32).to(device)

    # Create data loaders
    batch_size = config['training_config']['batch_size']
    
    train_dataset = TensorDataset(X_cat_train_tensor, X_num_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_cat_val_tensor, X_num_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_cat_test_tensor, X_num_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    task_info = {
        'task_type': task_type,
        'num_classes': num_classes,
        'label_encoder': label_encoder
    }
    
    # Run experiment
    pe_alphas = config['experiment_config']['pe_alphas']
    baseline_type = config['experiment_config']['baseline_type']  # alpha_zero as the baseline without PE
    experiment_type = config['experiment_config']['experiment_type']  # real_pe as our Tab-PET
    
    multi_seed_alpha_results = {}
    
    for alpha in pe_alphas:
        def create_alpha_model():
            if alpha == 0.0:
                # alpha_zero: baseline
                return FTTransformerWithPE(
                    **model_config,  # d_token = dynamic_d_token
                    positional_encodings=pe_values_aligned,  # Real PE matrix (but alpha=0)
                    use_pe=True,
                    pe_integration_type=config['experiment_config']['pe_integration_type'],
                    pe_alpha=0.0
                ).to(device)
            else:
                # real_pe: Tab-PET
                return FTTransformerWithPE(
                    **model_config,  # d_token = dynamic_d_token
                    positional_encodings=pe_values_aligned,  # Real PE matrix
                    use_pe=True,
                    pe_integration_type=config['experiment_config']['pe_integration_type'],
                    pe_alpha=alpha
                ).to(device)
        
        alpha_results = run_multi_seed_experiment(
            create_alpha_model, train_loader, val_loader, test_loader,
            task_info,
            seeds=config['experiment_config']['seeds'], 
            num_epochs=config['training_config']['num_epochs'],
            early_stopping=config['training_config']['early_stopping'],
            verbose=verbose
        )
        
        if alpha == 0.0:
            model_description = f"{baseline_type} baseline"
        else:
            model_description = f"{experiment_type} (alpha={alpha})"
        
        alpha_summary = summarize_multi_seed_results(alpha_results, model_description, task_info, verbose)
        
        multi_seed_alpha_results[alpha] = alpha_summary
        
        if task_info['task_type'] == 'regression':
            printer.print(f"Alpha {alpha}: RMSE={alpha_summary['rmse_mean']:.4f}±{alpha_summary['rmse_std']:.4f}, R²={alpha_summary['r2_mean']:.4f}±{alpha_summary['r2_std']:.4f}")
        else:
            printer.print(f"Alpha {alpha}: Accuracy={alpha_summary['acc_mean']:.4f}±{alpha_summary['acc_std']:.4f}, F1={alpha_summary['f1_mean']:.4f}±{alpha_summary['f1_std']:.4f}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Extract baseline results (alpha = 0.0) for comparison
    baseline_results = multi_seed_alpha_results[0.0]

    if task_info['task_type'] == 'regression':
        baseline_metric1 = baseline_results['rmse_mean']
        baseline_metric2 = baseline_results['r2_mean']
        metric1_name, metric2_name = 'RMSE', 'R²'
        metric1_better_direction = 'lower'
        metric2_better_direction = 'higher'
    else:
        baseline_metric1 = baseline_results['acc_mean']
        baseline_metric2 = baseline_results['f1_mean'] 
        metric1_name, metric2_name = 'Accuracy', 'F1'
        metric1_better_direction = 'higher'
        metric2_better_direction = 'higher'

    # Calculate improvements for all PE alpha values
    for alpha in pe_alphas:
        if alpha > 0.0:  # Tab-PET
            result = multi_seed_alpha_results[alpha]
            
            if task_info['task_type'] == 'regression':
                metric1_improvement = ((baseline_metric1 - result['rmse_mean']) / baseline_metric1) * 100  # RMSE improvement
                metric2_improvement = ((result['r2_mean'] - baseline_metric2) / baseline_metric2) * 100  # R² improvement
            else:
                metric1_improvement = ((result['acc_mean'] - baseline_metric1) / baseline_metric1) * 100  # Accuracy improvement
                metric2_improvement = ((result['f1_mean'] - baseline_metric2) / baseline_metric2) * 100  # F1 improvement
            
            result['metric1_improvement'] = metric1_improvement
            result['metric2_improvement'] = metric2_improvement

    # Find best alpha using greedy search
    pe_alphas_only = [a for a in pe_alphas if a > 0.0]

    if task_info['task_type'] == 'regression':
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['r2_mean'])
    else:
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['acc_mean'])

    best_result = multi_seed_alpha_results[best_alpha]
    
    printer.print(f"\n" + "="*60)
    printer.print(f"TAB-PET FINAL RESULTS")
    printer.print("="*60)
    printer.print(f"Best Alpha: {best_alpha}")
    if task_info['task_type'] == 'regression':
        printer.print(f"Best Performance: RMSE={best_result['rmse_mean']:.4f}±{best_result['rmse_std']:.4f}, R²={best_result['r2_mean']:.4f}±{best_result['r2_std']:.4f}")
        baseline_rmse = baseline_results['rmse_mean']
        improvement = ((baseline_rmse - best_result['rmse_mean']) / baseline_rmse) * 100
        printer.print(f"RMSE Improvement: {improvement:+.2f}%")
    else:
        printer.print(f"Best Performance: Accuracy={best_result['acc_mean']:.4f}±{best_result['acc_std']:.4f}, F1={best_result['f1_mean']:.4f}±{best_result['f1_std']:.4f}")
        baseline_acc = baseline_results['acc_mean']
        improvement = ((best_result['acc_mean'] - baseline_acc) / baseline_acc) * 100
        printer.print(f"Accuracy Improvement: {improvement:+.2f}%")

    save_experiment_results(multi_seed_alpha_results, best_alpha, best_result, task_info, 
                          dataset_name, dataset_version, config, verbose)
    
    return {
        'multi_seed_alpha_results': multi_seed_alpha_results,
        'best_alpha': best_alpha,
        'best_result': best_result,
        'baseline_results': baseline_results,
        'task_info': task_info,
        'model_config': model_config
    }


def save_experiment_results(multi_seed_alpha_results, best_alpha, best_result, task_info, 
                          dataset_name, dataset_version, config, verbose=True):
    printer = VerbosePrinter(verbose)
    
    auto_dataset_info = {
        'dataset_name': dataset_name,
        'dataset_version': dataset_version,
        'dataset_identifier': f'{dataset_name}_v{dataset_version}',
        'task_type': task_info['task_type'],
        'n_classes': task_info['num_classes'] if task_info['task_type'] == 'classification' else None
    }
    
    save_to_unified_csv(multi_seed_alpha_results, auto_dataset_info, task_info, verbose)


def save_to_unified_csv(experiment_results, dataset_info, task_info, verbose=True):
    printer = VerbosePrinter(verbose)
    
    dataset_id = dataset_info.get('dataset_identifier', 'unknown')
    task_type = task_info.get('task_type', 'unknown')
    
    if task_type == 'classification':
        unified_csv_path = "ft_transformer_classification_results.csv"
    elif task_type == 'regression':
        unified_csv_path = "ft_transformer_regression_results.csv"
    else:
        return False
    
    new_results = []
    
    if task_type == 'classification':
        # For classification, use balanced accuracy as the main metric
        if 0.0 in experiment_results:
            baseline_result = experiment_results[0.0]
            baseline_accuracy = baseline_result['acc_mean']
            
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                accuracy_result = f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}"
                
                if alpha == 0.0:
                    improvement_pct = "—"  # Baseline has no improvement
                else:
                    improvement = ((result['acc_mean'] - baseline_accuracy) / baseline_accuracy) * 100
                    improvement_pct = f"{improvement:+.2f}%"
                
                new_results.append({
                    'Dataset': dataset_id,
                    'Alpha': alpha,
                    'Accuracy_Result': accuracy_result,
                    'Accuracy_Improvement_Pct': improvement_pct
                })
        else:
            return False
            
    elif task_type == 'regression':
        # For regression, use RMSE as the main metric
        if 0.0 in experiment_results:
            baseline_result = experiment_results[0.0]
            baseline_rmse = baseline_result['rmse_mean']
            
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                rmse_result = f"{result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}"
                
                if alpha == 0.0:
                    improvement_pct = "—"  # Baseline has no improvement
                else:
                    improvement = ((baseline_rmse - result['rmse_mean']) / baseline_rmse) * 100
                    improvement_pct = f"{improvement:+.2f}%"
                
                new_results.append({
                    'Dataset': dataset_id,
                    'Alpha': alpha,
                    'RMSE_Result': rmse_result,
                    'RMSE_Improvement_Pct': improvement_pct
                })
        else:
            return False
    
    new_df = pd.DataFrame(new_results)
    
    # Load existing unified CSV if it exists
    if os.path.exists(unified_csv_path):
        try:
            existing_df = pd.read_csv(unified_csv_path)           
            existing_df = existing_df[existing_df['Dataset'] != dataset_id]           
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
        except Exception as e:
            combined_df = new_df
    else:
        combined_df = new_df
    
    combined_df = combined_df.sort_values(['Dataset', 'Alpha']).reset_index(drop=True)
    
    try:
        combined_df.to_csv(unified_csv_path, index=False)
        return True
        
    except Exception as e:
        return False