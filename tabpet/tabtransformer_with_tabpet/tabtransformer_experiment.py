"""
Simplified TabTransformer experiment module with positional encodings
Only supports concat_fixed PE integration and alpha_zero vs real_pe comparison
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
import os

# Import from other modules
from dag_construction import load_dataset
from utils import VerbosePrinter, set_random_seeds

warnings.filterwarnings('ignore')

import math
import typing as ty
from torch import Tensor

from tab_transformer_pytorch import TabTransformer, MLP

def get_openml_categorical_features(dataset_name, dataset_version):
    """
    Get categorical features directly from OpenML dataset
    """
    # Manual mapping for datasets that fail to load from OpenML
    dataset_key = f"{dataset_name}_v{dataset_version}"
    
    manual_mappings = {
        "adult_v2": ["workclass", "education", "marital-status", "occupation", 
                     "relationship", "race", "sex", "native-country"],
        "cleveland_v1": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
        "cholesterol_v1": ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
        "pharynx_v1": ["Inst", "sex", "Treatment", "Grade", "Condition", 
                       "Site", "T", "N", "Entry", "Status"]
    }
    
    if dataset_key in manual_mappings:
        return manual_mappings[dataset_key]
    
    try:
        # Try different variations of dataset names for OpenML
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
        
        # Identify categorical features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            else:
                # Check if it's actually categorical but encoded as numeric
                unique_vals = X[col].nunique()
                if unique_vals <= 10 and unique_vals < len(X) * 0.05:  
                    categorical_features.append(col)
        
        return categorical_features
        
    except Exception as e:
        return []

class TabTransformerWithPE(TabTransformer):
    """
    TabTransformer with Positional Encodings applied after categorical embedding
    Only supports concat_fixed integration method
    """
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=2,
        continuous_mean_std=None,
        attn_dropout=0.1,
        ff_dropout=0.1,
        use_shared_categ_embed=True,
        shared_categ_dim_divisor=8,
        num_residual_streams=4,
        # PE parameters
        positional_encodings=None,
        pe_alpha=0.1,
        use_pe=True
    ):
        # Calculate adjusted dim for concat_fixed integration
        if use_pe and positional_encodings is not None and len(categories) > 0:
            pe_array = np.array(positional_encodings, dtype=np.float32)
            pe_dim = pe_array.shape[1]
            adjusted_dim = dim - pe_dim
            if adjusted_dim <= 0:
                raise ValueError(f"PE dimension ({pe_dim}) too large! Must be < embedding dim ({dim})")
        else:
            adjusted_dim = dim
        
        # Initialize parent TabTransformer
        super().__init__(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dim_out=dim_out,
            mlp_hidden_mults=mlp_hidden_mults,
            mlp_act=mlp_act,
            num_special_tokens=num_special_tokens,
            continuous_mean_std=continuous_mean_std,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            use_shared_categ_embed=use_shared_categ_embed,
            shared_categ_dim_divisor=shared_categ_dim_divisor,
            num_residual_streams=num_residual_streams
        )
        
        # Store target dimension
        self.target_dim = dim
        self.adjusted_dim = adjusted_dim
        
        # Store PE configuration
        self.use_pe = use_pe and positional_encodings is not None
        self.pe_alpha = pe_alpha
        
        # Process PE for categorical features only
        if self.use_pe and self.num_categories > 0:
            self.positional_encodings = self._process_positional_encodings(positional_encodings)
            
            # Ensure PE matches number of categorical features
            if self.positional_encodings.shape[0] != self.num_categories:
                if self.positional_encodings.shape[0] > self.num_categories:
                    self.positional_encodings = self.positional_encodings[:self.num_categories]
                else:
                    # Pad with zeros if needed
                    padding_size = self.num_categories - self.positional_encodings.shape[0]
                    pe_dim = self.positional_encodings.shape[1]
                    padding = torch.zeros(padding_size, pe_dim, dtype=self.positional_encodings.dtype)
                    self.positional_encodings = torch.cat([self.positional_encodings, padding], dim=0)
            
            pe_dim = self.positional_encodings.shape[1]
            self.pe_dim = pe_dim
        else:
            self.positional_encodings = None

        # For concat_fixed, manually replace category embedding layer
        if self.use_pe and self.num_categories > 0:
            # Replace category embedding layer to output adjusted_dim
            total_tokens = self.num_unique_categories + self.num_special_tokens
            shared_embed_dim = 0 if not use_shared_categ_embed else int(self.adjusted_dim // shared_categ_dim_divisor)
            
            self.category_embed = nn.Embedding(total_tokens, self.adjusted_dim - shared_embed_dim)
            
            if use_shared_categ_embed:
                self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
                nn.init.normal_(self.shared_category_embed, std=0.02)
    
    def _process_positional_encodings(self, pe_values):
        pe_array = np.array(pe_values, dtype=np.float32)
        pe_clipped = np.clip(pe_array, -10, 10)
        return torch.tensor(pe_clipped, dtype=torch.float32)

    def forward(self, x_categ, x_cont, return_attn=False):
        # Handle FT-Transformer calling convention: model(x_num, x_cat)
        # TabTransformer expects: forward(x_categ, x_cont)
        # We need to swap to correct TabTransformer convention:
        x_categ, x_cont = x_cont, x_categ  # Now: x_categ=categorical, x_cont=continuous
        
        xs = []

        # Handle categorical features
        if self.num_categories > 0 and x_categ is not None and x_categ.shape[1] > 0:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

            if self.num_unique_categories > 0:
                x_categ = x_categ + self.categories_offset
                categ_embed = self.category_embed(x_categ)

                if self.use_shared_categ_embed:
                    shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                    categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

                # Apply PE after embedding but before transformer
                if self.use_pe and self.positional_encodings is not None:
                    pe_tensor = self.positional_encodings.to(categ_embed.device)
                    
                    # Apply PE scaling (NO learnable transformation)
                    pe_scaled = self.pe_alpha * pe_tensor  # [num_categories, pe_dim]
                    # Expand PE to match batch size
                    pe_expanded = pe_scaled.unsqueeze(0).expand(categ_embed.shape[0], -1, -1)  # [batch, num_categories, pe_dim]
                    # Concatenate embeddings with PE to get target dimension
                    categ_embed = torch.cat([categ_embed, pe_expanded], dim=-1)  # [batch, num_categories, target_dim]

                x, attns = self.transformer(categ_embed, return_attn = True)
                flat_categ = rearrange(x, 'b ... -> b (...)')
                xs.append(flat_categ)

        # Handle continuous features
        if self.num_continuous > 0 and x_cont is not None and x_cont.shape[1] > 0:
            assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

            if self.continuous_mean_std is not None:
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        # Handle edge cases
        if not xs:
            raise ValueError("No valid input features found. Both categorical and continuous inputs are None or empty.")

        # Combine and get final output
        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns

def prepare_data_for_ft_transformer(X_df, categorical_feature_names, pe_values, verbose=True):
    """
    Prepare data for FT-Transformer format
    """
    printer = VerbosePrinter(verbose)
    
    # Separate categorical and numerical features from original data
    categorical_data = []
    numerical_data = []
    categorical_categories = []  # Store number of categories for each feature
    
    # Track which features are categorical vs numerical
    categorical_feature_info = []
    numerical_feature_info = []
    
    for col in X_df.columns:
        # Check if categorical by multiple criteria
        is_categorical = (
            (categorical_feature_names is not None and col in categorical_feature_names) or 
            pd.api.types.is_categorical_dtype(X_df[col]) or 
            X_df[col].dtype == 'object'
        )
        
        if is_categorical:
            # This is a categorical feature - use label encoding instead of one-hot
            col_data = X_df[col]
            
            # Handle pandas categorical properly
            if pd.api.types.is_categorical_dtype(col_data):
                # Convert categorical to string to avoid category restriction
                col_data = col_data.astype(str)
            
            unique_values = col_data.unique()
            n_categories = len(unique_values)
            
            # Create label encoder for this feature
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
            # This is a numerical feature
            numerical_data.append(X_df[col].values)
            numerical_feature_info.append({'name': col})
    
    # Convert to arrays
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


def aggregate_positional_encodings_average(pe_feature_names, pe_values, current_features, 
                                          dataset_name, dataset_version, verbose=True):
    """
    Aggregate positional encodings using averaging strategy for features from the same original feature
    """
    printer = VerbosePrinter(verbose)
    
    # Get categorical features from OpenML
    categorical_features = get_openml_categorical_features(dataset_name, dataset_version)
    categorical_features_set = set(categorical_features)
    
    # Create mapping from feature names to PE indices
    pe_name_to_idx = {name: idx for idx, name in enumerate(pe_feature_names)}
    
    aggregated_pe_list = []
    feature_pe_mapping = []
    
    for current_feature in current_features:
        # Check if this feature is categorical
        if current_feature in categorical_features_set:
            # For categorical features: find all one-hot encoded versions
            # Look for PE features that start with "{current_feature}_"
            related_pe_indices = []
            related_pe_names = []
            
            target_prefix = f"{current_feature}_"
            for pe_name in pe_feature_names:
                if pe_name.startswith(target_prefix):
                    related_pe_indices.append(pe_name_to_idx[pe_name])
                    related_pe_names.append(pe_name)
            
            if related_pe_indices:
                related_pe_values = pe_values[related_pe_indices]
                # Average the PEs from the same categorical feature
                aggregated_pe = related_pe_values.mean(axis=0)
                feature_pe_mapping.append(f"{current_feature} <- {related_pe_names}")
            else:
                # No matching PE found, create zero vector
                aggregated_pe = np.zeros(pe_values.shape[1])
                feature_pe_mapping.append(f"{current_feature} <- [NO_PE_FOUND]")
        
        else:
            # For numerical features: look for exact match
            if current_feature in pe_name_to_idx:
                # Direct match found
                pe_idx = pe_name_to_idx[current_feature]
                aggregated_pe = pe_values[pe_idx]
                feature_pe_mapping.append(f"{current_feature} <- [{current_feature}]")
            else:
                # No exact match found, create zero vector
                aggregated_pe = np.zeros(pe_values.shape[1])
                feature_pe_mapping.append(f"{current_feature} <- [NO_PE_FOUND]")
        
        aggregated_pe_list.append(aggregated_pe)
    
    aggregated_pe_array = np.array(aggregated_pe_list)
    
    return aggregated_pe_array, feature_pe_mapping


def detect_task_type_and_prepare_target(y, verbose=True):
    """
    Automatically detect if this is a classification or regression task and prepare target accordingly
    """
    printer = VerbosePrinter(verbose)
    
    # Check if target contains non-numeric values or has categorical type
    if hasattr(y, 'dtype'):
        if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
            task_type = 'classification'
        else:
            # Check if it's integer-like (could be classification) or float (likely regression)
            unique_vals = len(np.unique(y))
            if unique_vals <= 20 and np.allclose(y, np.round(y)):  # Heuristic for classification
                task_type = 'classification'
            else:
                task_type = 'regression'
    else:
        # For pandas Series or other types, check unique values
        try:
            # Try to convert to numeric first
            y_numeric = pd.to_numeric(y, errors='coerce')
            if y_numeric.isnull().any():
                # Contains non-numeric values, must be classification
                task_type = 'classification'
            else:
                # All numeric, use heuristic
                unique_vals = len(pd.Series(y).unique())
                if unique_vals <= 20:
                    task_type = 'classification'
                else:
                    task_type = 'regression'
        except:
            task_type = 'classification'
    
    if task_type == 'classification':
        # Encode categorical targets to integers
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        
        return y_encoded, task_type, num_classes, label_encoder
    else:
        # For regression, return as-is
        y_numeric = pd.to_numeric(y, errors='coerce')
        return y_numeric.values, task_type, 1, None


def train_model(model, train_loader, val_loader, num_epochs=500, learning_rate=1e-4, task_type='auto', 
                early_stopping=False, patience=50, min_delta=1e-6, min_epochs=50, verbose_freq=20, verbose=True):
    """
    Train the TabTransformer model with GPU support and validation-based model selection
    """
    printer = VerbosePrinter(verbose)
    
    # Move model to GPU
    device = next(model.parameters()).device
    
    # Auto-detect task type based on output dimension
    if task_type == 'auto':
        if hasattr(model, 'head'):
            output_dim = model.head.out_features
        else:
            output_dim = model.d_out
        
        if output_dim == 1:
            task_type = 'regression'
        else:
            task_type = 'classification'
    
    # Choose appropriate loss function
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
    
    # Model selection variables
    if task_type == 'classification':
        best_metric = -float('inf')  # Balanced accuracy (higher is better)
        metric_name = "Balanced Accuracy"
        # Adjust default parameters for accuracy
        if early_stopping:
            patience = max(30, patience)  # Minimum 30 epochs patience
            min_delta = max(1e-4, min_delta)  # Minimum 1e-4 improvement threshold
    else:
        best_metric = float('inf')   # Validation loss (lower is better)
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
        
        # Training phase
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        current_metric = 0.0  # Current epoch evaluation metric
        
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
                
                current_metric = val_loss / len(val_loader)  # Regression still uses validation loss
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if task_type == 'classification':
            val_balanced_accuracies.append(current_metric)  # Store balanced accuracy
        
        # Model selection and early stopping based on selected metric
        improved = False
        if task_type == 'classification':
            # Classification: higher balanced accuracy is better
            if current_metric > best_metric:
                improved = True
        else:
            # Regression: lower validation loss is better
            if current_metric < best_metric:
                improved = True
        
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
        # Early stopping logic
        if early_stopping and epoch >= min_epochs:
            if task_type == 'classification':
                # Classification: balanced accuracy improvement
                if current_metric > best_metric - min_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                # Regression: validation loss reduction
                if current_metric < best_metric + min_delta:
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                early_stopped = True
                break
    
    # Always load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        # Verify the best model metric
        model.eval()
        verified_metric = 0.0
        with torch.no_grad():
            if task_type == 'classification':
                # Recalculate balanced accuracy for verification
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
    """
    Evaluate TabTransformer model performance with GPU support
    """
    printer = VerbosePrinter(verbose)
    
    # Auto-detect task type
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
            # Handle empty features
            if x_cat.shape[1] == 0:
                x_cat = None
            if x_num.shape[1] == 0:
                x_num = None
            
            outputs = model(x_num, x_cat)
            
            if task_type == 'classification':
                # For classification, get predicted classes
                predicted_classes = torch.argmax(outputs, dim=1)
                predictions.extend(predicted_classes.cpu().numpy())
                
                # Handle target format
                if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                    y_batch = y_batch.squeeze(-1)
                actuals.extend(y_batch.cpu().numpy())
            else:
                # For regression, use raw outputs
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    if task_type == 'regression':
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)  # Convert MSE to RMSE
        r2 = r2_score(actuals, predictions)
        return rmse, r2, predictions, actuals
    else:  # classification
        # Calculate balanced accuracy
        model.eval()
        balanced_accuracy = torch.tensor(0.0).to(device)
        
        # Calculate Class Weights
        test_labels = test_loader.dataset.tensors[2]  # test labels
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
                # Handle empty features
                if x_cat.shape[1] == 0:
                    x_cat = None
                if x_num.shape[1] == 0:
                    x_num = None
                
                # Handle target format
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
    all_results = []
    task_type = task_info['task_type']
    
    for seed in seeds:
        # Reset environment with specific seed
        device = next(iter(train_loader.dataset.tensors)).device
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Set all random seeds
        set_random_seeds(seed)
        
        # Create fresh model
        model = model_creator_func()
        
        # Train model with early stopping using validation set
        train_losses, val_losses, best_val_loss_verified, best_epoch = train_model(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, learning_rate=learning_rate, task_type=task_type,
            early_stopping=early_stopping, patience=patience, min_delta=min_delta, 
            min_epochs=min_epochs, verbose_freq=20, verbose=False
        )
        
        # Calculate time saved
        epochs_trained = len(train_losses)
        epochs_saved = num_epochs - epochs_trained
        
        # Evaluate model on test set
        metric1, metric2, preds, actuals = evaluate_model(model, test_loader, task_type=task_type, verbose=False)
        
        # Store results based on task type
        result = {
            'seed': seed,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': best_val_loss_verified,  # Use best model validation loss
            'best_val_loss': best_val_loss_verified,   # Should be same as final_val_loss
            'best_epoch': best_epoch,
            'epochs_trained': epochs_trained,
            'epochs_saved': epochs_saved,
            'early_stopped': epochs_trained < num_epochs
        }
        
        if task_type == 'regression':
            result['rmse'] = metric1
            result['r2'] = metric2
        else:  # classification
            result['accuracy'] = metric1
            result['f1'] = metric2
        
        all_results.append(result)
        
        # Clean up
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return all_results


def summarize_multi_seed_results(results, experiment_name="", task_info=None, verbose=True):
    """
    Summarize results from multiple seeds
    """
    if task_info is None:
        # Fallback: try to detect task type from results
        if 'rmse' in results[0]:
            task_type = 'regression'
        else:
            task_type = 'classification'
    else:
        task_type = task_info['task_type']
    
    # Extract early stopping statistics
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
    else:  # classification
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


def run_tabtransformer_experiment(dag_results, llpe_results, dataset_name, dataset_version, 
                                config, device, verbose=True):
    """Main function to run TabTransformer experiment"""
    printer = VerbosePrinter(verbose)
    
    # Load dataset and prepare data
    data = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)
    
    # Extract features and target using OpenML structure
    X_df_features = data.data  # Features only
    y = data.target   # Target only
    
    # Detect task type
    y_processed, task_type, num_classes, label_encoder = detect_task_type_and_prepare_target(y, verbose=False)
    
    # Prepare data for TabTransformer
    categorical_feature_names = None
    if 'categorical_cols' in dag_results:
        categorical_feature_names = dag_results['categorical_cols']

    X_categorical, X_numerical, categories, categorical_info, numerical_info = prepare_data_for_ft_transformer(
        X_df_features, categorical_feature_names, llpe_results['positional_encodings'], verbose=False
    )

    # Fix categories format for TabTransformer
    if not categories or len(categories) == 0:
        # No categorical features, ensure X_categorical is empty
        X_categorical = np.array([]).reshape(len(X_df_features), 0).astype(np.int64)
        categories = ()
    else:
        # Ensure categories is tuple format for TabTransformer
        categories = tuple(categories)
    
    # Load and aggregate positional encodings
    pe_feature_names = llpe_results['encoding_df'].index.tolist()
    pe_values = llpe_results['positional_encodings']
    ft_feature_order = list(X_df_features.columns)
    
    pe_values_aligned, pe_mapping = aggregate_positional_encodings_average(
        pe_feature_names, pe_values, ft_feature_order, dataset_name, dataset_version, verbose=False
    )    
    
    # Create model configuration
    model_config = config['tabtransformer_config'].copy()
    model_config.update({
        'categories': tuple(categories) if categories else (),
        'num_continuous': X_numerical.shape[1],
        'dim_out': num_classes if task_type == 'classification' else 1,
    })

    # Split the data
    stratify_param = y_processed if task_type == 'classification' else None

    # First split: 80% train+val, 20% test
    X_cat_temp, X_cat_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
        X_categorical, X_numerical, y_processed, test_size=0.2, random_state=42,
        stratify=stratify_param
    )

    # Second split: 75% of remaining data for train (60% of total), 25% for val (20% of total)
    if task_type == 'classification':
        stratify_temp = y_temp
    else:
        stratify_temp = None

    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_cat_temp, X_num_temp, y_temp, test_size=0.25, random_state=42,
        stratify=stratify_temp
    )

    # Normalize numerical features
    if X_numerical.shape[1] > 0:
        scaler = StandardScaler()
        X_num_train = scaler.fit_transform(X_num_train)
        X_num_val = scaler.transform(X_num_val)
        X_num_test = scaler.transform(X_num_test)

    # Convert to tensors and move to GPU
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

    # Store task information
    task_info = {
        'task_type': task_type,
        'num_classes': num_classes,
        'label_encoder': label_encoder
    }
    
    # Run experiment with alpha_zero vs real_pe
    pe_alphas = config['experiment_config']['pe_alphas']
    
    multi_seed_alpha_results = {}
    
    for alpha in pe_alphas:
        # Define model creator based on configuration
        def create_alpha_model():
            if alpha == 0.0:
                # alpha_zero baseline: Extended architecture with alpha=0
                return TabTransformerWithPE(
                    **model_config,
                    positional_encodings=pe_values_aligned,  # Real PE matrix (but alpha=0)
                    use_pe=True,
                    pe_alpha=0.0
                ).to(device)
            else:
                # real_pe: Extended architecture with real PE
                return TabTransformerWithPE(
                    **model_config,
                    positional_encodings=pe_values_aligned,  # Real PE matrix
                    use_pe=True,
                    pe_alpha=alpha
                ).to(device)
        
        # Run multi-seed experiment
        alpha_results = run_multi_seed_experiment(
            create_alpha_model, train_loader, val_loader, test_loader,
            task_info,
            seeds=config['experiment_config']['seeds'], 
            num_epochs=config['training_config']['num_epochs'],
            early_stopping=config['training_config']['early_stopping'],
            verbose=False
        )
        
        # Summarize results
        alpha_summary = summarize_multi_seed_results(alpha_results, "", task_info, verbose=False)
        
        # Store results
        multi_seed_alpha_results[alpha] = alpha_summary
        
        # Print Stage 3: Performance Results
        if alpha == 0.0:
            if task_info['task_type'] == 'regression':
                printer.print(f"Baseline (alpha={alpha}): RMSE={alpha_summary['rmse_mean']:.4f}±{alpha_summary['rmse_std']:.4f}, R²={alpha_summary['r2_mean']:.4f}±{alpha_summary['r2_std']:.4f}")
            else:
                printer.print(f"Baseline (alpha={alpha}): Accuracy={alpha_summary['acc_mean']:.4f}±{alpha_summary['acc_std']:.4f}, F1={alpha_summary['f1_mean']:.4f}±{alpha_summary['f1_std']:.4f}")
        else:
            # Calculate and show improvement
            baseline_results = multi_seed_alpha_results[0.0]
            if task_info['task_type'] == 'regression':
                rmse_improvement = ((baseline_results['rmse_mean'] - alpha_summary['rmse_mean']) / baseline_results['rmse_mean']) * 100
                r2_improvement = ((alpha_summary['r2_mean'] - baseline_results['r2_mean']) / baseline_results['r2_mean']) * 100
                printer.print(f"Alpha {alpha}: RMSE={alpha_summary['rmse_mean']:.4f}±{alpha_summary['rmse_std']:.4f} ({rmse_improvement:+.2f}%), R²={alpha_summary['r2_mean']:.4f}±{alpha_summary['r2_std']:.4f} ({r2_improvement:+.2f}%)")
            else:
                acc_improvement = ((alpha_summary['acc_mean'] - baseline_results['acc_mean']) / baseline_results['acc_mean']) * 100
                f1_improvement = ((alpha_summary['f1_mean'] - baseline_results['f1_mean']) / baseline_results['f1_mean']) * 100
                printer.print(f"Alpha {alpha}: Accuracy={alpha_summary['acc_mean']:.4f}±{alpha_summary['acc_std']:.4f} ({acc_improvement:+.2f}%), F1={alpha_summary['f1_mean']:.4f}±{alpha_summary['f1_std']:.4f} ({f1_improvement:+.2f}%)")
        
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Calculate improvements for all PE alpha values
    baseline_results = multi_seed_alpha_results[0.0]
    for alpha in pe_alphas:
        if alpha > 0.0:  # Skip baseline itself
            result = multi_seed_alpha_results[alpha]
            
            # Calculate improvements relative to baseline
            if task_info['task_type'] == 'regression':
                baseline_metric1 = baseline_results['rmse_mean']
                baseline_metric2 = baseline_results['r2_mean']
                metric1_improvement = ((baseline_metric1 - result['rmse_mean']) / baseline_metric1) * 100  # RMSE improvement
                metric2_improvement = ((result['r2_mean'] - baseline_metric2) / baseline_metric2) * 100  # R² improvement
            else:  # classification
                baseline_metric1 = baseline_results['acc_mean']
                baseline_metric2 = baseline_results['f1_mean'] 
                metric1_improvement = ((result['acc_mean'] - baseline_metric1) / baseline_metric1) * 100  # Accuracy improvement
                metric2_improvement = ((result['f1_mean'] - baseline_metric2) / baseline_metric2) * 100  # F1 improvement
            
            result['metric1_improvement'] = metric1_improvement
            result['metric2_improvement'] = metric2_improvement

    # Find best alpha (excluding baseline)
    pe_alphas_only = [a for a in pe_alphas if a > 0.0]

    if task_info['task_type'] == 'regression':
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['r2_mean'])
    else:
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['acc_mean'])

    best_result = multi_seed_alpha_results[best_alpha]
    
    # Print final TabPET results
    printer.print(f"\nTabPET Best Performance (Alpha {best_alpha}):")
    if task_info['task_type'] == 'regression':
        printer.print(f"RMSE={best_result['rmse_mean']:.4f}±{best_result['rmse_std']:.4f}, R²={best_result['r2_mean']:.4f}±{best_result['r2_std']:.4f}")
        baseline_rmse = baseline_results['rmse_mean']
        improvement = ((baseline_rmse - best_result['rmse_mean']) / baseline_rmse) * 100
        printer.print(f"RMSE Improvement: {improvement:+.2f}%")
    else:
        printer.print(f"Accuracy={best_result['acc_mean']:.4f}±{best_result['acc_std']:.4f}, F1={best_result['f1_mean']:.4f}±{best_result['f1_std']:.4f}")
        baseline_acc = baseline_results['acc_mean']
        improvement = ((best_result['acc_mean'] - baseline_acc) / baseline_acc) * 100
        printer.print(f"Accuracy Improvement: {improvement:+.2f}%")

    # Save experiment results
    save_to_unified_csv(multi_seed_alpha_results, {
        'dataset_name': dataset_name,
        'dataset_version': dataset_version,
        'dataset_identifier': f'{dataset_name}_v{dataset_version}',
        'task_type': task_info['task_type'],
        'n_classes': task_info['num_classes'] if task_info['task_type'] == 'classification' else None
    }, task_info, verbose=False)
    
    return {
        'multi_seed_alpha_results': multi_seed_alpha_results,
        'best_alpha': best_alpha,
        'best_result': best_result,
        'baseline_results': baseline_results,
        'task_info': task_info,
        'model_config': model_config
    }


def save_to_unified_csv(experiment_results, dataset_info, task_info, verbose=True):
    """Save experiment results to unified CSV files separated by task type"""
    dataset_id = dataset_info.get('dataset_identifier', 'unknown')
    task_type = task_info.get('task_type', 'unknown')
    
    # Determine file path and columns based on task type
    if task_type == 'classification':
        unified_csv_path = "unified_tabtransformer_classification_results.csv"
    elif task_type == 'regression':
        unified_csv_path = "unified_tabtransformer_regression_results.csv"
    else:
        return False
    
    # Prepare new results data
    new_results = []
    
    if task_type == 'classification':
        # Extract baseline results (alpha = 0.0)
        if 0.0 in experiment_results:
            baseline_result = experiment_results[0.0]
            baseline_accuracy = baseline_result['acc_mean']
            
            # Process all alpha values
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                # Format accuracy result as "mean ± std"
                accuracy_result = f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}"
                
                # Calculate improvement percentage relative to baseline
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
            
            # Process all alpha values
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                # Format RMSE result as "mean ± std"
                rmse_result = f"{result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}"
                
                # Calculate improvement percentage relative to baseline
                # For RMSE, lower is better, so improvement = (baseline - current) / baseline * 100
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
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_results)
    
    # Load existing unified CSV if it exists
    if os.path.exists(unified_csv_path):
        try:
            existing_df = pd.read_csv(unified_csv_path)
            
            # Remove existing results for this dataset (to allow overwriting)
            existing_df = existing_df[existing_df['Dataset'] != dataset_id]
            
            # Combine with new results
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
        except Exception as e:
            combined_df = new_df
    else:
        combined_df = new_df
    
    # Sort by Dataset and Alpha for better organization
    combined_df = combined_df.sort_values(['Dataset', 'Alpha']).reset_index(drop=True)
    
    # Save to CSV
    try:
        combined_df.to_csv(unified_csv_path, index=False)
        return True
        
    except Exception as e:
        return False