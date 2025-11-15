"""
SAINT experiment module with positional encodings
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from torch.utils.data import DataLoader
import warnings
import time
import copy
import os

# Import SAINT components
from models import SAINT
from augmentations import embed_data_mask

# Import from our merged utils
from utils import (
    DataSetCatCon, 
    prepare_data_for_saint, 
    data_split_saint,
    count_parameters,
    VerbosePrinter, 
    set_random_seeds
)

from config import SAINT_CONFIG

warnings.filterwarnings('ignore')

def get_openml_categorical_features(dataset_name, dataset_version):
    """Get categorical features directly from OpenML dataset"""
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
    
def detect_task_type_and_prepare_target(y, verbose=True):
    """Automatically detect if this is a classification or regression task and prepare target accordingly"""
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

def apply_pe_integration_to_embeddings(x_categ_enc, x_cont_enc, positional_encodings, pe_alpha, cat_idxs, con_idxs):
    """Apply PE integration using fixed concatenation"""
    if positional_encodings is None:
        return x_categ_enc, x_cont_enc
    
    device = x_categ_enc.device
    batch_size = x_categ_enc.shape[0]
    full_embedding_dim = x_categ_enc.shape[-1]
    
    # Get PE tensor and apply scaling
    pe_tensor = torch.tensor(positional_encodings, dtype=torch.float32).to(device)
    pe_scaled = pe_alpha * pe_tensor
    pe_dim = pe_scaled.shape[1]
    
    # Calculate truncation dimension
    truncated_dim = full_embedding_dim - pe_dim
    
    # Expand PE for batch
    pe_expanded = pe_scaled.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Process categorical embeddings
    if x_categ_enc.shape[1] > 1 and len(cat_idxs) > 0:
        cls_tokens = x_categ_enc[:, :1, :]
        cat_feature_tokens = x_categ_enc[:, 1:1+len(cat_idxs), :]
        
        cat_truncated = cat_feature_tokens[:, :, :truncated_dim]
        pe_cat = pe_expanded[:, cat_idxs, :]
        cat_enhanced = torch.cat([cat_truncated, pe_cat], dim=-1)
        
        cls_truncated = cls_tokens[:, :, :truncated_dim]
        cls_pe_padding = torch.zeros(batch_size, 1, pe_dim, device=device, dtype=cls_tokens.dtype)
        cls_enhanced = torch.cat([cls_truncated, cls_pe_padding], dim=-1)
        
        x_categ_enc = torch.cat([cls_enhanced, cat_enhanced], dim=1)
    
    # Process continuous embeddings
    if x_cont_enc.shape[1] > 0 and len(con_idxs) > 0:
        cont_truncated = x_cont_enc[:, :, :truncated_dim]
        pe_cont = pe_expanded[:, con_idxs, :]
        x_cont_enc = torch.cat([cont_truncated, pe_cont], dim=-1)
    
    return x_categ_enc, x_cont_enc

def train_model_saint(model, train_loader, val_loader, positional_encodings=None, pe_alpha=0.1, cat_idxs=None, con_idxs=None,
                     num_epochs=500, learning_rate=1e-4, task_type='auto', early_stopping=False, patience=50, 
                     min_delta=1e-6, min_epochs=50, verbose=True):
    """Train the SAINT model with PE integration"""
    printer = VerbosePrinter(verbose)
    
    device = next(model.parameters()).device
    
    # Auto-detect task type
    if task_type == 'auto':
        if hasattr(model, 'mlpfory'):
            output_dim = model.mlpfory.layers[-1].out_features
        else:
            output_dim = 1
        
        if output_dim == 1:
            task_type = 'regression'
        else:
            task_type = 'classification'
    
    # Choose appropriate loss function
    if task_type == 'regression':
        criterion = nn.MSELoss()
    else:
        y_all = []
        for batch in train_loader:
            y_all.extend(batch[2].numpy())
        
        unique_labels, counts = np.unique(y_all, return_counts=True)
        num_classes = len(unique_labels)
        
        weights = []
        for i in range(num_classes):
            class_freq = counts[i] / len(y_all)
            weights.append(1.0 / class_freq)
        
        weights = torch.tensor(weights, dtype=torch.float32) / num_classes
        weights = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    
    # Model selection variables
    if task_type == 'classification':
        best_metric = -float('inf')
        metric_name = "Balanced Accuracy"
        if early_stopping:
            patience = max(30, patience)
            min_delta = max(1e-4, min_delta)
    else:
        best_metric = float('inf')
        metric_name = "Validation Loss"
    
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # Training phase
        for batch_idx, (x_categ, x_cont, y_batch, cat_mask, con_mask) in enumerate(train_loader):
            x_categ, x_cont, y_batch = x_categ.to(device), x_cont.to(device), y_batch.to(device)
            cat_mask, con_mask = cat_mask.to(device), con_mask.to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings using SAINT's embed_data_mask
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
            
            # Apply PE integration
            x_categ_enc, x_cont_enc = apply_pe_integration_to_embeddings(
                x_categ_enc, x_cont_enc, positional_encodings, pe_alpha, cat_idxs, con_idxs
            )
            
            # Continue with SAINT's forward pass
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]  # CLS token
            outputs = model.mlpfory(y_reps)
            
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
        current_metric = 0.0
        
        with torch.no_grad():
            if task_type == 'classification':
                all_preds = []
                all_targets = []
                
                for x_categ, x_cont, y_batch, cat_mask, con_mask in val_loader:
                    x_categ, x_cont, y_batch = x_categ.to(device), x_cont.to(device), y_batch.to(device)
                    cat_mask, con_mask = cat_mask.to(device), con_mask.to(device)
                    
                    if len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
                        y_batch = y_batch.squeeze(-1)
                    
                    _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                    x_categ_enc, x_cont_enc = apply_pe_integration_to_embeddings(
                        x_categ_enc, x_cont_enc, positional_encodings, pe_alpha, cat_idxs, con_idxs
                    )
                    
                    reps = model.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    outputs = model.mlpfory(y_reps)
                    predicted = torch.argmax(outputs, dim=1)
                    
                    val_loss += criterion(outputs, y_batch.long()).item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
                
                # Calculate balanced accuracy
                unique_labels = np.unique(all_targets)
                balanced_acc = 0.0
                for label in unique_labels:
                    mask = np.array(all_targets) == label
                    if mask.sum() > 0:
                        class_acc = (np.array(all_preds)[mask] == label).mean()
                        balanced_acc += class_acc
                balanced_acc /= len(unique_labels)
                
                current_metric = balanced_acc
                
            else:  # regression
                for x_categ, x_cont, y_batch, cat_mask, con_mask in val_loader:
                    x_categ, x_cont, y_batch = x_categ.to(device), x_cont.to(device), y_batch.to(device)
                    cat_mask, con_mask = cat_mask.to(device), con_mask.to(device)
                    
                    _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
                    x_categ_enc, x_cont_enc = apply_pe_integration_to_embeddings(
                        x_categ_enc, x_cont_enc, positional_encodings, pe_alpha, cat_idxs, con_idxs
                    )
                    
                    reps = model.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    outputs = model.mlpfory(y_reps)
                    val_loss += criterion(outputs, y_batch).item()
                
                current_metric = val_loss / len(val_loader)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Model selection and early stopping
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
        
        # Early stopping logic
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
    
    # Always load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, best_metric, best_epoch

def evaluate_model_saint(model, test_loader, positional_encodings=None, pe_alpha=0.1, cat_idxs=None, con_idxs=None, task_type='auto', verbose=True):
    """Evaluate SAINT model performance with PE integration"""
    
    # Auto-detect task type
    if task_type == 'auto':
        if hasattr(model, 'mlpfory'):
            output_dim = model.mlpfory.layers[-1].out_features
        else:
            output_dim = 1
        
        if output_dim == 1:
            task_type = 'regression'
        else:
            task_type = 'classification'
    
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x_categ, x_cont, y_batch, cat_mask, con_mask in test_loader:
            x_categ, x_cont, y_batch = x_categ.to(device), x_cont.to(device), y_batch.to(device)
            cat_mask, con_mask = cat_mask.to(device), con_mask.to(device)
            
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False)
            x_categ_enc, x_cont_enc = apply_pe_integration_to_embeddings(
                x_categ_enc, x_cont_enc, positional_encodings, pe_alpha, cat_idxs, con_idxs
            )
            
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:, 0, :]
            outputs = model.mlpfory(y_reps)
            
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
        unique_labels = np.unique(actuals)
        balanced_acc = 0.0
        for label in unique_labels:
            mask = actuals == label
            if mask.sum() > 0:
                class_acc = (predictions[mask] == label).mean()
                balanced_acc += class_acc
        balanced_acc /= len(unique_labels)
        
        f1 = f1_score(actuals, predictions, average='weighted')
        
        return balanced_acc, f1, predictions, actuals

def run_multi_seed_experiment(model_creator_func, train_loader, val_loader, test_loader, task_info, 
                             device, positional_encodings=None, pe_alpha=0.1, cat_idxs=None, con_idxs=None,
                             seeds=[1,2,3,4,5], num_epochs=500, learning_rate=1e-4,
                             early_stopping=False, patience=50, min_delta=1e-6, min_epochs=50, verbose=True):
    """Run experiment with multiple random seeds for robust results"""
    all_results = []
    task_type = task_info['task_type']
    
    for seed in seeds:
        # Reset environment with specific seed
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Set all random seeds
        set_random_seeds(seed)
        
        # Create fresh model
        model = model_creator_func()
        
        # Train model
        train_losses, val_losses, best_val_metric, best_epoch = train_model_saint(
            model, train_loader, val_loader, 
            positional_encodings=positional_encodings, pe_alpha=pe_alpha, cat_idxs=cat_idxs, con_idxs=con_idxs,
            num_epochs=num_epochs, learning_rate=learning_rate, task_type=task_type,
            early_stopping=early_stopping, patience=patience, min_delta=min_delta, 
            min_epochs=min_epochs, verbose=verbose
        )
        
        # Evaluate model on test set
        metric1, metric2, preds, actuals = evaluate_model_saint(
            model, test_loader, positional_encodings=positional_encodings, pe_alpha=pe_alpha, 
            cat_idxs=cat_idxs, con_idxs=con_idxs, task_type=task_type, verbose=verbose
        )
        
        # Store results
        result = {
            'seed': seed,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': best_val_metric,
            'best_val_loss': best_val_metric,
            'best_epoch': best_epoch,
            'epochs_trained': len(train_losses),
            'epochs_saved': num_epochs - len(train_losses),
            'early_stopped': len(train_losses) < num_epochs
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
    """Summarize results from multiple seeds"""
    printer = VerbosePrinter(verbose)
    
    if task_info is None:
        if 'rmse' in results[0]:
            task_type = 'regression'
        else:
            task_type = 'classification'
    else:
        task_type = task_info['task_type']
    
    if task_type == 'regression':
        rmses = [r['rmse'] for r in results]
        r2s = [r['r2'] for r in results]
        
        rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
        r2_mean, r2_std = np.mean(r2s), np.std(r2s)
        
        return {
            'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
            'r2_mean': r2_mean, 'r2_std': r2_std,
            'all_results': results,
            'task_type': task_type
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
            'task_type': task_type
        }

def aggregate_positional_encodings_average(pe_feature_names, pe_values, current_features, 
                                          dataset_name, dataset_version, verbose=True):
    """Aggregate positional encodings using averaging strategy"""
    
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

def run_saint_experiment(dag_results, llpe_results, dataset_name, dataset_version, 
                        config, device, verbose=True):
    """Main function to run SAINT experiment with PE integration"""
    printer = VerbosePrinter(verbose)
    
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load dataset and prepare data
    data = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)
    
    X_df_features = data.data
    y = data.target
    
    # Detect task type
    y_processed, task_type, num_classes, label_encoder = detect_task_type_and_prepare_target(y, verbose)
    
    # Prepare data for SAINT
    categorical_feature_names = None
    if 'categorical_cols' in dag_results:
        categorical_feature_names = dag_results['categorical_cols']

    X_processed, cat_dims, cat_idxs, con_idxs, nan_mask = prepare_data_for_saint(
        X_df_features, categorical_feature_names, verbose
    )
    
    # Load and aggregate positional encodings
    pe_feature_names = llpe_results['encoding_df'].index.tolist()
    pe_values = llpe_results['positional_encodings']
    saint_feature_order = list(X_df_features.columns)
    
    pe_values_aligned, pe_mapping = aggregate_positional_encodings_average(
        pe_feature_names, pe_values, saint_feature_order, dataset_name, dataset_version, verbose
    )
    
    # Create model configuration for SAINT
    cat_dims_with_cls = np.append(np.array([1]), np.array(cat_dims)).astype(int)
    
    if task_type == 'regression':
        y_dim = 1
    else:
        y_dim = num_classes
    
    pe_dim = pe_values_aligned.shape[1] if pe_values_aligned is not None else 0
    target_embedding_dim = SAINT_CONFIG['target_embedding_dim']

    model_config = {
        'categories': tuple(cat_dims_with_cls),
        'num_continuous': len(con_idxs),
        'dim': target_embedding_dim,
        'depth': SAINT_CONFIG['transformer_depth'],
        'heads': SAINT_CONFIG['attention_heads'],
        'attn_dropout': SAINT_CONFIG['attention_dropout'],
        'ff_dropout': SAINT_CONFIG['ffn_dropout'],
        'mlp_hidden_mults': (4, 2),
        'cont_embeddings': SAINT_CONFIG['cont_embeddings'],
        'attentiontype': SAINT_CONFIG['attentiontype'],
        'final_mlp_style': SAINT_CONFIG['final_mlp_style'],
        'y_dim': y_dim
    }

    # Split the data
    stratify_param = y_processed if task_type == 'classification' else None

    # First split: 80% train+val, 20% test
    train_indices, test_indices = train_test_split(
        range(len(X_processed)), test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Second split: 75% of remaining data for train (60% of total), 25% for val (20% of total)
    if task_type == 'classification':
        stratify_temp = y_processed[train_indices]
    else:
        stratify_temp = None

    train_indices_final, val_indices = train_test_split(
        train_indices, test_size=0.25, random_state=42, stratify=stratify_temp
    )

    # Create SAINT datasets
    X_train, y_train = data_split_saint(X_processed, y_processed, nan_mask, train_indices_final)
    X_val, y_val = data_split_saint(X_processed, y_processed, nan_mask, val_indices)
    X_test, y_test = data_split_saint(X_processed, y_processed, nan_mask, test_indices)

    # Normalize continuous features
    if len(con_idxs) > 0:
        train_mean = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0)
        train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)
    else:
        continuous_mean_std = None

    # Create data loaders
    batch_size = config['training_config']['batch_size']

    # Apply SAINT official batch_size adjustment based on number of features
    n_features = X_df_features.shape[1]
    if n_features > 100:
        batch_size = min(64, batch_size)

    train_ds = DataSetCatCon(X_train, y_train, cat_idxs, task='clf' if task_type == 'classification' else 'reg', continuous_mean_std=continuous_mean_std)
    val_ds = DataSetCatCon(X_val, y_val, cat_idxs, task='clf' if task_type == 'classification' else 'reg', continuous_mean_std=continuous_mean_std)
    test_ds = DataSetCatCon(X_test, y_test, cat_idxs, task='clf' if task_type == 'classification' else 'reg', continuous_mean_std=continuous_mean_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                        num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True)

    # Store task information
    task_info = {
        'task_type': task_type,
        'num_classes': num_classes,
        'label_encoder': label_encoder
    }
    
    # Run alpha_zero vs real_pe experiment
    pe_alphas = config['experiment_config']['pe_alphas']
    
    multi_seed_alpha_results = {}
    
    for alpha in pe_alphas:
        # Define model creator
        def create_alpha_model():
            model = SAINT(**model_config)
            return model.to(device)
        
        # Use real PE for all alphas (alpha=0 means PE scaling is zero)
        current_pe = pe_values_aligned
        
        # Run multi-seed experiment
        alpha_results = run_multi_seed_experiment(
            create_alpha_model, train_loader, val_loader, test_loader,
            task_info, device,
            positional_encodings=current_pe, pe_alpha=alpha, cat_idxs=cat_idxs, con_idxs=con_idxs,
            seeds=config['experiment_config']['seeds'], 
            num_epochs=config['training_config']['num_epochs'],
            early_stopping=config['training_config']['early_stopping'],
            verbose=verbose
        )
        
        # Summarize results
        if alpha == 0.0:
            model_description = "alpha_zero baseline"
        else:
            model_description = f"real_pe (alpha={alpha})"
        
        alpha_summary = summarize_multi_seed_results(alpha_results, model_description, task_info, verbose)
        
        # Store results
        multi_seed_alpha_results[alpha] = alpha_summary
        
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Calculate improvements and find best alpha
    baseline_results = multi_seed_alpha_results[0.0]

    if task_info['task_type'] == 'regression':
        baseline_metric1 = baseline_results['rmse_mean']
        baseline_metric2 = baseline_results['r2_mean']
        printer.print(f"Baseline (alpha=0.0): RMSE={baseline_metric1:.4f}±{baseline_results['rmse_std']:.4f}, R²={baseline_metric2:.4f}±{baseline_results['r2_std']:.4f}")
    else:  # classification
        baseline_metric1 = baseline_results['acc_mean']
        baseline_metric2 = baseline_results['f1_mean'] 
        printer.print(f"Baseline (alpha=0.0): Accuracy={baseline_metric1:.4f}±{baseline_results['acc_std']:.4f}, F1={baseline_metric2:.4f}±{baseline_results['f1_std']:.4f}")

    # Calculate improvements for all PE alpha values
    for alpha in pe_alphas:
        if alpha > 0.0:  # Skip baseline itself
            result = multi_seed_alpha_results[alpha]
            
            # Calculate improvements relative to baseline
            if task_info['task_type'] == 'regression':
                metric1_improvement = ((baseline_metric1 - result['rmse_mean']) / baseline_metric1) * 100  # RMSE improvement
                metric2_improvement = ((result['r2_mean'] - baseline_metric2) / baseline_metric2) * 100  # R² improvement
                printer.print(f"Alpha {alpha}: RMSE={result['rmse_mean']:.4f}±{result['rmse_std']:.4f} ({metric1_improvement:+.2f}%), R²={result['r2_mean']:.4f}±{result['r2_std']:.4f} ({metric2_improvement:+.2f}%)")
            else:  # classification
                metric1_improvement = ((result['acc_mean'] - baseline_metric1) / baseline_metric1) * 100  # Accuracy improvement
                metric2_improvement = ((result['f1_mean'] - baseline_metric2) / baseline_metric2) * 100  # F1 improvement
                printer.print(f"Alpha {alpha}: Accuracy={result['acc_mean']:.4f}±{result['acc_std']:.4f} ({metric1_improvement:+.2f}%), F1={result['f1_mean']:.4f}±{result['f1_std']:.4f} ({metric2_improvement:+.2f}%)")
            
            result['metric1_improvement'] = metric1_improvement
            result['metric2_improvement'] = metric2_improvement

    # Find best alpha (excluding baseline)
    pe_alphas_only = [a for a in pe_alphas if a > 0.0]

    if task_info['task_type'] == 'regression':
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['r2_mean'])
    else:
        best_alpha = max(pe_alphas_only, key=lambda a: multi_seed_alpha_results[a]['acc_mean'])

    best_result = multi_seed_alpha_results[best_alpha]

    printer.print(f"\nTAB-PET FINAL RESULTS")
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
    
    # Save experiment results
    save_to_unified_csv_saint(multi_seed_alpha_results, dataset_name, dataset_version, task_info, verbose)
    
    return {
        'multi_seed_alpha_results': multi_seed_alpha_results,
        'best_alpha': best_alpha,
        'best_result': best_result,
        'baseline_results': baseline_results,
        'task_info': task_info,
        'model_config': model_config
    }

def save_to_unified_csv_saint(experiment_results, dataset_name, dataset_version, task_info, verbose=True):
    """Save experiment results to unified CSV files separated by task type"""
    
    dataset_id = f"{dataset_name}_v{dataset_version}"
    task_type = task_info.get('task_type', 'unknown')
    
    # Determine file path based on task type
    if task_type == 'classification':
        unified_csv_path = "unified_saint_classification_results.csv"
    elif task_type == 'regression':
        unified_csv_path = "unified_saint_regression_results.csv"
    else:
        return False
    
    # Prepare new results data
    new_results = []
    
    if task_type == 'classification':
        if 0.0 in experiment_results:
            baseline_result = experiment_results[0.0]
            baseline_accuracy = baseline_result['acc_mean']
            
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                accuracy_result = f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}"
                
                if alpha == 0.0:
                    improvement_pct = "—"
                else:
                    improvement = ((result['acc_mean'] - baseline_accuracy) / baseline_accuracy) * 100
                    improvement_pct = f"{improvement:+.2f}%"
                
                new_results.append({
                    'Dataset': dataset_id,
                    'Alpha': alpha,
                    'Accuracy_Result': accuracy_result,
                    'Accuracy_Improvement_Pct': improvement_pct
                })
            
    elif task_type == 'regression':
        if 0.0 in experiment_results:
            baseline_result = experiment_results[0.0]
            baseline_rmse = baseline_result['rmse_mean']
            
            for alpha in sorted(experiment_results.keys()):
                result = experiment_results[alpha]
                
                rmse_result = f"{result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}"
                
                if alpha == 0.0:
                    improvement_pct = "—"
                else:
                    improvement = ((baseline_rmse - result['rmse_mean']) / baseline_rmse) * 100
                    improvement_pct = f"{improvement:+.2f}%"
                
                new_results.append({
                    'Dataset': dataset_id,
                    'Alpha': alpha,
                    'RMSE_Result': rmse_result,
                    'RMSE_Improvement_Pct': improvement_pct
                })
    
    # Convert to DataFrame
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
    
    # Sort by Dataset and Alpha
    combined_df = combined_df.sort_values(['Dataset', 'Alpha']).reset_index(drop=True)
    
    # Save to CSV
    try:
        combined_df.to_csv(unified_csv_path, index=False)
        return True
    except Exception as e:
        return False