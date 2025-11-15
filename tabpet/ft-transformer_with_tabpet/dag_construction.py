"""
Graph estimation module, corresponding to (a) and (b) in Figure 1 of the main paper.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import igraph as ig
import networkx as nx
import warnings
from config import DAG_METHOD
import scipy.stats as stats
import lingam
from lingam import DirectLiNGAM
from utils import VerbosePrinter
warnings.filterwarnings('ignore')

def sample_for_dag_construction(X_df, target, max_samples=5000, random_state=42, verbose=True):
    """
    Sample data for DAG construction if dataset is too large
    """
    printer = VerbosePrinter(verbose)
    
    n_samples = len(X_df)
    
    if n_samples <= max_samples:
        return X_df, {'sampled': False, 'original_size': n_samples, 'final_size': n_samples}
    
    try:
        # Detect task type for appropriate sampling strategy
        if hasattr(target, 'dtype'):
            if target.dtype == 'object' or pd.api.types.is_categorical_dtype(target):
                task_type = 'classification'
            else:
                unique_vals = len(np.unique(target))
                if unique_vals <= 20 and np.allclose(target, np.round(target)):
                    task_type = 'classification'
                else:
                    task_type = 'regression'
        else:
            try:
                target_numeric = pd.to_numeric(target, errors='coerce')
                if target_numeric.isnull().any():
                    task_type = 'classification'
                else:
                    unique_vals = len(pd.Series(target).unique())
                    if unique_vals <= 20:
                        task_type = 'classification'
                    else:
                        task_type = 'regression'
            except:
                task_type = 'classification'
        
        if task_type == 'classification':
            # Stratified sampling to preserve class distribution
            target_series = pd.Series(target)
            
            # Use stratified sampling
            X_sampled, _, y_sampled, _ = train_test_split(
                X_df, target_series, 
                train_size=max_samples, 
                stratify=target_series,
                random_state=random_state
            )
            
        else:
            # Simple random sampling for regression
            np.random.seed(random_state)
            sample_indices = np.random.choice(n_samples, max_samples, replace=False)
            X_sampled = X_df.iloc[sample_indices].reset_index(drop=True)
        
        sampling_info = {
            'sampled': True,
            'original_size': n_samples,
            'final_size': max_samples,
            'task_type': task_type,
            'sampling_ratio': max_samples / n_samples
        }
        
        return X_sampled, sampling_info
        
    except Exception as e:
        # Fallback to random sampling
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)
        X_sampled = X_df.iloc[sample_indices].reset_index(drop=True)
        
        sampling_info = {
            'sampled': True,
            'original_size': n_samples,
            'final_size': max_samples,
            'task_type': 'unknown',
            'sampling_ratio': max_samples / n_samples
        }
        
        return X_sampled, sampling_info
    
def run_lingam_analysis(X, method='direct', **kwargs):
    """
    Run LinGAM algorithm
    """
    if method == 'direct':
        model = DirectLiNGAM(random_state=42)
    else:
        raise ValueError(f"Unsupported LinGAM method: {method}")
    
    model.fit(X)
    
    adjacency_matrix = model.adjacency_matrix_
    
    causal_order = model.causal_order_
    
    return adjacency_matrix, causal_order, model

# https://github.com/xunzheng/notears/blob/master/notears/linear.py
def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian."""

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def run_notears_analysis(X, lambda1_values, loss_type='l2', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """
    Run NOTEARS algorithm
    """
    results = {}
    
    for lambda1 in lambda1_values:
        W_est = notears_linear(X, lambda1=lambda1, loss_type=loss_type, 
                              max_iter=max_iter, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold)
        
        if is_dag(W_est):
            results[lambda1] = W_est
    
    # Choose the best lambda1 based on sparsity and DAG constraint
    if results:
        best_lambda1 = min(results.keys())
        W_best = results[best_lambda1]
    else:
        W_best = notears_linear(X, lambda1=0.1, loss_type=loss_type, 
                               max_iter=max_iter, h_tol=h_tol, rho_max=rho_max, w_threshold=w_threshold)
        best_lambda1 = 0.1
    
    return best_lambda1, W_best, results

def run_pearson_correlation_analysis(X, verbose=True):
    """
    Run Pearson correlation analysis
    """
    n_samples, n_features = X.shape
    
    # Compute Pearson correlation matrix
    W_correlation = np.corrcoef(X.T)
    
    if np.isscalar(W_correlation):
        W_correlation = np.array([[0.0]])
    elif W_correlation.ndim == 1:
        W_correlation = W_correlation.reshape(1, 1)
    
    np.fill_diagonal(W_correlation, 0)
    
    stats_dict = {
        'total_edges': np.count_nonzero(W_correlation),
        'sparsity': np.count_nonzero(W_correlation) / W_correlation.size,
        'average_weight': np.abs(W_correlation[W_correlation != 0]).mean() if np.any(W_correlation != 0) else 0,
        'max_weight': np.abs(W_correlation).max(),
        'min_nonzero_weight': np.abs(W_correlation[W_correlation != 0]).min() if np.any(W_correlation != 0) else 0
    }
    
    return W_correlation, stats_dict

def run_spearman_correlation_analysis(X, verbose=True):
    """
    Run Spearman correlation analysis
    """
    n_samples, n_features = X.shape
    
    # Compute Spearman correlation matrix
    spearman_result = stats.spearmanr(X, axis=0)
    W_correlation = spearman_result.correlation
    
    if np.isscalar(W_correlation):
        W_correlation = np.array([[0.0]])
    elif W_correlation.ndim == 1:
        W_correlation = W_correlation.reshape(1, 1)
    
    np.fill_diagonal(W_correlation, 0)
    
    stats_dict = {
        'total_edges': np.count_nonzero(W_correlation),
        'sparsity': np.count_nonzero(W_correlation) / W_correlation.size,
        'average_weight': np.abs(W_correlation[W_correlation != 0]).mean() if np.any(W_correlation != 0) else 0,
        'max_weight': np.abs(W_correlation).max(),
        'min_nonzero_weight': np.abs(W_correlation[W_correlation != 0]).min() if np.any(W_correlation != 0) else 0
    }
    
    return W_correlation, stats_dict

def calculate_mutual_information(x, y, n_bins=10):

    discretizer_x = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretizer_y = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    x_discrete = discretizer_x.fit_transform(x.reshape(-1, 1)).flatten().astype(int)
    y_discrete = discretizer_y.fit_transform(y.reshape(-1, 1)).flatten().astype(int)
    
    joint_hist, x_edges, y_edges = np.histogram2d(x_discrete, y_discrete, bins=[n_bins, n_bins])
    joint_prob = joint_hist / joint_hist.sum()
    
    x_prob = joint_prob.sum(axis=1)
    y_prob = joint_prob.sum(axis=0)
    
    # Calculate MI
    mutual_info = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                mutual_info += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
    
    return mutual_info

def maximum_spanning_tree_from_similarity(similarity_matrix):
    """
    Find maximum spanning tree using Kruskal's algorithm
    """
    n_nodes = similarity_matrix.shape[0]
    
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if similarity_matrix[i, j] > 0:
                edges.append((similarity_matrix[i, j], i, j))
    
    edges.sort(reverse=True)
    
    # Kruskal's algorithm with Union-Find
    parent = list(range(n_nodes))
    rank = [0] * n_nodes
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    # Build tree
    tree_edges = []
    for weight, u, v in edges:
        if union(u, v):
            tree_edges.append((u, v, weight))
            if len(tree_edges) == n_nodes - 1:
                break
    
    tree_adjacency = np.zeros((n_nodes, n_nodes))
    for u, v, weight in tree_edges:
        tree_adjacency[u, v] = weight
        tree_adjacency[v, u] = weight
    
    return tree_adjacency

def run_chowliu_analysis(X, verbose=True):
    """
    Run Chow-Liu algorithm
    """
    n_samples, n_features = X.shape
    
    # Calculate pairwise MI matrix
    mutual_info_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Calculate MI between features i and j
            mi = calculate_mutual_information(X[:, i], X[:, j], n_bins=10)
            mutual_info_matrix[i, j] = mi
            mutual_info_matrix[j, i] = mi  # Symmetric matrix
    
    # Find maximum spanning tree using Chow-Liu algorithm
    W_tree = maximum_spanning_tree_from_similarity(mutual_info_matrix)
    
    stats_dict = {
        'total_edges': np.count_nonzero(W_tree),
        'sparsity': np.count_nonzero(W_tree) / W_tree.size,
        'average_weight': np.mean(W_tree[W_tree != 0]) if np.any(W_tree != 0) else 0,
        'max_weight': np.max(W_tree),
        'min_nonzero_weight': np.min(W_tree[W_tree != 0]) if np.any(W_tree != 0) else 0
    }
    
    return W_tree, stats_dict

def evaluate_lingam_results(adjacency_matrix, threshold=0.1):
    """
    Evaluate LinGAM results
    """
    processed_matrix = adjacency_matrix.copy()
    processed_matrix[np.abs(processed_matrix) < threshold] = 0
    
    stats = {
        'total_edges': np.count_nonzero(processed_matrix),
        'sparsity': np.count_nonzero(processed_matrix) / processed_matrix.size,
        'average_weight': np.abs(processed_matrix[processed_matrix != 0]).mean() if np.any(processed_matrix != 0) else 0,
        'max_weight': np.abs(processed_matrix).max(),
        'min_nonzero_weight': np.abs(processed_matrix[processed_matrix != 0]).min() if np.any(processed_matrix != 0) else 0
    }
    
    return processed_matrix, stats

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def load_dataset(dataset_name, dataset_version, verbose=True):
    printer = VerbosePrinter(verbose)
    
    data = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)
    D_df = data.frame
    X_df = data.data
    
    return data, D_df, X_df

def apply_simple_preprocessing(X_df, verbose=True):
    """Apply one-hot encoding preprocessing"""
    printer = VerbosePrinter(verbose)
    
    numerical_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    X_numerical = X_df[numerical_cols].copy()

    if X_numerical.isnull().any().any():
        rows_before = X_numerical.shape[0]
        X_numerical = X_numerical.dropna()

    # Handle categorical columns with one-hot encoding
    X_categorical_encoded = pd.DataFrame(index=X_numerical.index)

    if categorical_cols:
        for col in categorical_cols:
            cat_data = X_df.loc[X_numerical.index, col].copy()
            
            if cat_data.isnull().any():
                cat_data = cat_data.fillna('Missing')
            
            n_unique = cat_data.nunique()
            if n_unique > 10:
                top_categories = cat_data.value_counts().head(9).index
                cat_data = cat_data.where(cat_data.isin(top_categories), 'Other')
            
            encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
            encoded = encoder.fit_transform(cat_data.values.reshape(-1, 1))
            
            if hasattr(encoder, 'get_feature_names_out'):
                encoded_cols = encoder.get_feature_names_out([col])
            else:
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            
            encoded_df = pd.DataFrame(encoded, index=X_numerical.index, columns=encoded_cols)
            X_categorical_encoded = pd.concat([X_categorical_encoded, encoded_df], axis=1)

    if not X_categorical_encoded.empty:
        X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    else:
        X_processed = X_numerical

    return X_processed, numerical_cols, categorical_cols

def apply_robust_preprocessing(X_df, verbose=True):
    """Handle datasets with many missing values"""
    printer = VerbosePrinter(verbose)
    
    ORIGINAL_OBSERVATIONS = X_df.shape[0]
    ORIGINAL_FEATURES = X_df.shape[1]

    def analyze_missing_data(df):
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100,
            'data_type': df.dtypes
        })
        missing_stats = missing_stats[missing_stats.missing_count > 0].sort_values('missing_percentage', ascending=False)
        
        return missing_stats

    def robust_data_type_conversion(df):
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].dtype.name in ['category', 'object']:
                df_processed[col] = df_processed[col].astype('object')
                df_processed[col] = df_processed[col].fillna('NaN_MISSING')
            elif df_processed[col].dtype.name in ['bool']:
                df_processed[col] = df_processed[col].astype('object')
                df_processed[col] = df_processed[col].fillna('NaN_MISSING')
        
        return df_processed

    X_df_converted = robust_data_type_conversion(X_df)

    numerical_cols = X_df_converted.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df_converted.select_dtypes(include=['object']).columns.tolist()

    missing_analysis = analyze_missing_data(X_df)

    def apply_robust_missing_data_strategy(df_original, df_converted, numerical_cols, categorical_cols, missing_analysis):
        
        # Drop features with >70% missing (academic threshold)
        high_missing_features = missing_analysis[missing_analysis.missing_percentage > 70].column.tolist()
        if high_missing_features:
            df_converted = df_converted.drop(columns=high_missing_features)
            numerical_cols = [col for col in numerical_cols if col not in high_missing_features]
            categorical_cols = [col for col in categorical_cols if col not in high_missing_features]
        
        X_numerical = df_converted[numerical_cols].copy() if numerical_cols else pd.DataFrame()
        
        if not X_numerical.empty:
            remaining_complete_cases = X_numerical.dropna().shape[0]
            remaining_percentage = remaining_complete_cases / X_numerical.shape[0] * 100
            
            if remaining_percentage >= 30:  # keep if >=30% complete cases
                X_numerical = X_numerical.dropna()
            else:
                try:
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    imputer = IterativeImputer(random_state=42, max_iter=10)
                    X_numerical_imputed = imputer.fit_transform(X_numerical)
                    X_numerical = pd.DataFrame(X_numerical_imputed, columns=X_numerical.columns, index=X_numerical.index)
                except ImportError:
                    imputer = SimpleImputer(strategy='median')
                    X_numerical_imputed = imputer.fit_transform(X_numerical)
                    X_numerical = pd.DataFrame(X_numerical_imputed, columns=X_numerical.columns, index=X_numerical.index)
        
        return df_converted, X_numerical, numerical_cols, categorical_cols

    X_df_processed, X_numerical, numerical_cols, categorical_cols = apply_robust_missing_data_strategy(
        X_df, X_df_converted, numerical_cols, categorical_cols, missing_analysis
    )

    X_categorical_encoded = pd.DataFrame(index=X_numerical.index)

    if categorical_cols and not X_numerical.empty:
        for col in categorical_cols:
            if col in X_df_processed.columns:
                cat_data = X_df_processed.loc[X_numerical.index, col].copy()
                
                if cat_data.isnull().any():
                    cat_data = cat_data.fillna('Missing_Value')
                
                cat_data = cat_data.astype(str)
                
                n_unique = cat_data.nunique()
                max_categories = min(20, max(5, int(len(cat_data) * 0.1)))
                
                if n_unique > max_categories:
                    value_counts = cat_data.value_counts()
                    top_categories = value_counts.head(max_categories-1).index.tolist()
                    
                    mask = cat_data.isin(top_categories)
                    cat_data = cat_data.where(mask, 'Other_Category')
                    
                cat_data = cat_data.fillna('Unknown')
                
                try:
                    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                    encoded = encoder.fit_transform(cat_data.values.reshape(-1, 1))
                    
                    if hasattr(encoder, 'get_feature_names_out'):
                        encoded_cols = encoder.get_feature_names_out([col])
                    else:
                        categories = encoder.categories_[0]
                        if len(categories) > 1:
                            encoded_cols = [f"{col}_{cat}" for cat in categories[1:]]
                        else:
                            encoded_cols = []
                    
                    if len(encoded_cols) > 0:
                        encoded_df = pd.DataFrame(encoded, index=X_numerical.index, columns=encoded_cols)
                        X_categorical_encoded = pd.concat([X_categorical_encoded, encoded_df], axis=1)
                        
                except Exception as e:
                    continue

    if not X_categorical_encoded.empty and not X_numerical.empty:
        X_df_final = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    elif not X_numerical.empty:
        X_df_final = X_numerical
    else:
        raise ValueError("No data remaining after preprocessing!")

    # Handle infinite or large values
    if np.any(np.isinf(X_df_final.values)):
        X_df_final = X_df_final.replace([np.inf, -np.inf], np.nan)
        if X_df_final.isnull().any().any():
            imputer = SimpleImputer(strategy='median')
            X_df_final = pd.DataFrame(imputer.fit_transform(X_df_final), 
                                      columns=X_df_final.columns, 
                                      index=X_df_final.index)

    return X_df_final, numerical_cols, categorical_cols

def preprocess_data(X_df, verbose=True):
    try:
        # Try simple preprocessing first
        X_processed, numerical_cols, categorical_cols = apply_simple_preprocessing(X_df, verbose)
        return X_processed, numerical_cols, categorical_cols
    except Exception as e:
        X_processed, numerical_cols, categorical_cols = apply_robust_preprocessing(X_df, verbose)
        return X_processed, numerical_cols, categorical_cols

def run_dag_construction(dataset_name, dataset_version, method='lingam', thresholds=None, verbose=True):
    """Main function to run graph estimation"""
    printer = VerbosePrinter(verbose)
    
    data, D_df, X_df = load_dataset(dataset_name, dataset_version, verbose)

    X_df_sampled, sampling_info = sample_for_dag_construction(
        X_df, data.target, max_samples=5000, random_state=42, verbose=verbose
    )

    X_processed, numerical_cols, categorical_cols = preprocess_data(X_df_sampled, verbose)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    feature_names = list(X_processed.columns)
    n_features = len(feature_names)
    
    # Run graph estimation algorithm based on method
    results = {}
    
    if method == 'lingam':
        if thresholds is None:
            thresholds = [0.01, 0.05, 0.1, 0.2]
        
        # Run DirectLiNGAM
        W_raw, causal_order, lingam_model = run_lingam_analysis(X_scaled, method='direct')
        
        for threshold in thresholds:
            W_processed, stats = evaluate_lingam_results(W_raw, threshold=threshold)
            
            results[threshold] = {
                'matrix': W_processed,
                'stats': stats
            }

        best_threshold = min(results.keys())
        for threshold, result in results.items():
            if 10 <= result['stats']['total_edges'] <= 50:
                best_threshold = threshold
                break

        W_best = results[best_threshold]['matrix']
        best_stats = results[best_threshold]['stats']
        best_param = best_threshold
        
    elif method == 'notears':

        from config import NOTEARS_LAMBDA1_VALUES, NOTEARS_LOSS_TYPE, NOTEARS_MAX_ITER, NOTEARS_H_TOL, NOTEARS_RHO_MAX, NOTEARS_W_THRESHOLD
        
        lambda1_values = thresholds if thresholds is not None else NOTEARS_LAMBDA1_VALUES
        
        # Run NOTEARS
        best_lambda1, W_best, notears_results = run_notears_analysis(
            X_scaled, 
            lambda1_values=lambda1_values,
            loss_type=NOTEARS_LOSS_TYPE,
            max_iter=NOTEARS_MAX_ITER,
            h_tol=NOTEARS_H_TOL,
            rho_max=NOTEARS_RHO_MAX,
            w_threshold=NOTEARS_W_THRESHOLD
        )
        
        best_stats = {
            'total_edges': np.count_nonzero(W_best),
            'sparsity': np.count_nonzero(W_best) / W_best.size,
            'average_weight': np.abs(W_best[W_best != 0]).mean() if np.any(W_best != 0) else 0,
            'max_weight': np.abs(W_best).max(),
            'min_nonzero_weight': np.abs(W_best[W_best != 0]).min() if np.any(W_best != 0) else 0
        }
        
        best_param = best_lambda1
        results = {lambda1: {'matrix': W, 'stats': best_stats} for lambda1, W in notears_results.items()}
        
    elif method == 'pearson':
        # Run Pearson correlation
        W_best, best_stats = run_pearson_correlation_analysis(X_scaled, verbose=verbose)
        
        best_param = 'no_threshold'
        results = {'no_threshold': {'matrix': W_best, 'stats': best_stats}}
        W_raw_for_viz = W_best
        causal_order_for_viz = None
        
    elif method == 'spearman':
        # Run Spearman correlation
        W_best, best_stats = run_spearman_correlation_analysis(X_scaled, verbose=verbose)
        
        best_param = 'no_threshold'
        results = {'no_threshold': {'matrix': W_best, 'stats': best_stats}}
        W_raw_for_viz = W_best
        causal_order_for_viz = None
    
    elif method == 'chowliu':
        # Run Chow-Liu tree
        W_best, best_stats = run_chowliu_analysis(X_scaled, verbose=verbose)
        
        best_param = 'no_threshold'
        results = {'no_threshold': {'matrix': W_best, 'stats': best_stats}}
        W_raw_for_viz = W_best
        causal_order_for_viz = None

    else:
        raise ValueError(f"Unknown method: {method}")
    
    def identify_original_features_correct(feature_names, numerical_cols, categorical_cols):

        feature_mapping = {}
        original_features = {}
        
        for i, name in enumerate(feature_names):
            original_feature = None
            
            if name in numerical_cols:
                original_feature = name
            else:
                for cat_col in categorical_cols:
                    if name.startswith(cat_col + '_'):
                        original_feature = cat_col
                        break
                
                if original_feature is None:
                    parts = name.split('_')
                    if len(parts) >= 2:
                        potential_original = '_'.join(parts[:-1])
                        similar_features = [fn for fn in feature_names if fn.startswith(potential_original + '_') and fn != name]
                        if len(similar_features) >= 1:
                            original_feature = potential_original
                        else:
                            if len(parts) >= 3:
                                potential_original = '_'.join(parts[:-2])
                                similar_features = [fn for fn in feature_names if fn.startswith(potential_original + '_') and fn != name]
                                if len(similar_features) >= 1:
                                    original_feature = potential_original
                    
                    if original_feature is None:
                        original_feature = name
            
            feature_mapping[i] = original_feature
            if original_feature not in original_features:
                original_features[original_feature] = []
            original_features[original_feature].append(i)
        
        return feature_mapping, original_features

    feature_mapping, original_features = identify_original_features_correct(feature_names, numerical_cols, categorical_cols)

    # just used for analysis
    def calculate_inter_intra_efficiency(W_matrix, feature_mapping, original_features):
        """Calculate inter-feature vs intra-feature relationship efficiency."""
        n_nodes = W_matrix.shape[0]
        
        inter_feature_weight_sq_sum = 0.0
        intra_feature_weight_sq_sum = 0.0
        
        inter_edge_count = 0
        intra_edge_count = 0
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and W_matrix[i, j] != 0:
                    weight_sq = W_matrix[i, j] ** 2
                    
                    if feature_mapping[i] == feature_mapping[j]:
                        intra_feature_weight_sq_sum += weight_sq
                        intra_edge_count += 1
                    else:
                        inter_feature_weight_sq_sum += weight_sq
                        inter_edge_count += 1
        
        total_weight_sq_sum = inter_feature_weight_sq_sum + intra_feature_weight_sq_sum
        efficiency = inter_feature_weight_sq_sum / total_weight_sq_sum if total_weight_sq_sum > 0 else 0.0
        
        detailed_stats = {
            'total_edges': inter_edge_count + intra_edge_count,
            'inter_edge_count': inter_edge_count,
            'intra_edge_count': intra_edge_count,
            'inter_weight_sq_sum': inter_feature_weight_sq_sum,
            'intra_weight_sq_sum': intra_feature_weight_sq_sum,
            'total_weight_sq_sum': total_weight_sq_sum,
            'efficiency': efficiency,
            'inter_edge_ratio': inter_edge_count / (inter_edge_count + intra_edge_count) if (inter_edge_count + intra_edge_count) > 0 else 0.0
        }
        
        return inter_feature_weight_sq_sum, intra_feature_weight_sq_sum, efficiency, detailed_stats

    inter_weight_sq, intra_weight_sq, efficiency, stats = calculate_inter_intra_efficiency(
        W_best, feature_mapping, original_features
    )

    printer.print(f"\nDAG Matrix ({W_best.shape[0]}x{W_best.shape[1]}):")
    printer.print(W_best)

    return {
        'W_raw': W_raw if method == 'lingam' else W_best,
        'W_best': W_best,
        'causal_order': causal_order if method == 'lingam' else None,
        'feature_names': feature_names,
        'n_features': n_features,
        'best_param': best_param,
        'best_stats': best_stats,
        'method': method,
        'scaler': scaler,
        'X_processed': X_processed,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'efficiency': efficiency,
        'feature_mapping': feature_mapping,
        'original_features': original_features,
        'sampling_info': sampling_info
    }