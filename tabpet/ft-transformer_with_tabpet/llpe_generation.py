"""
PE creation module, corresponding to (c) in Figure 1 of the main paper.
"""

import numpy as np
import pandas as pd
import torch
import math

from utils import VerbosePrinter

def compute_laplacian_eigenvectors(A, k_first=5, k_last=5):
    """
    Compute normalized Laplacian eigenvectors
    """
    nnodes = A.shape[0]
    
    D_vec = np.sum(A, axis=1).flatten()
    D_vec[D_vec == 0] = 1
    
    # Compute D^(-1/2)
    D_vec_invsqrt = 1 / np.sqrt(D_vec)
    D_invsqrt = np.diag(D_vec_invsqrt)
    
    # Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    L = np.eye(nnodes) - D_invsqrt @ A @ D_invsqrt
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    sort_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    
    # Skip the first eigenvector (corresponding to eigenvalue ≈ 0)
    # Take first k_first eigenvectors after skipping the first one
    first_eigs = eigenvectors[:, 1:k_first+1] if k_first > 0 else np.empty((nnodes, 0))
    first_eigenvals = eigenvalues[1:k_first+1] if k_first > 0 else np.array([])
    
    # Take last k_last eigenvectors
    last_eigs = eigenvectors[:, -k_last:] if k_last > 0 else np.empty((nnodes, 0))
    last_eigenvals = eigenvalues[-k_last:] if k_last > 0 else np.array([])
    
    # Combine first and last eigenvectors
    if first_eigs.size > 0 and last_eigs.size > 0:
        combined_eigenvectors = np.concatenate([first_eigs, last_eigs], axis=1)
        combined_eigenvalues = np.concatenate([first_eigenvals, last_eigenvals])
    elif first_eigs.size > 0:
        combined_eigenvectors = first_eigs
        combined_eigenvalues = first_eigenvals
    elif last_eigs.size > 0:
        combined_eigenvectors = last_eigs
        combined_eigenvalues = last_eigenvals
    else:
        raise ValueError("No eigenvectors selected")
    
    return combined_eigenvalues, combined_eigenvectors, eigenvalues, eigenvectors

def automatic_k_selection(A_symmetric):
    """
    Automatically select k_first and k_last based on eigenvalue window around 1.0
    """
    nnodes = A_symmetric.shape[0]

    # Use same computation as in compute_laplacian_eigenvectors
    D_vec = np.sum(A_symmetric, axis=1).flatten()
    D_vec[D_vec == 0] = 1
    D_vec_invsqrt = 1 / np.sqrt(D_vec)
    D_invsqrt = np.diag(D_vec_invsqrt)
    L = np.eye(nnodes) - D_invsqrt @ A_symmetric @ D_invsqrt
    eigenvalues_temp = np.sort(np.linalg.eigvals(L))

    # Define window to avoid mid-frequency around 1.0
    window_size = 0.25
    low_threshold = 1.0 - window_size   # 0.75
    high_threshold = 1.0 + window_size  # 1.25

    # Count available eigenvalues
    low_freq_count = np.sum((eigenvalues_temp[1:] <= low_threshold))
    high_freq_count = np.sum(eigenvalues_temp >= high_threshold)

    # Get the actual eigenvalues for gap analysis
    low_freq_eigenvals = eigenvalues_temp[1:][eigenvalues_temp[1:] <= low_threshold]
    high_freq_eigenvals = eigenvalues_temp[eigenvalues_temp >= high_threshold]

    # Gap analysis for low frequency
    if len(low_freq_eigenvals) > 3:
        gaps = np.diff(low_freq_eigenvals)
        median_gap = np.median(gaps)
        significant_gaps = gaps > max(10 * median_gap, 0.5)
        if np.any(significant_gaps):
            first_gap_idx = np.where(significant_gaps)[0][0]
            low_freq_count = first_gap_idx + 1  # Cut before the gap

    # Gap analysis for high frequency  
    if len(high_freq_eigenvals) > 3:
        gaps = np.diff(high_freq_eigenvals)
        median_gap = np.median(gaps)
        significant_gaps = gaps > max(10 * median_gap, 0.5)
        if np.any(significant_gaps):
            last_gap_idx = np.where(significant_gaps)[0][-1]
            high_freq_count = len(high_freq_eigenvals) - (last_gap_idx + 1)

    # Auto-select k_first with max constraint of 10
    max_first_k = 10
    k_first = min(low_freq_count, max_first_k)

    # Ensure minimum value
    k_first = max(2, k_first)
    
    # Set k_last equal to k_first
    k_last = k_first
    
    return k_first, k_last

def apply_simple_pe_normalization(eigenvectors):
    # Convert to torch tensors for consistency
    eigenvectors_torch = torch.from_numpy(eigenvectors).float()
    
    # Z-score standardization: (x - mean) / std
    eig_mean = torch.mean(eigenvectors_torch, dim=0, keepdim=True)
    eig_std = torch.std(eigenvectors_torch, dim=0, keepdim=True)
    
    eig_std[eig_std == 0] = 1.0
    
    # Standardize to zero mean and unit variance
    eigenvectors_normalized = (eigenvectors_torch - eig_mean) / eig_std
    
    return eigenvectors_normalized.numpy()

def run_llpe_generation(dag_results, dataset_name, dataset_version, verbose=True):
    """Main function to run PE creation"""
    printer = VerbosePrinter(verbose)
    
    W_best = dag_results['W_best']
    feature_names = dag_results['feature_names']
    n_features = dag_results['n_features']
    
    # Convert graph to graph laplacian
    A_weighted = np.abs(W_best)
    A_symmetric = np.maximum(A_weighted, A_weighted.T)
    
    # Compute Laplacian eigenvectors
    # Automatic k selection based on spectral theory
    k_first, k_last = automatic_k_selection(A_symmetric)

    eigenvalues_selected, eigenvectors_selected, all_eigenvalues, all_eigenvectors = compute_laplacian_eigenvectors(
        A_symmetric, k_first=k_first, k_last=k_last
    )
    
    # Apply normalization
    positional_encodings = apply_simple_pe_normalization(eigenvectors_selected)
    
    encoding_df = pd.DataFrame(positional_encodings, 
                              index=feature_names,
                              columns=[f'PE_dim_{i+1}' for i in range(positional_encodings.shape[1])])

    pe_summary = pd.DataFrame({
        'feature': feature_names,
        'encoding_magnitude': np.linalg.norm(positional_encodings, axis=1),
        'encoding_mean': np.mean(positional_encodings, axis=1),
        'encoding_std': np.std(positional_encodings, axis=1),
        'encoding_min': np.min(positional_encodings, axis=1),
        'encoding_max': np.max(positional_encodings, axis=1)
    }).sort_values('encoding_magnitude', ascending=False)
    
    printer.print(f"\nPE Matrix ({positional_encodings.shape[0]}x{positional_encodings.shape[1]}):")
    printer.print(positional_encodings)

    return {
        'positional_encodings': positional_encodings,
        'encoding_df': encoding_df,
        'pe_summary': pe_summary,
        'eigenvalues_selected': eigenvalues_selected,
        'all_eigenvalues': all_eigenvalues,
        'k_first': k_first,
        'k_last': k_last,
        'A_symmetric': A_symmetric
    }