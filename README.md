# Tab-PET: Graph-Based Positional Encoding for Tabular Transformers

This repository contains the core implementation code for **Tab-PET**, a graph-based framework for estimating positional encodings (PEs) to inject structural inductive biases into transformer-based architectures for tabular data.

## File Structure

* main.py: Main execution script
* config.py: Configuration parameters
* dag_construction.py: Graph estimation, corresponding to (a) and (b) in Figure 1 of the main paper.
* llpe_generation.py: PE creation, corresponding to (c) in Figure 1 of the main paper.
* ft_transformer_experiment.py: FT-Transformer with Tab-PET implementation, corresponding to (d) in Figure 1 of the main paper.
* utils.py: Utility functions
* README.md
* requirements.txt

## Requirements

Please install necessary packages before running the code, such as:
- PyTorch
- scikit-learn
- pandas
- numpy
- scipy
- lingam

For full package list and version requirements, please refer to `environment.yml`.

## Quick Start

For a quick demonstration, run:

```bash
python main.py cmc 1 --method spearman --gpu_id 0 --pe_alphas 0 1
```
This command will:

* Use the cmc dataset (version 1) from OpenML
* Apply Spearman correlation for graph estimation
* Compare vanilla FT-Transformer baseline (α=0) with Tab-PET (α=1)

## Usage
Basic Command Structure
```bash
python main.py <dataset_name> <dataset_version> [options]
```

### Required Arguments

* dataset_name: OpenML dataset name (e.g., "cmc", "california", "blood-transfusion")
* dataset_version: OpenML dataset version number (e.g., 1, 4). Please refer to technical appendix B.1 for detailed dataset names and properties.

### Optional Arguments

--method: Graph estimation method (default: spearman)

* lingam: Linear Non-Gaussian Acyclic Model
* notears: NOTEARS
* pearson: Pearson correlation
* spearman: Spearman correlation
* chowliu: Chow-Liu trees

--gpu_id: GPU device ID (default: 1)

--pe_alphas: PE scaling factors(default: 0.0 0.05 0.1 0.2 0.5 0.7 1.0 2.0 5.0 10.0)

--seeds: Random seeds for experiments (default: 1 2 3 4 5)

--num_epochs: Maximum training epochs (default: 500)

--early_stopping: Enable early stopping (default: True)

--batch_size: Training batch size (default: 32)

### Example Commands
```bash 
# Test multiple alphas
python main.py steel-plates-fault 3 --method spearman --pe_alphas 0 0.1 0.5 1.0 2.0

# Use different graph estimation method
python main.py blood-transfusion-service-center 1 --method pearson --gpu_id 0

# Custom configuration
python main.py california 4 --num_epochs 200 --batch_size 128 --seeds 1 2 3
```

The provided code gives a comparison between:

1. FT-Transformer Baseline: Standard FT-Transformer without PEs (α=0)
2. FT-Transformer + Tab-PET: Enhanced with graph-derived PEs (α>0)

### Extension to Other Architectures
To reproduce results for SAINT and TabTransformer:

1. Run the graph estimation and PE creation stages using this code
2. Extract the generated PE matrix from the results
3. Integrate the PE matrix into vanilla SAINT/TabTransformer implementations using fixed concatenation. The PE integration follows the similar principle as FT-Transformer: concatenate the scaled graph-derived PEs with the original feature embeddings.

## Fair Experiment Setup

For detailed fairness of comparison, please refer to technical appendix F.3.
