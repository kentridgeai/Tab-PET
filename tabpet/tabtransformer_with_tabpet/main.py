"""
Main execution script for FT-Transformer with Positional Encodings experiment
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from utils import setup_gpu, set_random_seeds, VerbosePrinter
from dag_construction import run_dag_construction
from llpe_generation import run_llpe_generation
from tabtransformer_experiment import run_tabtransformer_experiment

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FT-Transformer with Positional Encodings experiment')
    
    # Required arguments
    parser.add_argument('dataset_name', type=str, help='OpenML dataset name')
    parser.add_argument('dataset_version', type=int, help='OpenML dataset version')
    
    # Optional arguments with defaults from config
    parser.add_argument('--method', type=str, default=DAG_METHOD,
                       choices=['lingam', 'notears', 'pearson', 'spearman', 'chowliu'],
                       help='DAG construction method (default: spearman)')
    parser.add_argument('--thresholds', nargs='+', type=float, default=DEFAULT_THRESHOLDS,
                       help='LinGAM thresholds or NOTEARS lambda1 values to test (default: 0.01 0.05 0.1 0.2)')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help=f'Batch size for training (default: {TRAINING_CONFIG["batch_size"]})')
    parser.add_argument('--gpu_id', type=int, default=GPU_CONFIG['default_gpu_id'],
                       help='GPU ID to use (default: 1)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output (default: True)')
    parser.add_argument('--seeds', nargs='+', type=int, default=EXPERIMENT_CONFIG['seeds'],
                       help='Random seeds for experiments (default: 1 2 3 4 5)')
    parser.add_argument('--pe_alphas', nargs='+', type=float, default=EXPERIMENT_CONFIG['pe_alphas'],
                       help='PE alpha values to test (default: 0.0 0.05 0.1 0.2 0.5 0.7 1.0 2.0 5.0 10.0)')
    parser.add_argument('--num_epochs', type=int, default=TRAINING_CONFIG['num_epochs'],
                       help='Maximum training epochs (default: 500)')
    parser.add_argument('--early_stopping', action='store_true', default=TRAINING_CONFIG['early_stopping'],
                       help='Enable early stopping (default: True)')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration dictionary from arguments"""
    config = {
        'target_total_dim': TARGET_TOTAL_DIM,
        'model_config': MODEL_CONFIG.copy(),
        'tabtransformer_config': TABTRANSFORMER_CONFIG.copy(),
        'training_config': TRAINING_CONFIG.copy(),
        'experiment_config': EXPERIMENT_CONFIG.copy()
    }
    
    # Update with command line arguments
    config['experiment_config']['seeds'] = args.seeds
    config['experiment_config']['pe_alphas'] = args.pe_alphas
    config['training_config']['num_epochs'] = args.num_epochs
    config['training_config']['early_stopping'] = args.early_stopping
    config['training_config']['batch_size'] = args.batch_size
    
    return config

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create config from arguments
    config = create_config_from_args(args)
    
    # Setup printer
    printer = VerbosePrinter(args.verbose)
    
    # Print experiment configuration
    printer.section("TABTRANSFORMER WITH POSITIONAL ENCODINGS EXPERIMENT")
    printer.print(f"Dataset: {args.dataset_name} v{args.dataset_version}")
    printer.print(f"Method: {args.method}")
    printer.print(f"GPU ID: {args.gpu_id}")
    printer.print(f"Seeds: {args.seeds}")
    printer.print(f"PE alphas: {args.pe_alphas}")
    
    # Setup GPU
    device = setup_gpu(args.gpu_id)
    
    # Set random seeds
    set_random_seeds(RANDOM_SEEDS['numpy'])
    
    try:
        # Stage 1: DAG Construction
        printer.section("STAGE 1: DAG CONSTRUCTION")
        dag_results = run_dag_construction(
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            method=args.method,
            thresholds=args.thresholds,
            verbose=args.verbose
        )
        
        # Stage 2: LLPE Generation
        printer.section("STAGE 2: LLPE GENERATION")
        llpe_results = run_llpe_generation(
            dag_results=dag_results,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            verbose=args.verbose
        )
        
        # Stage 3: TabTransformer Experiment
        printer.section("STAGE 3: TABTRANSFORMER EXPERIMENT")
        experiment_results = run_tabtransformer_experiment(
            dag_results=dag_results,
            llpe_results=llpe_results,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            config=config,
            device=device,
            verbose=args.verbose
        )
        
        # Final summary
        printer.section("EXPERIMENT COMPLETED SUCCESSFULLY")
        printer.print(f"Dataset: {args.dataset_name} v{args.dataset_version}")
        printer.print(f"DAG edges: {dag_results.get('best_stats', {}).get('total_edges', 'N/A')}")
        printer.print(f"PE dimensions: {llpe_results['positional_encodings'].shape[1]}")
        printer.print(f"Best alpha: {experiment_results['best_alpha']}")
        
        if experiment_results['task_info']['task_type'] == 'regression':
            printer.print(f"Best RMSE: {experiment_results['best_result']['rmse_mean']:.4f}")
            printer.print(f"Best R²: {experiment_results['best_result']['r2_mean']:.4f}")
        else:
            printer.print(f"Best Accuracy: {experiment_results['best_result']['acc_mean']:.4f}")
            printer.print(f"Best F1: {experiment_results['best_result']['f1_mean']:.4f}")
        
    except Exception as e:
        printer.print(f"Experiment failed with error: {str(e)}")
        import traceback
        if args.verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()