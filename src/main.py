"""
Main orchestrator for chargeback prediction pipeline.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from preprocess import DataPreprocessor
from model import ChargebackModel
from typing import Optional

# Initial basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_run_directory():
    """
    Create a timestamped directory for the current run.
    
    Returns:
        Path to the run directory and timestamp string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the project root (parent of src/)
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir, timestamp


def setup_logging_with_file(run_dir: Path):
    """
    Setup logging to both console and file.
    
    Args:
        run_dir: Directory to save log file
    """
    log_file = run_dir / "training.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Add file handler to root logger
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return log_file


def train_pipeline(
    data_path: str,
    test_size: float = 0.2,
    fp_cost: float = -1.0,
    fn_cost: float = -5.0,
    artifacts_dir: str = "artefacts",
    output_path: Optional[str] = None,
    output_format: str = "csv"
):
    """
    Complete training pipeline for chargeback prediction.
    
    Args:
        data_path: Path to training data CSV
        test_size: Proportion of data for test set (default: 0.2)
        fp_cost: Cost of false positive (default: -1)
        fn_cost: Cost of false negative (default: -5)
        artifacts_dir: Directory to save model artifacts
        output_path: Path to save predictions (optional)
        output_format: Output format - 'csv' or 'excel' (default: 'csv')
    """
    # Setup run directory with timestamp
    run_dir, timestamp = setup_run_directory()
    
    # Setup logging to file
    log_file = setup_logging_with_file(run_dir)
    
    logger.info("="*80)
    logger.info("CHARGEBACK PREDICTION - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"False Positive cost: {fp_cost}")
    logger.info(f"False Negative cost: {fn_cost}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info("="*80 + "\n")
    
    # Step 1: Preprocessing
    logger.info("STEP 1: Data Preprocessing")
    logger.info("-"*80)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(
        data_path=data_path,
        test_size=test_size
    )
    
    # Save preprocessor
    import pickle
    preprocessor_path = Path(artifacts_dir) / "preprocessor.pkl"
    Path(artifacts_dir).mkdir(exist_ok=True)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor saved to {preprocessor_path}\n")
    
    # Step 2: Model Training
    logger.info("STEP 2: Model Training")
    logger.info("-"*80)
    model = ChargebackModel(
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        artifacts_dir=artifacts_dir
    )
    
    results = model.train(X_train, y_train, X_test, y_test)
    
    # Step 3: Generate predictions on test set (always save to run directory)
    logger.info("\nSTEP 3: Generating Predictions")
    logger.info("-"*80)
    
    predictions, probabilities = model.predict(X_test)
    
    # Get user_ids for test set
    # Load original data to get user_ids
    df_original = preprocessor.load_data(data_path)
    df_original = preprocessor.create_features(df_original, is_training=False)
    test_user_ids = df_original.loc[X_test.index, 'user_id']
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'user_id': test_user_ids.values,
        'prob_cbk': probabilities,
        'pred': predictions
    })
    
    # Save results to run directory with timestamp in filename
    results_filename = f"results_{timestamp}.csv"
    if output_format.lower() == 'excel':
        results_filename = f"results_{timestamp}.xlsx"
        results_path = run_dir / results_filename
        results_df.to_excel(results_path, index=False)
    else:
        results_path = run_dir / results_filename
        results_df.to_csv(results_path, index=False)
    
    logger.info(f"Predictions saved to {results_path}")
    logger.info(f"Total predictions: {len(results_df)}")
    logger.info(f"Predicted chargebacks: {predictions.sum()}")
    logger.info(f"Average chargeback probability: {probabilities.mean():.4f}")
    
    # Also save to custom output path if provided
    if output_path:
        custom_output = Path(output_path)
        if output_format.lower() == 'excel':
            custom_output = custom_output.with_suffix('.xlsx')
            results_df.to_excel(custom_output, index=False)
        else:
            custom_output = custom_output.with_suffix('.csv')
            results_df.to_csv(custom_output, index=False)
        logger.info(f"Predictions also saved to {custom_output}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"All outputs saved to: {run_dir}")
    logger.info("="*80)
    
    return results


def predict_pipeline(
    data_path: str,
    artifacts_dir: str = "artefacts",
    output_path: str = "predictions",
    output_format: str = "csv"
):
    """
    Prediction pipeline for new data.
    
    Args:
        data_path: Path to new data CSV
        artifacts_dir: Directory with saved model artifacts
        output_path: Path to save predictions
        output_format: Output format - 'csv' or 'excel'
    """
    logger.info("="*80)
    logger.info("CHARGEBACK PREDICTION - PREDICTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Artifacts directory: {artifacts_dir}")
    logger.info("="*80 + "\n")
    
    # Load preprocessor
    logger.info("Loading preprocessor...")
    import pickle
    preprocessor_path = Path(artifacts_dir) / "preprocessor.pkl"
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded\n")
    
    # Load model
    logger.info("Loading model...")
    model = ChargebackModel(artifacts_dir=artifacts_dir)
    model.load_artifacts()
    logger.info("Model loaded\n")
    
    # Load and preprocess new data
    logger.info("Preprocessing new data...")
    df = preprocessor.load_data(data_path)
    
    # For prediction, we need to properly handle the features
    # Create a temporary version with features for prediction
    df_temp = df.copy()
    df_temp = preprocessor.create_features(df_temp, is_training=False)
    
    X, metadata = preprocessor.prepare_prediction_data(df_temp)
    logger.info(f"Data preprocessed: {len(X)} records\n")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions, probabilities = model.predict(X)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'user_id': metadata['user_id'].values,
        'prob_cbk': probabilities,
        'pred': predictions
    })
    
    # Save results
    output_path = Path(output_path)
    if output_format.lower() == 'excel':
        output_path = output_path.with_suffix('.xlsx')
        results_df.to_excel(output_path, index=False)
    else:
        output_path = output_path.with_suffix('.csv')
        results_df.to_csv(output_path, index=False)
    
    logger.info(f"\nPredictions saved to {output_path}")
    logger.info(f"Total predictions: {len(results_df)}")
    logger.info(f"Predicted chargebacks: {predictions.sum()}")
    logger.info(f"Average chargeback probability: {probabilities.mean():.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    
    return results_df


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Chargeback Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with default parameters
  python main.py train --data data.csv
  
  # Train with custom costs and output predictions
  python main.py train --data data.csv --fp-cost -2 --fn-cost -10 --output results.csv
  
  # Train with Excel output
  python main.py train --data data.csv --output results --format excel
  
  # Make predictions on new data
  python main.py predict --data new_data.csv --output predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or predict')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', required=True, help='Path to training data CSV')
    train_parser.add_argument('--test-size', type=float, default=0.2, 
                             help='Test set proportion (default: 0.2)')
    train_parser.add_argument('--fp-cost', type=float, default=-1.0,
                             help='False positive cost (default: -1)')
    train_parser.add_argument('--fn-cost', type=float, default=-5.0,
                             help='False negative cost (default: -5)')
    train_parser.add_argument('--artifacts-dir', default='artefacts',
                             help='Directory to save artifacts (default: artefacts)')
    train_parser.add_argument('--output', help='Path to save test predictions (optional)')
    train_parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                             help='Output format (default: csv)')
    
    # Prediction mode
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new data')
    predict_parser.add_argument('--data', required=True, help='Path to new data CSV')
    predict_parser.add_argument('--artifacts-dir', default='artefacts',
                               help='Directory with saved artifacts (default: artefacts)')
    predict_parser.add_argument('--output', default='predictions',
                               help='Path to save predictions (default: predictions)')
    predict_parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                               help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline(
            data_path=args.data,
            test_size=args.test_size,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
            artifacts_dir=args.artifacts_dir,
            output_path=args.output,
            output_format=args.format
        )
    elif args.mode == 'predict':
        predict_pipeline(
            data_path=args.data,
            artifacts_dir=args.artifacts_dir,
            output_path=args.output,
            output_format=args.format
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
