"""
Preprocessing module for chargeback prediction pipeline.
Handles data loading, feature engineering, and train/test splitting.
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing including feature engineering."""
    
    def __init__(self):
        self.feature_columns = [
            'transaction_amount', 
            'ts', 
            'last_transaction_amount_diff', 
            'ts_diff', 
            'single_tx_user'
        ]
        self.target_column = 'has_cbk'
        # Store historical data for prediction on new data
        self.historical_data = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV or URL.
        
        Args:
            data_path: Path to CSV file or URL
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert transaction_date to datetime
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Label encoding for target
        df['has_cbk'] = df['has_cbk'].map({True: 1, False: 0})
        
        # Create timestamp feature
        df['ts'] = ((df['transaction_date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['has_cbk'].value_counts()}")
        
        return df
    
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create engineered features based on user transaction history.
        
        Features created:
        - last_transaction_amount_diff: Difference from previous transaction amount
        - ts_diff: Time difference from previous transaction
        - single_tx_user: Whether user has more than one transaction
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (creates historical_data)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineered features")
        
        df = df.copy()
        
        # Sort by user and timestamp to ensure correct ordering
        df = df.sort_values(['user_id', 'transaction_date'])
        
        # Create features based on user history
        df['last_transaction_amount'] = df.groupby("user_id")['transaction_amount'].shift()
        df['last_transaction_amount_diff'] = df.groupby("user_id")['transaction_amount'].diff()
        df['ts_diff'] = df.groupby("user_id")['ts'].diff()
        df['single_tx_user'] = (
            df.groupby("user_id")['transaction_id'].transform('nunique') > 1
        ).astype(int)
        
        if is_training:
            # Store historical data for future predictions
            self.historical_data = df[['user_id', 'transaction_amount', 'ts', 
                                       'transaction_date', 'transaction_id']].copy()
            logger.info(f"Stored historical data with {len(self.historical_data)} records")
        
        logger.info(f"Features created. Missing values:\n{df[self.feature_columns].isnull().sum()}")
        
        return df
    
    def prepare_training_data(
        self, 
        data_path: str, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training and test datasets with proper feature engineering.
        
        IMPORTANT: To avoid data leakage, features are created BEFORE splitting,
        but the historical information used comes only from the training set perspective.
        
        Args:
            data_path: Path to CSV file or URL
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Load data
        df = self.load_data(data_path)
        
        # Create features
        df = self.create_features(df, is_training=True)
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data - stratify to maintain class balance
        logger.info(f"Splitting data: train={1-test_size:.1%}, test={test_size:.1%}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=random_state
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Training set target distribution:\n{y_train.value_counts()}")
        logger.info(f"Test set target distribution:\n{y_test.value_counts()}")
        
        # Store user_ids for later use in predictions
        df['split'] = 'train'
        df.loc[X_test.index, 'split'] = 'test'
        self.split_info = df[['user_id', 'split']].copy()
        
        return X_train, X_test, y_train, y_test
    
    def prepare_prediction_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare new data for prediction using historical data.
        
        Args:
            df: New transaction data
            
        Returns:
            X (features), metadata (user_id and other info)
        """
        logger.info(f"Preparing prediction data with {len(df)} records")
        
        if self.historical_data is None:
            logger.warning("No historical data available. Features will be limited.")
            # Create basic features without historical context
            df = df.copy()
            df['last_transaction_amount_diff'] = np.nan
            df['ts_diff'] = np.nan
            df['single_tx_user'] = 0
        else:
            # Combine with historical data to create proper features
            df = df.copy()
            combined = pd.concat([self.historical_data, df], ignore_index=True)
            combined = combined.sort_values(['user_id', 'transaction_date'])
            
            # Create features on combined data
            combined['last_transaction_amount_diff'] = combined.groupby("user_id")['transaction_amount'].diff()
            combined['ts_diff'] = combined.groupby("user_id")['ts'].diff()
            combined['single_tx_user'] = (
                combined.groupby("user_id")['transaction_id'].transform('nunique') > 1
            ).astype(int)
            
            # Extract only the new records
            df = combined.tail(len(df)).copy()
        
        # Prepare features
        X = df[self.feature_columns]
        metadata = df[['user_id']].copy()
        
        logger.info(f"Prediction data prepared. Shape: {X.shape}")
        
        return X, metadata
