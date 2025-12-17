"""
Model training module for chargeback prediction.
Implements XGBoost with cost-sensitive threshold tuning.
"""
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChargebackModel:
    """XGBoost model with cost-sensitive threshold tuning for chargeback prediction."""
    
    def __init__(
        self, 
        fp_cost: float = -1.0,
        fn_cost: float = -5.0,
        artifacts_dir: str = "artefacts"
    ):
        """
        Initialize the chargeback prediction model.
        
        Args:
            fp_cost: Cost of false positive (default: -1)
            fn_cost: Cost of false negative (default: -5)
            artifacts_dir: Directory to save model artifacts
        """
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        self.base_model = None
        self.tuned_model = None
        self.pos_label = 1
        self.neg_label = 0
        
        logger.info(f"Model initialized with FP cost={fp_cost}, FN cost={fn_cost}")
    
    def _credit_gain_score(self, y, y_pred):
        """Calculate credit gain score based on business costs."""
        cm = confusion_matrix(y, y_pred, labels=[self.neg_label, self.pos_label])
        gain_matrix = np.array([
            [0, self.fp_cost],  # True negative, False positive
            [self.fn_cost, 0],  # False negative, True positive
        ])
        return np.sum(cm * gain_matrix)
        
    def _create_cost_scorer(self):
        """Create custom scorer based on business costs."""
        return make_scorer(self._credit_gain_score)
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict:
        """
        Train XGBoost model with cost-sensitive threshold tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional, for evaluation)
            y_test: Test labels (optional, for evaluation)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("="*80)
        logger.info("Starting model training")
        logger.info("="*80)
        
        # Calculate class ratio for scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        cls_ratio = neg_count / pos_count
        
        logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
        logger.info(f"Scale pos weight: {cls_ratio:.2f}")
        
        # Train base XGBoost model
        logger.info("Training base XGBoost model...")
        self.base_model = xgb.XGBClassifier(
            tree_method="exact",
            scale_pos_weight=cls_ratio,
            random_state=42
        )
        self.base_model.fit(X_train, y_train)
        logger.info("Base model training completed")
        
        # Create cost-sensitive scorer
        cost_scorer = self._create_cost_scorer()
        
        # Tune threshold using cross-validation
        logger.info("Tuning decision threshold using cross-validation...")
        self.tuned_model = TunedThresholdClassifierCV(
            estimator=self.base_model,
            scoring=cost_scorer,
            store_cv_results=True,
            cv=5
        )
        self.tuned_model.fit(X_train, y_train)
        
        logger.info(f"Optimal threshold found: {self.tuned_model.best_threshold_:.4f}")
        logger.info(f"Best CV score: {self.tuned_model.best_score_:.2f}")
        
        # Evaluate on training set
        logger.info("\n" + "="*80)
        logger.info("TRAINING SET EVALUATION")
        logger.info("="*80)
        train_metrics = self._evaluate(X_train, y_train, dataset_name="Training")
        
        # Evaluate on test set if provided
        test_metrics = {}
        if X_test is not None and y_test is not None:
            logger.info("\n" + "="*80)
            logger.info("TEST SET EVALUATION")
            logger.info("="*80)
            test_metrics = self._evaluate(X_test, y_test, dataset_name="Test")
        
        # Save model artifacts
        self._save_artifacts()
        
        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_threshold": self.tuned_model.best_threshold_,
            "best_cv_score": self.tuned_model.best_score_
        }
    
    def _evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "") -> Dict:
        """
        Evaluate model on given dataset.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_pred = self.tuned_model.predict(X)
        y_proba = self.tuned_model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y, y_pred, pos_label=self.pos_label, zero_division=0)
        recall = recall_score(y, y_pred, pos_label=self.pos_label, zero_division=0)
        f1 = f1_score(y, y_pred, pos_label=self.pos_label, zero_division=0)
        roc_auc = roc_auc_score(y, y_proba)
        
        # Cost-sensitive metric
        cm = confusion_matrix(y, y_pred, labels=[self.neg_label, self.pos_label])
        gain_matrix = np.array([
            [0, self.fp_cost],
            [self.fn_cost, 0],
        ])
        credit_gain = np.sum(cm * gain_matrix)
        
        # Log results
        logger.info(f"\n{dataset_name} Set Metrics:")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Credit Gain (Business Metric): {credit_gain:.2f}")
        
        logger.info(f"\nConfusion Matrix ({dataset_name}):")
        logger.info(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        logger.info(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
        
        logger.info(f"\nClassification Report ({dataset_name}):")
        logger.info("\n" + classification_report(y, y_pred, target_names=['No Chargeback', 'Chargeback']))
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "credit_gain": credit_gain,
            "confusion_matrix": cm.tolist()
        }
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (predictions, probabilities for positive class)
        """
        if self.tuned_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.tuned_model.predict(X)
        probabilities = self.tuned_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def _save_artifacts(self):
        """Save model artifacts to disk."""
        logger.info(f"\nSaving model artifacts to {self.artifacts_dir}")
        
        # Save tuned model
        tuned_model_path = self.artifacts_dir / "tuned_model.pkl"
        with open(tuned_model_path, 'wb') as f:
            pickle.dump(self.tuned_model, f)
        logger.info(f"  Saved tuned model: {tuned_model_path}")
        
        # Save base model
        base_model_path = self.artifacts_dir / "base_model.pkl"
        with open(base_model_path, 'wb') as f:
            pickle.dump(self.base_model, f)
        logger.info(f"  Saved base model: {base_model_path}")
        
        # Save configuration
        config = {
            "fp_cost": self.fp_cost,
            "fn_cost": self.fn_cost,
            "best_threshold": self.tuned_model.best_threshold_,
            "best_cv_score": self.tuned_model.best_score_
        }
        config_path = self.artifacts_dir / "model_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        logger.info(f"  Saved configuration: {config_path}")
        
        logger.info("All artifacts saved successfully")
    
    def load_artifacts(self):
        """Load model artifacts from disk."""
        logger.info(f"Loading model artifacts from {self.artifacts_dir}")
        
        # Load tuned model
        tuned_model_path = self.artifacts_dir / "tuned_model.pkl"
        with open(tuned_model_path, 'rb') as f:
            self.tuned_model = pickle.load(f)
        logger.info(f"  Loaded tuned model: {tuned_model_path}")
        
        # Load base model
        base_model_path = self.artifacts_dir / "base_model.pkl"
        with open(base_model_path, 'rb') as f:
            self.base_model = pickle.load(f)
        logger.info(f"  Loaded base model: {base_model_path}")
        
        # Load configuration
        config_path = self.artifacts_dir / "model_config.pkl"
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        self.fp_cost = config["fp_cost"]
        self.fn_cost = config["fn_cost"]
        logger.info(f"  Loaded configuration: {config_path}")
        
        logger.info(f"Model loaded with threshold: {self.tuned_model.best_threshold_:.4f}")
