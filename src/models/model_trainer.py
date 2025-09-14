"""Model training pipeline with cross-validation and experiment tracking.

This module provides functionality for training trading models with proper
time series validation and MLflow experiment tracking.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
from loguru import logger
from pathlib import Path
import json
from datetime import datetime

from src.models.base_model import TradingModel
from src.feature_engineering.feature_pipeline import FeaturePipeline


class ModelTrainer:
    """Training pipeline for trading models with time series cross-validation."""
    
    def __init__(self, 
                 mlflow_tracking_uri: Optional[str] = None,
                 experiment_name: str = "trading_model_training"):
        """Initialize the model trainer.
        
        Args:
            mlflow_tracking_uri: URI for MLflow tracking server
            experiment_name: Name for the MLflow experiment
        """
        self.experiment_name = experiment_name
        
        # Set MLflow tracking URI if provided
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Set or create experiment
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
    
    def train_with_validation(self,
                            model: TradingModel,
                            X: pd.DataFrame,
                            y: pd.DataFrame,
                            validation_split: float = 0.2,
                            shuffle: bool = False) -> Dict[str, Any]:
        """Train a model with a simple train/validation split.
        
        Args:
            model: TradingModel instance to train
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data to use for validation
            shuffle: Whether to shuffle data before splitting (not recommended for time series)
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model.model_name} with {validation_split:.0%} validation split")
        
        # Split data
        if shuffle:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, shuffle=True
            )
        else:
            # Time series split - use latest data for validation
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        
        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Generate predictions for detailed analysis
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        # Compile results
        results = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_scores': model.train_scores,
            'validation_scores': model.validation_scores,
            'feature_importance': model.get_feature_importance(top_n=20),
            'mlflow_run_id': model.mlflow_run_id,
            'training_timestamp': model.training_timestamp
        }
        
        # Add classification-specific results
        if model.model_type == "classification":
            y_train_true = y_train.iloc[:, 0].values if isinstance(y_train, pd.DataFrame) else y_train.values
            y_val_true = y_val.iloc[:, 0].values if isinstance(y_val, pd.DataFrame) else y_val.values
            
            results.update({
                'train_classification_report': classification_report(y_train_true, train_preds, output_dict=True),
                'val_classification_report': classification_report(y_val_true, val_preds, output_dict=True),
                'train_confusion_matrix': confusion_matrix(y_train_true, train_preds).tolist(),
                'val_confusion_matrix': confusion_matrix(y_val_true, val_preds).tolist(),
            })
        
        return results
    
    def cross_validate(self,
                      model: TradingModel,
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      cv_folds: int = 5,
                      gap: int = 0) -> Dict[str, Any]:
        """Perform time series cross-validation on the model.
        
        Args:
            model: TradingModel instance to validate
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            gap: Gap between train and test sets (in time steps)
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold time series cross-validation for {model.model_name}")
        
        # Use TimeSeriesSplit for proper time series validation
        tscv = TimeSeriesSplit(n_splits=cv_folds, gap=gap)
        
        cv_scores = []
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold_idx + 1}/{cv_folds}")
            
            # Split data for this fold
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Create a fresh model instance for this fold
            fold_model = model.__class__(**model.model_params, 
                                       model_type=model.model_type,
                                       random_state=model.random_state)
            
            # Train on this fold
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Calculate scores for this fold
            train_scores = fold_model._calculate_scores(X_train_fold, y_train_fold)
            test_scores = fold_model._calculate_scores(X_test_fold, y_test_fold)
            
            fold_result = {
                'fold': fold_idx + 1,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'train_scores': train_scores,
                'test_scores': test_scores
            }
            
            fold_results.append(fold_result)
            cv_scores.append(test_scores)
            
            logger.debug(f"Fold {fold_idx + 1} completed. Test scores: {test_scores}")
        
        # Aggregate results across folds
        if model.model_type == "classification":
            primary_metric = 'accuracy'
        else:
            primary_metric = 'r2'
        
        # Calculate mean and std for each metric
        aggregated_scores = {}
        for metric in cv_scores[0].keys():
            scores = [fold[metric] for fold in cv_scores]
            aggregated_scores[f'{metric}_mean'] = np.mean(scores)
            aggregated_scores[f'{metric}_std'] = np.std(scores)
        
        cv_results = {
            'model_name': model.model_name,
            'model_type': model.model_type,
            'cv_folds': cv_folds,
            'gap': gap,
            'primary_metric': primary_metric,
            'primary_score_mean': aggregated_scores[f'{primary_metric}_mean'],
            'primary_score_std': aggregated_scores[f'{primary_metric}_std'],
            'aggregated_scores': aggregated_scores,
            'fold_results': fold_results,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Cross-validation completed. {primary_metric}: "
                   f"{cv_results['primary_score_mean']:.4f} ± {cv_results['primary_score_std']:.4f}")
        
        return cv_results
    
    def train_and_validate_pipeline(self,
                                  model: TradingModel,
                                  feature_pipeline: FeaturePipeline,
                                  data: pd.DataFrame,
                                  target_column: str,
                                  validation_split: float = 0.2) -> Dict[str, Any]:
        """Complete training pipeline including feature engineering.
        
        Args:
            model: TradingModel instance to train
            feature_pipeline: FeaturePipeline for feature engineering
            data: Raw OHLCV data
            target_column: Name of the target column in engineered features
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with complete pipeline results
        """
        logger.info("Starting complete training pipeline with feature engineering")
        
        # Extract symbol for feature computation
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Generate features
        logger.info("Computing features...")
        features_df = feature_pipeline.compute_features(data, symbol=symbol, include_targets=True)
        
        # Prepare ML data
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        # Select specific target if multiple targets available
        if target_column not in y.columns:
            available_targets = list(y.columns)
            logger.warning(f"Target '{target_column}' not found. Available targets: {available_targets}")
            target_column = available_targets[0]  # Use first available target
        
        y_target = y[[target_column]]
        
        # Preprocess features
        logger.info("Preprocessing features...")
        feature_pipeline.fit_preprocessors(X)
        X_processed = feature_pipeline.transform(X)
        
        # Train with validation
        results = self.train_with_validation(
            model=model,
            X=X_processed,
            y=y_target,
            validation_split=validation_split,
            shuffle=False  # Keep time series order
        )
        
        # Add pipeline metadata
        results.update({
            'target_column': target_column,
            'total_features': len(X.columns),
            'total_samples': len(X),
            'symbol': symbol,
            'pipeline_preprocessing': {
                'scaling_method': feature_pipeline.scaling_method,
                'imputation_strategy': feature_pipeline.imputation_strategy
            }
        })
        
        return results
    
    def save_training_results(self, results: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save training results to JSON file.
        
        Args:
            results: Training results dictionary
            filepath: Path to save results
        """
        # Convert non-JSON serializable objects
        results_copy = results.copy()
        
        # Convert datetime objects
        if 'training_timestamp' in results_copy:
            results_copy['training_timestamp'] = results_copy['training_timestamp'].isoformat()
        
        if 'timestamp' in results_copy:
            results_copy['timestamp'] = results_copy['timestamp'].isoformat()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {filepath}")
    
    def load_training_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load training results from JSON file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Training results dictionary
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Training results loaded from {filepath}")
        return results
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare results from multiple model training runs.
        
        Args:
            results_list: List of training results dictionaries
            
        Returns:
            DataFrame comparing model performance
        """
        comparison_data = []
        
        for results in results_list:
            if 'validation_scores' in results and results['validation_scores']:
                row = {
                    'model_name': results['model_name'],
                    'model_type': results['model_type'],
                    'train_samples': results.get('train_samples', 0),
                    'val_samples': results.get('val_samples', 0),
                    'mlflow_run_id': results.get('mlflow_run_id', ''),
                }
                
                # Add validation scores
                for metric, value in results['validation_scores'].items():
                    row[f'val_{metric}'] = value
                
                # Add training scores
                if 'train_scores' in results:
                    for metric, value in results['train_scores'].items():
                        row[f'train_{metric}'] = value
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_experiment_runs(self, limit: int = 10) -> pd.DataFrame:
        """Get recent MLflow runs for the current experiment.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            DataFrame with run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{self.experiment_name}' not found")
                return pd.DataFrame()
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            return runs
            
        except Exception as e:
            logger.warning(f"Could not retrieve MLflow runs: {e}")
            return pd.DataFrame()