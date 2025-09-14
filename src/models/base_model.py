"""Base model class for algorithmic trading ML models.

This module provides a common interface and shared functionality for all
trading models in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path
from loguru import logger
import mlflow
import mlflow.sklearn
from datetime import datetime


class TradingModel(ABC, BaseEstimator):
    """Abstract base class for all trading models."""
    
    def __init__(self, 
                 model_name: str,
                 model_type: str = "classification",  # or "regression"
                 random_state: int = 42,
                 **kwargs):
        """Initialize base trading model.
        
        Args:
            model_name: Human readable name for the model
            model_type: Either 'classification' or 'regression'
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type
        self.random_state = random_state
        self.model_params = kwargs
        
        # Training metadata
        self.is_trained = False
        self.training_timestamp = None
        self.feature_names = []
        self.target_names = []
        self.training_samples = 0
        
        # Performance tracking
        self.train_scores = {}
        self.validation_scores = {}
        self.feature_importance = {}
        
        # MLflow tracking
        self.mlflow_run_id = None
        self.mlflow_experiment_name = f"trading_models_{model_type}"
    
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the underlying ML model instance.
        
        Returns:
            Scikit-learn compatible model instance
        """
        pass
    
    @abstractmethod
    def _validate_input(self, X: pd.DataFrame, y: pd.DataFrame = None) -> None:
        """Validate input data for the specific model type.
        
        Args:
            X: Feature matrix
            y: Target vector (optional for prediction)
            
        Raises:
            ValueError: If input validation fails
        """
        pass
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.DataFrame, 
            validation_data: Tuple[pd.DataFrame, pd.DataFrame] = None,
            sample_weight: np.ndarray = None,
            **fit_kwargs) -> 'TradingModel':
        """Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional (X_val, y_val) for validation scoring
            sample_weight: Optional sample weights
            **fit_kwargs: Additional arguments passed to model.fit()
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} on {len(X)} samples")
        
        # Validate inputs
        self._validate_input(X, y)
        
        # Store training metadata
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns) if isinstance(y, pd.DataFrame) else ['target']
        self.training_samples = len(X)
        self.training_timestamp = datetime.now()
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self._get_or_create_experiment()) as run:
            self.mlflow_run_id = run.info.run_id
            
            # Log model parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "model_type": self.model_type,
                "random_state": self.random_state,
                "training_samples": self.training_samples,
                **self.model_params
            })
            
            # Create and train model
            self.model = self._create_model()
            
            # Prepare target data
            y_train = y.iloc[:, 0].values if isinstance(y, pd.DataFrame) else y.values
            
            # Train the model
            if sample_weight is not None:
                self.model.fit(X.values, y_train, sample_weight=sample_weight, **fit_kwargs)
            else:
                self.model.fit(X.values, y_train, **fit_kwargs)
            
            self.is_trained = True
            
            # Calculate training scores
            self.train_scores = self._calculate_scores(X, y)
            mlflow.log_metrics({f"train_{k}": v for k, v in self.train_scores.items()})
            
            # Calculate validation scores if validation data provided
            if validation_data is not None:
                X_val, y_val = validation_data
                self.validation_scores = self._calculate_scores(X_val, y_val)
                mlflow.log_metrics({f"val_{k}": v for k, v in self.validation_scores.items()})
            
            # Extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))
                
                # Log top 10 most important features
                top_features = sorted(self.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, importance) in enumerate(top_features):
                    mlflow.log_metric(f"feature_importance_top_{i+1}", importance)
            
            # Log the model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info(f"Model training completed. MLflow run: {self.mlflow_run_id}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Model predictions
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        
        if set(X.columns) != set(self.feature_names):
            logger.warning("Feature columns don't match training features")
            # Ensure columns are in the same order as training
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models).
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Prediction probabilities
            
        Raises:
            ValueError: If model doesn't support probability predictions
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.model_name} doesn't support probability predictions")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self._validate_input(X)
        
        if set(X.columns) != set(self.feature_names):
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        return self.model.predict_proba(X.values)
    
    def _calculate_scores(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance scores for the model.
        
        Args:
            X: Feature matrix
            y: True targets
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(X)
        y_true = y.iloc[:, 0].values if isinstance(y, pd.DataFrame) else y.values
        
        scores = {}
        
        if self.model_type == "classification":
            # Classification metrics
            scores['accuracy'] = (predictions == y_true).mean()
            
            # Calculate precision, recall, f1 for each class
            if len(np.unique(y_true)) == 2:  # Binary classification
                from sklearn.metrics import precision_score, recall_score, f1_score
                scores['precision'] = precision_score(y_true, predictions, average='binary')
                scores['recall'] = recall_score(y_true, predictions, average='binary')
                scores['f1'] = f1_score(y_true, predictions, average='binary')
            
            # Get probabilities if available for additional metrics
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score, log_loss
                probas = self.predict_proba(X)
                
                if probas.shape[1] == 2:  # Binary classification
                    scores['auc'] = roc_auc_score(y_true, probas[:, 1])
                    scores['log_loss'] = log_loss(y_true, probas)
        
        else:  # Regression
            scores['mse'] = mean_squared_error(y_true, predictions)
            scores['rmse'] = np.sqrt(scores['mse'])
            scores['mae'] = mean_absolute_error(y_true, predictions)
            scores['r2'] = self.model.score(X.values, y_true)
        
        return scores
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get the top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.feature_importance:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))
            else:
                return {}
        
        return dict(sorted(self.feature_importance.items(), 
                          key=lambda x: x[1], reverse=True)[:top_n])
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'training_timestamp': self.training_timestamp,
            'train_scores': self.train_scores,
            'validation_scores': self.validation_scores,
            'feature_importance': self.feature_importance,
            'mlflow_run_id': self.mlflow_run_id
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'TradingModel':
        """Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.training_timestamp = model_data['training_timestamp']
        self.train_scores = model_data['train_scores']
        self.validation_scores = model_data['validation_scores']
        self.feature_importance = model_data['feature_importance']
        self.mlflow_run_id = model_data['mlflow_run_id']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def _get_or_create_experiment(self) -> str:
        """Get or create MLflow experiment for this model type.
        
        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"MLflow experiment creation failed: {e}")
            experiment_id = None
        
        return experiment_id
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model's key information.
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_timestamp': self.training_timestamp,
            'training_samples': self.training_samples,
            'num_features': len(self.feature_names),
            'num_targets': len(self.target_names),
            'mlflow_run_id': self.mlflow_run_id
        }
        
        if self.train_scores:
            summary['train_scores'] = self.train_scores
        
        if self.validation_scores:
            summary['validation_scores'] = self.validation_scores
        
        if self.feature_importance:
            top_features = self.get_feature_importance(top_n=5)
            summary['top_features'] = list(top_features.keys())
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name} ({self.model_type}, {status})"