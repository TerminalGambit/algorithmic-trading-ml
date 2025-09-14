"""Feature engineering pipeline for algorithmic trading.

This module provides a comprehensive feature engineering pipeline that transforms
raw market data into ML-ready features for trading strategy development.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from loguru import logger
import joblib
from pathlib import Path

from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.data_collection.cache_manager import cache_manager


class FeaturePipeline:
    """Complete feature engineering pipeline for trading data."""
    
    def __init__(self, 
                 cache_features: bool = True,
                 scaling_method: str = 'robust',
                 imputation_strategy: str = 'median'):
        """Initialize the feature engineering pipeline.
        
        Args:
            cache_features: Whether to cache computed features
            scaling_method: Scaling method ('standard', 'robust', 'minmax', 'none')
            imputation_strategy: Strategy for handling missing values
        """
        self.cache_features = cache_features
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        
        # Initialize components
        self.technical_indicators = TechnicalIndicators()
        
        # Scalers and preprocessors (fitted during training)
        self.scaler = self._get_scaler()
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.is_fitted = False
        
        # Feature metadata
        self.feature_names = []
        self.feature_importance_scores = {}
        self.feature_categories = {}
    
    def _get_scaler(self):
        """Get the appropriate scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'none': None
        }
        
        if self.scaling_method not in scalers:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
        return scalers[self.scaling_method]
    
    def _generate_cache_key(self, symbol: str, start_date: str = None, end_date: str = None) -> str:
        """Generate cache key for features."""
        key_parts = [symbol]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        return "_".join(key_parts)
    
    def _add_target_variables(self, df: pd.DataFrame, 
                            prediction_horizons: List[int] = None) -> pd.DataFrame:
        """Add target variables for supervised learning.
        
        Args:
            df: Input DataFrame with features
            prediction_horizons: List of days to predict [1, 5, 10]
            
        Returns:
            DataFrame with target variables added
        """
        if prediction_horizons is None:
            prediction_horizons = [1, 5, 10]
            
        result_df = df.copy()
        
        for horizon in prediction_horizons:
            # Price direction (classification target)
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            result_df[f'target_direction_{horizon}d'] = (future_return > 0).astype(int)
            
            # Actual return (regression target)
            result_df[f'target_return_{horizon}d'] = future_return
            
            # Volatility-adjusted return
            vol_window = min(20, len(df) // 4)
            if vol_window > 5:
                volatility = df['close'].pct_change().rolling(vol_window).std()
                result_df[f'target_vol_adj_return_{horizon}d'] = future_return / volatility
        
        return result_df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features.
        
        Args:
            df: Input DataFrame with price data
            
        Returns:
            DataFrame with regime features added
        """
        result_df = df.copy()
        
        # Volatility regime (high/low)
        if 'volatility_20d' in df.columns:
            vol_median = df['volatility_20d'].rolling(252).median()
            result_df['high_vol_regime'] = (df['volatility_20d'] > vol_median).astype(int)
        
        # Trend regime (trending/ranging)
        if 'adx' in df.columns:
            result_df['trending_regime'] = (df['adx'] > 25).astype(int)
        
        # Market direction regime
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            result_df['bull_regime'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        return result_df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between technical indicators.
        
        Args:
            df: Input DataFrame with technical features
            
        Returns:
            DataFrame with interaction features added
        """
        result_df = df.copy()
        
        # RSI-Volume interaction
        if 'rsi_14' in df.columns and 'volume_ratio' in df.columns:
            result_df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio']
        
        # Momentum-Volatility interaction
        if 'roc_10' in df.columns and 'atr_ratio' in df.columns:
            result_df['momentum_vol_interaction'] = df['roc_10'] * df['atr_ratio']
        
        # Trend-Momentum interaction
        if 'macd' in df.columns and 'adx' in df.columns:
            result_df['trend_momentum_interaction'] = df['macd'] * df['adx']
        
        return result_df
    
    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with infinite values handled
        """
        result_df = df.copy()
        
        # Replace infinite values with NaN
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log if many infinite values were found
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Replaced {inf_count} infinite values with NaN")
        
        return result_df
    
    def compute_features(self, df: pd.DataFrame, 
                        symbol: str,
                        include_targets: bool = True,
                        prediction_horizons: List[int] = None) -> pd.DataFrame:
        """Compute all features for the given data.
        
        Args:
            df: Input DataFrame with OHLCV data
            symbol: Stock symbol for caching
            include_targets: Whether to include target variables
            prediction_horizons: List of prediction horizons for targets
            
        Returns:
            DataFrame with computed features
        """
        # Check cache first
        if self.cache_features:
            cache_key = self._generate_cache_key(symbol)
            cached_features = cache_manager.get(
                source="feature_pipeline",
                data_type="computed_features",
                symbol=symbol,
                cache_key=cache_key
            )
            
            if cached_features is not None and len(cached_features) >= len(df) * 0.9:
                logger.info(f"Using cached features for {symbol}")
                return cached_features.head(len(df))
        
        logger.info(f"Computing features for {symbol} with {len(df)} data points")
        
        # Start with technical indicators
        features_df = self.technical_indicators.compute_all_features(df)
        
        # Add regime features
        features_df = self._add_regime_features(features_df)
        
        # Add interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Handle infinite values
        features_df = self._handle_infinite_values(features_df)
        
        # Add target variables if requested
        if include_targets:
            features_df = self._add_target_variables(features_df, prediction_horizons)
        
        # Store feature names for later use
        self.feature_names = [col for col in features_df.columns 
                            if col not in ['symbol', 'source', 'timeframe']]
        
        logger.info(f"Generated {len(self.feature_names)} total features for {symbol}")
        
        # Cache the results
        if self.cache_features:
            cache_manager.put(
                data=features_df,
                source="feature_pipeline",
                data_type="computed_features",
                metadata={"symbol": symbol, "features": len(self.feature_names)},
                symbol=symbol,
                cache_key=cache_key
            )
        
        return features_df
    
    def prepare_ml_data(self, features_df: pd.DataFrame,
                       target_columns: List[str] = None,
                       feature_columns: List[str] = None,
                       dropna: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for ML training by separating features and targets.
        
        Args:
            features_df: DataFrame with computed features
            target_columns: List of target column names
            feature_columns: List of feature column names (if None, auto-detect)
            dropna: Whether to drop rows with NaN values
            
        Returns:
            Tuple of (X, y) DataFrames
        """
        if target_columns is None:
            target_columns = [col for col in features_df.columns if col.startswith('target_')]
        
        if feature_columns is None:
            # Exclude metadata, target columns, and date columns
            exclude_cols = set(['symbol', 'source', 'timeframe'] + target_columns)
            feature_columns = [col for col in features_df.columns 
                             if col not in exclude_cols and not col.startswith('target_')]
        
        # Extract features and targets
        X = features_df[feature_columns].copy()
        y = features_df[target_columns].copy() if target_columns else pd.DataFrame()
        
        if dropna:
            # Only drop rows where all values are NaN to preserve as much data as possible
            X = X.dropna(how='all')
            if not y.empty:
                y = y.loc[X.index]
                # Drop rows where any target is NaN
                valid_targets = y.dropna().index
                X = X.loc[valid_targets]
                y = y.loc[valid_targets]
        
        logger.info(f"Prepared ML data: {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    
    def fit_preprocessors(self, X: pd.DataFrame) -> 'FeaturePipeline':
        """Fit the preprocessing components on training data.
        
        Args:
            X: Training features DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessing components")
        
        # Handle columns with all NaN values by filling with 0
        X_cleaned = X.copy()
        all_nan_cols = X_cleaned.columns[X_cleaned.isna().all()]
        if len(all_nan_cols) > 0:
            logger.warning(f"Found {len(all_nan_cols)} features with all NaN values, filling with 0")
            X_cleaned[all_nan_cols] = 0
        
        # Fit imputer on cleaned data
        self.imputer.fit(X_cleaned)
        
        # Fit scaler if specified
        if self.scaler is not None:
            X_imputed = self.imputer.transform(X_cleaned)
            self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        logger.info("Preprocessing components fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to features.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed features DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit_preprocessors() first.")
        
        # Handle columns with all NaN values by filling with 0 (same as in fit)
        X_cleaned = X.copy()
        all_nan_cols = X_cleaned.columns[X_cleaned.isna().all()]
        if len(all_nan_cols) > 0:
            X_cleaned[all_nan_cols] = 0
        
        # Apply imputation
        X_transformed = self.imputer.transform(X_cleaned)
        
        # Apply scaling if specified
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        
        # Convert back to DataFrame with original index and columns
        result_df = pd.DataFrame(X_transformed, index=X.index, columns=X.columns)
        
        return result_df
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessors and transform the data in one step.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Transformed features DataFrame
        """
        return self.fit_preprocessors(X).transform(X)
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by importance categories.
        
        Returns:
            Dictionary mapping importance level to feature lists
        """
        return self.technical_indicators.get_feature_importance_categories()
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """Save the fitted pipeline to disk.
        
        Args:
            filepath: Path to save the pipeline
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted pipeline")
        
        pipeline_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'scaling_method': self.scaling_method,
            'imputation_strategy': self.imputation_strategy,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: Union[str, Path]) -> 'FeaturePipeline':
        """Load a fitted pipeline from disk.
        
        Args:
            filepath: Path to load the pipeline from
            
        Returns:
            Self for method chaining
        """
        pipeline_data = joblib.load(filepath)
        
        self.scaler = pipeline_data['scaler']
        self.imputer = pipeline_data['imputer']
        self.feature_names = pipeline_data['feature_names']
        self.scaling_method = pipeline_data['scaling_method']
        self.imputation_strategy = pipeline_data['imputation_strategy']
        self.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return self
    
    def get_feature_statistics(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'min': X[col].min(),
                'max': X[col].max(),
                'missing_pct': X[col].isna().mean() * 100,
                'unique_values': X[col].nunique()
            }
        
        return stats
    
    def detect_feature_issues(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect potential issues with features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Dictionary mapping issue types to affected features
        """
        issues = {
            'high_missing': [],
            'zero_variance': [],
            'high_correlation': [],
            'outliers': []
        }
        
        # High missing values (>50%)
        for col in X.columns:
            if X[col].isna().mean() > 0.5:
                issues['high_missing'].append(col)
        
        # Zero variance features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].std() == 0:
                issues['zero_variance'].append(col)
        
        # High correlation pairs (>0.95)
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}")
            
            issues['high_correlation'] = high_corr_pairs
        
        return issues