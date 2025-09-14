"""Tests for feature engineering services.

Test suite for technical indicators and feature pipeline components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.technical_indicators import TechnicalIndicators
from src.feature_engineering.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Create 252 days of data (1 year)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    
    # Generate realistic stock price data
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns with small positive drift
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV data
    close_prices = np.array(prices)
    
    # High is close + random positive amount
    high_prices = close_prices + np.abs(np.random.normal(0, 1, 252))
    
    # Low is close - random positive amount  
    low_prices = close_prices - np.abs(np.random.normal(0, 1, 252))
    
    # Open is previous close + gap
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    gap = np.random.normal(0, 0.5, 252)
    open_prices = open_prices + gap
    
    # Volume
    base_volume = 1000000
    volume = np.random.lognormal(np.log(base_volume), 0.5, 252)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'symbol': 'TEST',
        'source': 'test',
        'timeframe': 'daily'
    }, index=dates)
    
    return df


@pytest.fixture
def technical_indicators():
    """Create TechnicalIndicators instance."""
    return TechnicalIndicators()


@pytest.fixture
def feature_pipeline():
    """Create FeaturePipeline instance."""
    return FeaturePipeline(cache_features=False)  # Disable caching for tests


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators class."""
    
    def test_validate_data_success(self, technical_indicators, sample_ohlcv_data):
        """Test data validation with valid data."""
        # Should not raise any exception
        technical_indicators.validate_data(sample_ohlcv_data)
    
    def test_validate_data_missing_columns(self, technical_indicators):
        """Test data validation with missing columns."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            technical_indicators.validate_data(df)
    
    def test_add_price_features(self, technical_indicators, sample_ohlcv_data):
        """Test adding basic price features."""
        result = technical_indicators.add_price_features(sample_ohlcv_data)
        
        # Check that new features were added
        expected_features = [
            'typical_price', 'hl_avg', 'price_range', 'body_size',
            'upper_shadow', 'lower_shadow', 'close_open_ratio',
            'high_close_ratio', 'low_close_ratio', 'gap', 'gap_pct'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
            
        # Check typical price calculation
        expected_typical = (sample_ohlcv_data['high'] + 
                          sample_ohlcv_data['low'] + 
                          sample_ohlcv_data['close']) / 3
        pd.testing.assert_series_equal(result['typical_price'], expected_typical, check_names=False)
        
        # Check that no infinite values were introduced
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()
    
    def test_add_returns_features(self, technical_indicators, sample_ohlcv_data):
        """Test adding return-based features."""
        result = technical_indicators.add_returns_features(sample_ohlcv_data)
        
        # Check that return features were added
        for period in [1, 5, 10, 20]:
            assert f'return_{period}d' in result.columns
            assert f'log_return_{period}d' in result.columns
            
        assert 'intraday_return' in result.columns
        assert 'overnight_return' in result.columns
        
        # Check 1-day return calculation
        expected_return_1d = sample_ohlcv_data['close'].pct_change(1)
        pd.testing.assert_series_equal(result['return_1d'], expected_return_1d, check_names=False)
    
    def test_add_volume_features(self, technical_indicators, sample_ohlcv_data):
        """Test adding volume-based features."""
        result = technical_indicators.add_volume_features(sample_ohlcv_data)
        
        expected_features = [
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'vwap', 'price_volume', 'volume_rsi'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
            
        # Check volume ratio calculation
        volume_sma_20 = sample_ohlcv_data['volume'].rolling(20).mean()
        expected_ratio = sample_ohlcv_data['volume'] / volume_sma_20
        pd.testing.assert_series_equal(result['volume_ratio'], expected_ratio, check_names=False)
    
    def test_add_trend_indicators(self, technical_indicators, sample_ohlcv_data):
        """Test adding trend indicators."""
        result = technical_indicators.add_trend_indicators(sample_ohlcv_data)
        
        # Check moving averages
        for period in [5, 10, 20, 50, 200]:
            assert f'sma_{period}' in result.columns
            assert f'ema_{period}' in result.columns
            
        # Check MACD components
        macd_features = ['macd', 'macd_signal', 'macd_histogram', 'macd_ratio']
        for feature in macd_features:
            assert feature in result.columns
            
        # Check ADX components  
        adx_features = ['adx', 'plus_di', 'minus_di']
        for feature in adx_features:
            assert feature in result.columns
    
    def test_add_momentum_indicators(self, technical_indicators, sample_ohlcv_data):
        """Test adding momentum indicators."""
        result = technical_indicators.add_momentum_indicators(sample_ohlcv_data)
        
        # Check RSI for different periods
        for period in [9, 14, 21]:
            assert f'rsi_{period}' in result.columns
            
        momentum_features = [
            'stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc_10', 'roc_20'
        ]
        
        for feature in momentum_features:
            assert feature in result.columns
    
    def test_add_volatility_indicators(self, technical_indicators, sample_ohlcv_data):
        """Test adding volatility indicators."""
        result = technical_indicators.add_volatility_indicators(sample_ohlcv_data)
        
        # Check Bollinger Bands
        bb_features = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        for feature in bb_features:
            assert feature in result.columns
            
        # Check ATR
        assert 'atr' in result.columns
        assert 'atr_ratio' in result.columns
        
        # Check historical volatility
        for period in [10, 20, 30]:
            assert f'volatility_{period}d' in result.columns
            
        # Check Keltner Channels
        kc_features = ['kc_upper', 'kc_lower', 'kc_position']
        for feature in kc_features:
            assert feature in result.columns
    
    def test_add_pattern_recognition(self, technical_indicators, sample_ohlcv_data):
        """Test adding candlestick patterns."""
        result = technical_indicators.add_pattern_recognition(sample_ohlcv_data)
        
        pattern_features = [
            'pattern_doji', 'pattern_hammer', 'pattern_hanging_man',
            'pattern_shooting_star', 'pattern_engulfing', 'pattern_harami',
            'pattern_dark_cloud_cover', 'pattern_piercing_pattern'
        ]
        
        for feature in pattern_features:
            assert feature in result.columns
    
    def test_add_rolling_statistics(self, technical_indicators, sample_ohlcv_data):
        """Test adding rolling statistics."""
        result = technical_indicators.add_rolling_statistics(sample_ohlcv_data)
        
        for window in [5, 10, 20, 50]:
            stat_features = [
                f'close_min_{window}', f'close_max_{window}',
                f'close_std_{window}', f'close_skew_{window}',
                f'close_kurt_{window}', f'close_position_{window}',
                f'volume_std_{window}'
            ]
            
            for feature in stat_features:
                assert feature in result.columns
    
    def test_compute_all_features(self, technical_indicators, sample_ohlcv_data):
        """Test computing all features at once."""
        result = technical_indicators.compute_all_features(sample_ohlcv_data)
        
        # Should have significantly more columns than original
        assert len(result.columns) > len(sample_ohlcv_data.columns) + 50
        
        # Should preserve original data
        for col in sample_ohlcv_data.columns:
            assert col in result.columns
            
        # Should not have infinite values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols]).any().any()
    
    def test_feature_names_consistency(self, technical_indicators):
        """Test that get_feature_names returns reasonable names."""
        feature_names = technical_indicators.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'rsi_14' in feature_names
        assert 'macd' in feature_names
    
    def test_feature_importance_categories(self, technical_indicators):
        """Test feature importance categorization."""
        categories = technical_indicators.get_feature_importance_categories()
        
        expected_categories = ['critical', 'important', 'useful', 'experimental']
        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)
            assert len(categories[category]) > 0


class TestFeaturePipeline:
    """Test cases for FeaturePipeline class."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeaturePipeline(
            cache_features=False,
            scaling_method='standard',
            imputation_strategy='mean'
        )
        
        assert not pipeline.cache_features
        assert pipeline.scaling_method == 'standard'
        assert pipeline.imputation_strategy == 'mean'
        assert not pipeline.is_fitted
    
    def test_invalid_scaling_method(self):
        """Test initialization with invalid scaling method."""
        with pytest.raises(ValueError, match="Unknown scaling method"):
            FeaturePipeline(scaling_method='invalid')
    
    def test_compute_features(self, feature_pipeline, sample_ohlcv_data):
        """Test feature computation."""
        result = feature_pipeline.compute_features(
            sample_ohlcv_data, 
            symbol='TEST',
            include_targets=True
        )
        
        # Should have many more columns than original
        assert len(result.columns) > len(sample_ohlcv_data.columns) + 80
        
        # Should have target variables
        target_cols = [col for col in result.columns if col.startswith('target_')]
        assert len(target_cols) > 0
        assert 'target_direction_1d' in result.columns
        assert 'target_return_1d' in result.columns
        
        # Should preserve index
        pd.testing.assert_index_equal(result.index, sample_ohlcv_data.index)
    
    def test_compute_features_no_targets(self, feature_pipeline, sample_ohlcv_data):
        """Test feature computation without targets."""
        result = feature_pipeline.compute_features(
            sample_ohlcv_data,
            symbol='TEST', 
            include_targets=False
        )
        
        # Should not have target variables
        target_cols = [col for col in result.columns if col.startswith('target_')]
        assert len(target_cols) == 0
    
    def test_add_regime_features(self, feature_pipeline, sample_ohlcv_data):
        """Test regime feature addition."""
        # First compute technical features to get required inputs
        features_df = feature_pipeline.technical_indicators.compute_all_features(sample_ohlcv_data)
        
        result = feature_pipeline._add_regime_features(features_df)
        
        regime_features = ['high_vol_regime', 'trending_regime', 'bull_regime']
        for feature in regime_features:
            if feature == 'high_vol_regime' and 'volatility_20d' in features_df.columns:
                assert feature in result.columns
            elif feature == 'trending_regime' and 'adx' in features_df.columns:
                assert feature in result.columns
            elif feature == 'bull_regime' and 'sma_20' in features_df.columns and 'sma_50' in features_df.columns:
                assert feature in result.columns
    
    def test_add_interaction_features(self, feature_pipeline, sample_ohlcv_data):
        """Test interaction feature addition."""
        # First compute technical features
        features_df = feature_pipeline.technical_indicators.compute_all_features(sample_ohlcv_data)
        
        result = feature_pipeline._add_interaction_features(features_df)
        
        # Check for expected interaction features
        possible_interactions = [
            'rsi_volume_interaction', 'momentum_vol_interaction', 'trend_momentum_interaction'
        ]
        
        # At least some interactions should be present given our technical features
        interaction_count = sum(1 for feature in possible_interactions if feature in result.columns)
        assert interaction_count > 0
    
    def test_prepare_ml_data(self, feature_pipeline, sample_ohlcv_data):
        """Test ML data preparation."""
        # Compute features first
        features_df = feature_pipeline.compute_features(sample_ohlcv_data, symbol='TEST')
        
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        # Check shapes and types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert len(X) == len(y)
        assert len(X.columns) > 50  # Should have many features
        assert len(y.columns) > 0   # Should have targets
        
        # Check that target columns are not in features
        target_cols = [col for col in features_df.columns if col.startswith('target_')]
        for target_col in target_cols:
            assert target_col not in X.columns
            
        # Check that metadata columns are not in features
        metadata_cols = ['symbol', 'source', 'timeframe']
        for meta_col in metadata_cols:
            assert meta_col not in X.columns
    
    def test_fit_transform_pipeline(self, feature_pipeline, sample_ohlcv_data):
        """Test fitting and transforming with pipeline."""
        # Compute features
        features_df = feature_pipeline.compute_features(sample_ohlcv_data, symbol='TEST')
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        # Fit and transform
        X_transformed = feature_pipeline.fit_transform(X)
        
        assert feature_pipeline.is_fitted
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == X.shape
        
        # Values should be different (scaled)
        if feature_pipeline.scaling_method != 'none':
            assert not np.allclose(X.fillna(0).values, X_transformed.fillna(0).values, atol=1e-3)
    
    def test_transform_unfitted_pipeline(self, feature_pipeline, sample_ohlcv_data):
        """Test transforming with unfitted pipeline."""
        features_df = feature_pipeline.compute_features(sample_ohlcv_data, symbol='TEST')
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        with pytest.raises(ValueError, match="Pipeline not fitted"):
            feature_pipeline.transform(X)
    
    def test_get_feature_statistics(self, feature_pipeline, sample_ohlcv_data):
        """Test feature statistics calculation."""
        features_df = feature_pipeline.compute_features(sample_ohlcv_data, symbol='TEST')
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        stats = feature_pipeline.get_feature_statistics(X)
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check structure of statistics
        for col_name, col_stats in stats.items():
            assert 'mean' in col_stats
            assert 'std' in col_stats
            assert 'min' in col_stats
            assert 'max' in col_stats
            assert 'missing_pct' in col_stats
            assert 'unique_values' in col_stats
    
    def test_detect_feature_issues(self, feature_pipeline, sample_ohlcv_data):
        """Test feature issue detection."""
        features_df = feature_pipeline.compute_features(sample_ohlcv_data, symbol='TEST')
        X, y = feature_pipeline.prepare_ml_data(features_df)
        
        issues = feature_pipeline.detect_feature_issues(X)
        
        expected_issue_types = ['high_missing', 'zero_variance', 'high_correlation', 'outliers']
        for issue_type in expected_issue_types:
            assert issue_type in issues
            assert isinstance(issues[issue_type], list)
    
    def test_handle_infinite_values(self, feature_pipeline):
        """Test infinite value handling."""
        # Create DataFrame with infinite values
        df = pd.DataFrame({
            'col1': [1, 2, np.inf, 4],
            'col2': [1, -np.inf, 3, 4],
            'col3': [1, 2, 3, 4]
        })
        
        result = feature_pipeline._handle_infinite_values(df)
        
        # Should not have any infinite values
        assert not np.isinf(result).any().any()
        
        # Should have NaN where infinite values were
        assert result.isna().sum().sum() == 2
    
    def test_pipeline_with_small_dataset(self, technical_indicators):
        """Test pipeline behavior with small dataset."""
        # Create very small dataset (less than 50 rows)
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        small_df = pd.DataFrame({
            'open': range(100, 110),
            'high': range(101, 111), 
            'low': range(99, 109),
            'close': range(100, 110),
            'volume': [1000] * 10,
            'symbol': 'SMALL',
            'source': 'test',
            'timeframe': 'daily'
        }, index=dates)
        
        # Should work but give warning
        result = technical_indicators.compute_all_features(small_df)
        
        # Should still compute features
        assert len(result.columns) > len(small_df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])