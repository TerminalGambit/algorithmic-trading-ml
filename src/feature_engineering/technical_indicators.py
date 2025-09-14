"""Technical indicators calculation module.

This module provides comprehensive technical analysis indicators using TA-Lib
and custom implementations for feature engineering in trading strategies.
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import talib
from loguru import logger


class TechnicalIndicators:
    """Comprehensive technical indicators calculator for feature engineering."""
    
    def __init__(self):
        """Initialize the technical indicators calculator."""
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(df) < 50:
            logger.warning(f"DataFrame has only {len(df)} rows. Some indicators may not be reliable.")
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-based features.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price features
        """
        result_df = df.copy()
        
        # Basic price features
        result_df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        result_df['hl_avg'] = (df['high'] + df['low']) / 2
        result_df['price_range'] = df['high'] - df['low']
        result_df['body_size'] = abs(df['close'] - df['open'])
        result_df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        result_df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Price ratios
        result_df['close_open_ratio'] = df['close'] / df['open']
        result_df['high_close_ratio'] = df['high'] / df['close']
        result_df['low_close_ratio'] = df['low'] / df['close']
        
        # Gap analysis
        result_df['gap'] = df['open'] - df['close'].shift(1)
        result_df['gap_pct'] = result_df['gap'] / df['close'].shift(1)
        
        return result_df
    
    def add_returns_features(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Add return-based features.
        
        Args:
            df: Input DataFrame with price data
            periods: List of periods for return calculation [1, 5, 10, 20]
            
        Returns:
            DataFrame with added return features
        """
        if periods is None:
            periods = [1, 5, 10, 20]
            
        result_df = df.copy()
        
        for period in periods:
            # Simple returns
            result_df[f'return_{period}d'] = df['close'].pct_change(period)
            result_df[f'return_{period}d_vol'] = result_df[f'return_{period}d'].rolling(20).std()
            
            # Log returns (more statistically stable)
            result_df[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
            
        # Intraday returns
        result_df['intraday_return'] = (df['close'] - df['open']) / df['open']
        result_df['overnight_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return result_df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume features
        """
        result_df = df.copy()
        
        # Compute typical price if not present
        if 'typical_price' not in result_df.columns:
            result_df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Volume moving averages
        result_df['volume_sma_10'] = df['volume'].rolling(10).mean()
        result_df['volume_sma_20'] = df['volume'].rolling(20).mean()
        result_df['volume_ratio'] = df['volume'] / result_df['volume_sma_20']
        
        # Volume-price indicators
        result_df['vwap'] = (result_df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        result_df['price_volume'] = df['close'] * df['volume']
        
        # Volume oscillators - ensure array is float64
        volume_values = df['volume'].astype(np.float64).values
        if len(volume_values) >= 14:  # RSI needs at least 14 points
            result_df['volume_rsi'] = talib.RSI(volume_values, timeperiod=14)
        else:
            result_df['volume_rsi'] = np.nan
        
        return result_df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added trend indicators
        """
        result_df = df.copy()
        
        # Moving averages
        close_values = df['close'].astype(np.float64).values
        for period in [5, 10, 20, 50, 200]:
            result_df[f'sma_{period}'] = talib.SMA(close_values, timeperiod=period)
            result_df[f'ema_{period}'] = talib.EMA(close_values, timeperiod=period)
        
        # Moving average ratios and crossovers
        result_df['sma_ratio_20_50'] = result_df['sma_20'] / result_df['sma_50']
        result_df['ema_ratio_10_20'] = result_df['ema_10'] / result_df['ema_20']
        result_df['price_sma20_ratio'] = df['close'] / result_df['sma_20']
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_values)
        result_df['macd'] = macd
        result_df['macd_signal'] = macd_signal
        result_df['macd_histogram'] = macd_hist
        result_df['macd_ratio'] = macd / macd_signal
        
        # ADX (Average Directional Index)
        high_values = df['high'].astype(np.float64).values
        low_values = df['low'].astype(np.float64).values
        result_df['adx'] = talib.ADX(high_values, low_values, close_values)
        result_df['plus_di'] = talib.PLUS_DI(high_values, low_values, close_values)
        result_df['minus_di'] = talib.MINUS_DI(high_values, low_values, close_values)
        
        return result_df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillator indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum indicators
        """
        result_df = df.copy()
        
        # RSI (Relative Strength Index)
        close_values = df['close'].astype(np.float64).values
        for period in [9, 14, 21]:
            result_df[f'rsi_{period}'] = talib.RSI(close_values, timeperiod=period)
        
        # Stochastic
        high_values = df['high'].astype(np.float64).values
        low_values = df['low'].astype(np.float64).values
        slowk, slowd = talib.STOCH(high_values, low_values, close_values)
        result_df['stoch_k'] = slowk
        result_df['stoch_d'] = slowd
        
        # Williams %R
        result_df['williams_r'] = talib.WILLR(high_values, low_values, close_values)
        
        # CCI (Commodity Channel Index)
        result_df['cci'] = talib.CCI(high_values, low_values, close_values)
        
        # Rate of Change
        result_df['roc_10'] = talib.ROC(close_values, timeperiod=10)
        result_df['roc_20'] = talib.ROC(close_values, timeperiod=20)
        
        return result_df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volatility indicators
        """
        result_df = df.copy()
        
        # Bollinger Bands
        close_values = df['close'].astype(np.float64).values
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_values)
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        result_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Average True Range (ATR)
        high_values = df['high'].astype(np.float64).values
        low_values = df['low'].astype(np.float64).values
        result_df['atr'] = talib.ATR(high_values, low_values, close_values)
        result_df['atr_ratio'] = result_df['atr'] / df['close']
        
        # Historical volatility
        for period in [10, 20, 30]:
            returns = df['close'].pct_change()
            result_df[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Keltner Channels - compute EMA 20 if not present
        if 'ema_20' not in result_df.columns:
            result_df['ema_20'] = talib.EMA(close_values, timeperiod=20)
        
        ema_20 = result_df['ema_20']
        kc_upper = ema_20 + (2 * result_df['atr'])
        kc_lower = ema_20 - (2 * result_df['atr'])
        result_df['kc_upper'] = kc_upper
        result_df['kc_lower'] = kc_lower
        result_df['kc_position'] = (df['close'] - kc_lower) / (kc_upper - kc_lower)
        
        return result_df
    
    def add_pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition features.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with added pattern features
        """
        result_df = df.copy()
        
        # Common candlestick patterns
        patterns = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'harami': talib.CDLHARAMI,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'piercing_pattern': talib.CDLPIERCING,
        }
        
        # Convert OHLC to float64 for pattern recognition
        open_values = df['open'].astype(np.float64).values
        high_values = df['high'].astype(np.float64).values
        low_values = df['low'].astype(np.float64).values
        close_values = df['close'].astype(np.float64).values
        
        for pattern_name, pattern_func in patterns.items():
            result_df[f'pattern_{pattern_name}'] = pattern_func(
                open_values, high_values, low_values, close_values
            )
        
        return result_df
    
    def add_rolling_statistics(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add rolling statistical features.
        
        Args:
            df: Input DataFrame with price data
            windows: List of rolling windows [5, 10, 20, 50]
            
        Returns:
            DataFrame with added rolling statistics
        """
        if windows is None:
            windows = [5, 10, 20, 50]
            
        result_df = df.copy()
        
        for window in windows:
            # Rolling statistics for close price
            result_df[f'close_min_{window}'] = df['close'].rolling(window).min()
            result_df[f'close_max_{window}'] = df['close'].rolling(window).max()
            result_df[f'close_std_{window}'] = df['close'].rolling(window).std()
            result_df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            result_df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            
            # Position within range
            result_df[f'close_position_{window}'] = (
                (df['close'] - result_df[f'close_min_{window}']) / 
                (result_df[f'close_max_{window}'] - result_df[f'close_min_{window}'])
            )
            
            # Volume statistics
            result_df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
        return result_df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all available technical indicators and features.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        logger.info(f"Computing technical features for {len(df)} data points")
        
        # Validate input
        self.validate_data(df)
        
        result_df = df.copy()
        
        # Apply all feature categories
        logger.debug("Adding price features")
        result_df = self.add_price_features(result_df)
        
        logger.debug("Adding return features")
        result_df = self.add_returns_features(result_df)
        
        logger.debug("Adding volume features")
        result_df = self.add_volume_features(result_df)
        
        logger.debug("Adding trend indicators")
        result_df = self.add_trend_indicators(result_df)
        
        logger.debug("Adding momentum indicators")
        result_df = self.add_momentum_indicators(result_df)
        
        logger.debug("Adding volatility indicators")
        result_df = self.add_volatility_indicators(result_df)
        
        logger.debug("Adding pattern recognition")
        result_df = self.add_pattern_recognition(result_df)
        
        logger.debug("Adding rolling statistics")
        result_df = self.add_rolling_statistics(result_df)
        
        # Count features
        original_cols = len(df.columns)
        final_cols = len(result_df.columns)
        features_added = final_cols - original_cols
        
        logger.info(f"Added {features_added} technical features")
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated.
        
        Returns:
            List of feature column names
        """
        # This would need to be updated as features are added
        # For now, return a representative list
        features = []
        
        # Price features
        features.extend(['typical_price', 'hl_avg', 'price_range', 'body_size'])
        
        # Return features
        for period in [1, 5, 10, 20]:
            features.extend([f'return_{period}d', f'log_return_{period}d'])
        
        # Technical indicators
        features.extend(['rsi_14', 'macd', 'bb_position', 'atr_ratio'])
        
        return features
    
    def get_feature_importance_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by importance categories.
        
        Returns:
            Dictionary mapping importance level to feature lists
        """
        return {
            'critical': ['rsi_14', 'macd', 'bb_position', 'return_1d', 'volume_ratio'],
            'important': ['sma_20', 'atr_ratio', 'adx', 'return_5d', 'volatility_20d'],
            'useful': ['stoch_k', 'williams_r', 'cci', 'return_10d', 'price_sma20_ratio'],
            'experimental': ['pattern_doji', 'pattern_hammer', 'close_skew_20']
        }