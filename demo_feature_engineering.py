#!/usr/bin/env python3
"""Demo script showcasing the Feature Engineering Service.

This script demonstrates the comprehensive feature engineering capabilities
of our algorithmic trading system.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.alpha_vantage import AlphaVantageCollector
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.feature_engineering.technical_indicators import TechnicalIndicators


def main():
    """Run the feature engineering demonstration."""
    print("🚀 Algorithmic Trading ML - Feature Engineering Demo")
    print("=" * 60)
    
    # Initialize services
    print("\n📊 Initializing services...")
    av_collector = AlphaVantageCollector()
    feature_pipeline = FeaturePipeline(cache_features=True)
    technical_indicators = TechnicalIndicators()
    
    # Fetch some real market data
    symbol = "AAPL"
    print(f"\n📈 Fetching market data for {symbol}...")
    
    try:
        # Get daily data
        market_data = av_collector.get_daily_data(symbol, outputsize="compact")
        print(f"✅ Retrieved {len(market_data)} days of data")
        print(f"   Date range: {market_data.index.min().date()} to {market_data.index.max().date()}")
        
        # Display basic market data info
        print(f"\n📋 Market Data Summary:")
        print(f"   Symbol: {market_data['symbol'].iloc[0]}")
        print(f"   Latest Close: ${market_data['close'].iloc[-1]:.2f}")
        print(f"   Price Range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
        print(f"   Avg Daily Volume: {market_data['volume'].mean():,.0f}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        print("   Using sample data instead...")
        
        # Create sample data if API fails
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        initial_price = 150.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices)
        high_prices = close_prices + np.abs(np.random.normal(0, 1, 100))
        low_prices = close_prices - np.abs(np.random.normal(0, 1, 100))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = initial_price
        volume = np.random.lognormal(np.log(50000000), 0.5, 100)
        
        market_data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'symbol': symbol,
            'source': 'sample',
            'timeframe': 'daily'
        }, index=dates)
    
    # Compute technical indicators
    print(f"\n🔧 Computing technical indicators...")
    technical_features = technical_indicators.compute_all_features(market_data)
    
    feature_categories = technical_indicators.get_feature_importance_categories()
    total_features = len(technical_features.columns) - len(market_data.columns)
    print(f"✅ Generated {total_features} technical features")
    
    # Show feature categories
    print(f"\n📊 Feature Categories:")
    for category, features in feature_categories.items():
        print(f"   {category.title()}: {len(features)} features")
        if category == 'critical':
            print(f"      Examples: {', '.join(features[:3])}...")
    
    # Compute full feature pipeline
    print(f"\n⚙️ Running complete feature pipeline...")
    features_df = feature_pipeline.compute_features(
        market_data, 
        symbol=symbol,
        include_targets=True,
        prediction_horizons=[1, 5, 10]
    )
    
    total_pipeline_features = len(feature_pipeline.feature_names)
    print(f"✅ Generated {total_pipeline_features} total features (including targets)")
    
    # Prepare ML data
    print(f"\n🤖 Preparing ML-ready data...")
    X, y = feature_pipeline.prepare_ml_data(features_df)
    
    print(f"✅ ML Data prepared:")
    print(f"   Features shape: {X.shape}")
    print(f"   Targets shape: {y.shape}")
    print(f"   Data points: {len(X)}")
    
    # Show sample features
    print(f"\n📈 Sample Feature Values (latest day):")
    sample_features = ['rsi_14', 'macd', 'bb_position', 'atr_ratio', 'volume_ratio']
    for feature in sample_features:
        if feature in X.columns:
            latest_value = X[feature].iloc[-1]
            if not pd.isna(latest_value):
                print(f"   {feature}: {latest_value:.4f}")
    
    # Show target values
    print(f"\n🎯 Sample Target Values (next day predictions):")
    target_cols = [col for col in y.columns if '1d' in col]
    for target in target_cols[:3]:
        if target in y.columns:
            latest_value = y[target].iloc[-2]  # -2 because last value would be NaN
            if not pd.isna(latest_value):
                if 'direction' in target:
                    direction = "Up" if latest_value == 1 else "Down"
                    print(f"   {target}: {direction} ({latest_value})")
                else:
                    print(f"   {target}: {latest_value:.4f}")
    
    # Feature statistics and quality checks
    print(f"\n🔍 Feature Quality Analysis:")
    issues = feature_pipeline.detect_feature_issues(X)
    
    for issue_type, affected_features in issues.items():
        if affected_features:
            print(f"   {issue_type.replace('_', ' ').title()}: {len(affected_features)} features")
        else:
            print(f"   {issue_type.replace('_', ' ').title()}: ✅ None detected")
    
    # Data preprocessing demonstration
    print(f"\n🔄 Preprocessing demonstration...")
    X_sample = X.head(50)  # Use subset for demo
    
    # Fit and transform
    X_transformed = feature_pipeline.fit_transform(X_sample)
    
    print(f"✅ Preprocessing complete:")
    print(f"   Scaling method: {feature_pipeline.scaling_method}")
    print(f"   Imputation strategy: {feature_pipeline.imputation_strategy}")
    print(f"   Missing values before: {X_sample.isna().sum().sum()}")
    print(f"   Missing values after: {X_transformed.isna().sum().sum()}")
    
    # Show transformation effect
    feature_example = 'rsi_14'
    if feature_example in X_sample.columns:
        original_stats = X_sample[feature_example].describe()
        transformed_stats = X_transformed[feature_example].describe()
        
        print(f"\n📊 Transformation Example ({feature_example}):")
        print(f"   Original  - Mean: {original_stats['mean']:.3f}, Std: {original_stats['std']:.3f}")
        print(f"   Transformed - Mean: {transformed_stats['mean']:.3f}, Std: {transformed_stats['std']:.3f}")
    
    # Summary
    print(f"\n🎉 Feature Engineering Demo Complete!")
    print(f"   Total raw data points: {len(market_data)}")
    print(f"   Total engineered features: {len(X.columns)}")
    print(f"   Total target variables: {len(y.columns)}")
    print(f"   Ready for ML training: ✅")
    
    print(f"\n💡 Next steps:")
    print(f"   • Train ML models on the engineered features")
    print(f"   • Implement trading strategies based on predictions")
    print(f"   • Backtest strategies with historical data")
    print(f"   • Deploy for paper trading")


if __name__ == "__main__":
    main()