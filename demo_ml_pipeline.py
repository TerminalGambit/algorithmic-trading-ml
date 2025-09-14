#!/usr/bin/env python3
"""Demo script showcasing the complete ML Pipeline.

This script demonstrates the end-to-end ML pipeline including:
- Feature engineering with technical indicators
- Model training with RandomForest
- MLflow experiment tracking
- Cross-validation with time series splits
- Model performance evaluation
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_collection.alpha_vantage import AlphaVantageCollector
from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.models.random_forest import RandomForestTradingModel
from src.models.model_trainer import ModelTrainer


def main():
    """Run the complete ML pipeline demonstration."""
    print("🚀 Algorithmic Trading ML - Complete Pipeline Demo")
    print("=" * 65)
    
    # Initialize services
    print("\n🔧 Initializing ML Pipeline Services...")
    av_collector = AlphaVantageCollector()
    feature_pipeline = FeaturePipeline(cache_features=True, scaling_method='robust')
    model_trainer = ModelTrainer(experiment_name="demo_trading_models")
    
    # Fetch market data
    symbol = "AAPL"
    print(f"\n📈 Fetching market data for {symbol}...")
    
    try:
        # Get daily data
        market_data = av_collector.get_daily_data(symbol, outputsize="compact")
        print(f"✅ Retrieved {len(market_data)} days of data")
        print(f"   Date range: {market_data.index.min().date()} to {market_data.index.max().date()}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        print("   Using sample data instead...")
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        initial_price = 150.0
        returns = np.random.normal(0.0005, 0.02, 200)
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices)
        high_prices = close_prices + np.abs(np.random.normal(0, 1, 200))
        low_prices = close_prices - np.abs(np.random.normal(0, 1, 200))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = initial_price
        volume = np.random.lognormal(np.log(50000000), 0.5, 200)
        
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
    
    # Feature Engineering
    print(f"\n⚙️ Running feature engineering pipeline...")
    features_df = feature_pipeline.compute_features(
        market_data, 
        symbol=symbol,
        include_targets=True,
        prediction_horizons=[1, 5]  # 1-day and 5-day predictions
    )
    
    X, y = feature_pipeline.prepare_ml_data(features_df)
    print(f"✅ Feature engineering completed:")
    print(f"   Features shape: {X.shape}")
    print(f"   Targets shape: {y.shape}")
    print(f"   Available targets: {list(y.columns)}")
    
    # Choose target for classification (1-day price direction)
    target_column = 'target_direction_1d'
    if target_column not in y.columns:
        target_column = y.columns[0]  # Fallback to first available
    
    y_target = y[[target_column]]
    print(f"   Using target: {target_column}")
    
    # Preprocess features
    print(f"\n🔄 Preprocessing features...")
    feature_pipeline.fit_preprocessors(X)
    X_processed = feature_pipeline.transform(X)
    print(f"✅ Preprocessing completed:")
    print(f"   Scaling method: {feature_pipeline.scaling_method}")
    print(f"   Missing values handled: {X.isna().sum().sum()} -> {X_processed.isna().sum().sum()}")
    
    # Train Classification Model
    print(f"\n🤖 Training RandomForest Classification Model...")
    
    classification_model = RandomForestTradingModel(
        model_type="classification",
        n_estimators=100,  # Reduced for demo speed
        max_depth=10,
        random_state=42
    )
    
    # Train with validation split
    classification_results = model_trainer.train_with_validation(
        model=classification_model,
        X=X_processed,
        y=y_target,
        validation_split=0.2,
        shuffle=False  # Maintain time series order
    )
    
    print(f"✅ Classification model trained:")
    print(f"   Training samples: {classification_results['train_samples']}")
    print(f"   Validation samples: {classification_results['val_samples']}")
    print(f"   Training accuracy: {classification_results['train_scores']['accuracy']:.4f}")
    print(f"   Validation accuracy: {classification_results['validation_scores']['accuracy']:.4f}")
    
    if 'auc' in classification_results['validation_scores']:
        print(f"   Validation AUC: {classification_results['validation_scores']['auc']:.4f}")
    
    # Show feature importance
    print(f"\n📊 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(classification_results['feature_importance'].items()):
        if i >= 10:
            break
        print(f"   {i+1:2d}. {feature}: {importance:.4f}")
    
    # Train Regression Model (for returns prediction)
    regression_target = 'target_return_1d'
    if regression_target in y.columns:
        print(f"\n📈 Training RandomForest Regression Model...")
        
        regression_model = RandomForestTradingModel(
            model_type="regression",
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        y_regression = y[[regression_target]]
        
        regression_results = model_trainer.train_with_validation(
            model=regression_model,
            X=X_processed,
            y=y_regression,
            validation_split=0.2,
            shuffle=False
        )
        
        print(f"✅ Regression model trained:")
        print(f"   Training R²: {regression_results['train_scores']['r2']:.4f}")
        print(f"   Validation R²: {regression_results['validation_scores']['r2']:.4f}")
        print(f"   Validation RMSE: {regression_results['validation_scores']['rmse']:.6f}")
        print(f"   Validation MAE: {regression_results['validation_scores']['mae']:.6f}")
    
    # Cross-Validation
    print(f"\n🔄 Performing Time Series Cross-Validation...")
    
    cv_model = RandomForestTradingModel(
        model_type="classification",
        n_estimators=50,  # Reduced for CV speed
        max_depth=8,
        random_state=42
    )
    
    cv_results = model_trainer.cross_validate(
        model=cv_model,
        X=X_processed,
        y=y_target,
        cv_folds=3,  # Reduced for demo
        gap=0
    )
    
    print(f"✅ Cross-validation completed:")
    print(f"   CV Folds: {cv_results['cv_folds']}")
    print(f"   Mean Accuracy: {cv_results['primary_score_mean']:.4f} ± {cv_results['primary_score_std']:.4f}")
    
    # Show detailed fold results
    print(f"\n📋 Detailed Cross-Validation Results:")
    for fold_result in cv_results['fold_results']:
        fold = fold_result['fold']
        train_acc = fold_result['train_scores']['accuracy']
        test_acc = fold_result['test_scores']['accuracy']
        print(f"   Fold {fold}: Train: {train_acc:.4f}, Test: {test_acc:.4f} "
              f"({fold_result['train_size']} -> {fold_result['test_size']} samples)")
    
    # Model Comparison
    results_list = [classification_results]
    if regression_target in y.columns:
        results_list.append(regression_results)
    
    if len(results_list) > 1:
        print(f"\n📊 Model Comparison:")
        comparison_df = model_trainer.compare_models(results_list)
        print(comparison_df.to_string(index=False))
    
    # Save model and results
    print(f"\n💾 Saving trained models...")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save classification model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    classification_model.save_model(results_dir / f"random_forest_classifier_{timestamp}.pkl")
    
    # Save training results
    model_trainer.save_training_results(
        classification_results, 
        results_dir / f"classification_results_{timestamp}.json"
    )
    
    # MLflow Integration Summary
    print(f"\n🔬 MLflow Experiment Tracking:")
    print(f"   Experiment Name: {model_trainer.experiment_name}")
    print(f"   Classification Run ID: {classification_results['mlflow_run_id']}")
    
    if regression_target in y.columns:
        print(f"   Regression Run ID: {regression_results['mlflow_run_id']}")
    
    # Try to get recent runs
    try:
        recent_runs = model_trainer.get_experiment_runs(limit=5)
        if not recent_runs.empty:
            print(f"\n📈 Recent MLflow Runs:")
            relevant_cols = ['run_id', 'status', 'start_time']
            metric_cols = [col for col in recent_runs.columns if col.startswith('metrics.')]
            display_cols = relevant_cols + metric_cols[:3]  # Show first few metrics
            
            print(recent_runs[display_cols].head().to_string(index=False))
        else:
            print(f"   No recent runs found (MLflow may not be configured)")
    except Exception as e:
        print(f"   Could not retrieve runs: {e}")
    
    # Summary
    print(f"\n🎉 ML Pipeline Demo Complete!")
    print(f"   Data Points Processed: {len(market_data)}")
    print(f"   Features Generated: {len(X.columns)}")
    print(f"   Models Trained: {len(results_list)}")
    print(f"   Best Classification Accuracy: {classification_results['validation_scores']['accuracy']:.4f}")
    
    if regression_target in y.columns:
        print(f"   Best Regression R²: {regression_results['validation_scores']['r2']:.4f}")
    
    print(f"   Cross-Validation Score: {cv_results['primary_score_mean']:.4f} ± {cv_results['primary_score_std']:.4f}")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Experiment with different model hyperparameters")
    print(f"   • Try additional algorithms (XGBoost, LSTM, etc.)")
    print(f"   • Implement trading strategies based on predictions")
    print(f"   • Set up automated model retraining")
    print(f"   • Deploy models for paper trading")


if __name__ == "__main__":
    main()