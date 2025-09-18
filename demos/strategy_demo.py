#!/usr/bin/env python3
"""
Trading Strategy Demonstration Script

This script demonstrates the complete trading strategy framework including:
- ML model training and predictions
- Signal generation and filtering
- Position management and risk controls
- Portfolio tracking and performance analysis
"""

import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.collectors.alpha_vantage_collector import AlphaVantageDataCollector
from src.features.feature_engineering import FeatureEngineer
from src.ml.models.random_forest_model import RandomForestTradingModel
from src.strategies import MLTradingStrategy, SignalType
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_data_and_features(symbols: List[str], days_back: int = 500) -> tuple:
    """Setup data collection and feature engineering."""
    logger.info("Setting up data collection and feature engineering...")
    
    # Initialize components
    config = Config()
    collector = AlphaVantageDataCollector(api_key=config.alpha_vantage_api_key)
    feature_engineer = FeatureEngineer()
    
    # Collect data for all symbols
    all_data = {}
    for symbol in symbols:
        logger.info(f"Collecting data for {symbol}...")
        
        try:
            data = collector.get_daily_data(symbol, period='daily', outputsize='full')
            if data is not None and not data.empty:
                # Get recent data
                data = data.tail(days_back)
                all_data[symbol] = data
                logger.info(f"Collected {len(data)} days of data for {symbol}")
            else:
                logger.warning(f"No data collected for {symbol}")
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
    
    if not all_data:
        raise ValueError("No data collected for any symbols")
    
    # Engineer features for all symbols
    featured_data = {}
    for symbol, data in all_data.items():
        logger.info(f"Engineering features for {symbol}...")
        
        try:
            features = feature_engineer.create_features(data)
            if features is not None and not features.empty:
                # Remove NaN rows from feature engineering
                features = features.dropna()
                featured_data[symbol] = features
                logger.info(f"Created {len(features.columns)} features for {symbol}, {len(features)} valid rows")
            else:
                logger.warning(f"No features created for {symbol}")
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
    
    if not featured_data:
        raise ValueError("No features created for any symbols")
    
    return collector, feature_engineer, featured_data


def train_ml_model(featured_data: Dict[str, pd.DataFrame], target_days: int = 5) -> RandomForestTradingModel:
    """Train ML model on featured data."""
    logger.info(f"Training ML model with {target_days}-day prediction horizon...")
    
    # Combine all symbol data for training
    all_features = []
    all_targets = []
    
    for symbol, data in featured_data.items():
        logger.info(f"Preparing training data for {symbol}...")
        
        # Create target (future return)
        data_copy = data.copy()
        data_copy['future_return'] = data_copy['close'].pct_change(target_days).shift(-target_days)
        
        # Remove NaN targets
        valid_data = data_copy.dropna()
        
        if len(valid_data) < 50:  # Need minimum data for training
            logger.warning(f"Insufficient data for {symbol}: {len(valid_data)} rows")
            continue
        
        # Get feature columns (exclude price columns and target)
        feature_cols = [col for col in valid_data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'future_return']
                       and not col.startswith('adj_')]
        
        if not feature_cols:
            logger.warning(f"No valid features found for {symbol}")
            continue
        
        features = valid_data[feature_cols]
        targets = (valid_data['future_return'] > 0.02).astype(int)  # Binary classification: >2% return
        
        all_features.append(features)
        all_targets.append(targets)
        
        logger.info(f"Added {len(features)} samples from {symbol} for training")
    
    if not all_features:
        raise ValueError("No training data prepared")
    
    # Combine all data
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_targets, axis=0)
    
    # Remove any remaining NaNs
    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[valid_mask]
    y = y[valid_mask]
    
    logger.info(f"Total training samples: {len(X)}, Features: {len(X.columns)}")
    logger.info(f"Target distribution - Positive: {y.sum()}, Negative: {len(y) - y.sum()}")
    
    # Train model
    model = RandomForestTradingModel(
        name="StrategyDemo_RF",
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42
    )
    
    # Split data for training (use last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Train model
    model.train(X_train, y_train, X_val, y_val)
    
    logger.info("Model training completed")
    return model


def run_strategy_simulation(
    strategy: MLTradingStrategy,
    featured_data: Dict[str, pd.DataFrame],
    start_date: str = None,
    end_date: str = None
) -> Dict:
    """Run strategy simulation over historical data."""
    logger.info("Running strategy simulation...")
    
    # Determine date range
    if start_date is None:
        # Use last 6 months of data
        all_dates = []
        for data in featured_data.values():
            all_dates.extend(data.index.tolist())
        
        end_dt = max(all_dates)
        start_dt = end_dt - timedelta(days=180)
    else:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
    
    logger.info(f"Simulation period: {start_dt.date()} to {end_dt.date()}")
    
    # Get simulation dates
    simulation_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    simulation_dates = [d for d in simulation_dates if d.weekday() < 5]  # Trading days only
    
    # Initialize tracking
    portfolio_history = []
    signal_history = []
    trade_history = []
    
    logger.info(f"Simulating {len(simulation_dates)} trading days...")
    
    for i, current_date in enumerate(simulation_dates):
        if i % 20 == 0:
            logger.info(f"Processing day {i+1}/{len(simulation_dates)}: {current_date.date()}")
        
        # Get market data for current date
        current_data = {}
        for symbol, data in featured_data.items():
            # Find the closest date <= current_date
            available_dates = data.index[data.index <= current_date]
            if len(available_dates) > 0:
                latest_date = available_dates.max()
                current_data[symbol] = data.loc[latest_date]
        
        if not current_data:
            continue
        
        # Convert to DataFrame for strategy
        market_df = pd.DataFrame(current_data).T
        market_df.index.name = 'symbol'
        
        # Generate signals
        try:
            signals = strategy.generate_signals(market_df, current_date)
            
            # Execute signals
            for signal in signals:
                if signal.symbol in current_data:
                    symbol_data = current_data[signal.symbol]
                    trade = strategy.execute_signal(signal, symbol_data)
                    if trade:
                        trade_history.append(trade)
                        logger.debug(f"Executed {trade.side} {trade.shares:.0f} shares of {trade.symbol} at ${trade.price:.2f}")
            
            signal_history.extend(signals)
            
        except Exception as e:
            logger.warning(f"Error processing signals for {current_date.date()}: {e}")
            continue
        
        # Update portfolio
        try:
            strategy.update_portfolio(market_df, current_date)
        except Exception as e:
            logger.warning(f"Error updating portfolio for {current_date.date()}: {e}")
            continue
        
        # Record portfolio state
        portfolio_history.append({
            'date': current_date,
            'total_value': strategy.portfolio.total_value,
            'cash': strategy.portfolio.cash,
            'positions_value': strategy.portfolio.get_position_value(),
            'num_positions': len(strategy.portfolio.positions),
            'positions': dict(strategy.portfolio.positions)
        })
    
    logger.info(f"Simulation completed: {len(trade_history)} trades executed")
    
    return {
        'portfolio_history': portfolio_history,
        'signal_history': signal_history,
        'trade_history': trade_history,
        'final_portfolio': strategy.portfolio,
        'performance_metrics': strategy.get_performance_metrics(),
        'model_metrics': strategy.get_model_performance_metrics()
    }


def analyze_results(results: Dict):
    """Analyze and visualize simulation results."""
    logger.info("Analyzing simulation results...")
    
    portfolio_history = results['portfolio_history']
    trade_history = results['trade_history']
    performance_metrics = results['performance_metrics']
    model_metrics = results['model_metrics']
    
    if not portfolio_history:
        logger.warning("No portfolio history to analyze")
        return
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df.set_index('date', inplace=True)
    
    # Calculate daily returns
    portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
    portfolio_df['cumulative_return'] = (portfolio_df['total_value'] / portfolio_df['total_value'].iloc[0]) - 1
    
    # Print performance summary
    print("\n" + "="*60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"Initial Capital: ${portfolio_df['total_value'].iloc[0]:,.2f}")
    print(f"Final Value: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {portfolio_df['cumulative_return'].iloc[-1]:.2%}")
    print(f"Total Trades: {len(trade_history)}")
    
    if performance_metrics:
        print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
        print(f"Current Positions: {performance_metrics.get('current_positions', 0)}")
    
    if model_metrics:
        print(f"\nModel: {model_metrics.get('model_name', 'Unknown')}")
        print(f"Total Predictions: {model_metrics.get('total_predictions', 0)}")
        print(f"Average Confidence: {model_metrics.get('avg_confidence', 0):.3f}")
        print(f"Trend Alignment Rate: {model_metrics.get('trend_aligned_rate', 0):.2%}")
    
    # Trade analysis
    if trade_history:
        buy_trades = [t for t in trade_history if t.side == 'BUY']
        sell_trades = [t for t in trade_history if t.side == 'SELL']
        
        print(f"\nBuy Trades: {len(buy_trades)}")
        print(f"Sell Trades: {len(sell_trades)}")
        
        if sell_trades:
            profitable_trades = [t for t in sell_trades if t.metadata.get('realized_pnl', 0) > 0]
            print(f"Profitable Trades: {len(profitable_trades)} ({len(profitable_trades)/len(sell_trades):.1%})")
            
            total_pnl = sum(t.metadata.get('realized_pnl', 0) for t in sell_trades)
            print(f"Total Realized P&L: ${total_pnl:.2f}")
    
    # Create visualizations
    create_performance_plots(portfolio_df, trade_history)
    
    print("\nAnalysis complete. Check generated plots.")


def create_performance_plots(portfolio_df: pd.DataFrame, trade_history: List):
    """Create performance visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trading Strategy Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Portfolio value over time
    ax1 = axes[0, 0]
    ax1.plot(portfolio_df.index, portfolio_df['total_value'], linewidth=2, color='blue')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add buy/sell markers
    if trade_history:
        buy_trades = [(t.timestamp, portfolio_df.loc[portfolio_df.index >= t.timestamp, 'total_value'].iloc[0] if any(portfolio_df.index >= t.timestamp) else None) for t in trade_history if t.side == 'BUY']
        sell_trades = [(t.timestamp, portfolio_df.loc[portfolio_df.index >= t.timestamp, 'total_value'].iloc[0] if any(portfolio_df.index >= t.timestamp) else None) for t in trade_history if t.side == 'SELL']
        
        buy_dates, buy_values = zip(*[(d, v) for d, v in buy_trades if v is not None])
        sell_dates, sell_values = zip(*[(d, v) for d, v in sell_trades if v is not None])
        
        if buy_dates:
            ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=50, alpha=0.7, label='Buys')
        if sell_dates:
            ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=50, alpha=0.7, label='Sells')
        ax1.legend()
    
    # 2. Cumulative returns
    ax2 = axes[0, 1]
    ax2.plot(portfolio_df.index, portfolio_df['cumulative_return'] * 100, linewidth=2, color='green')
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Cash vs Positions over time
    ax3 = axes[1, 0]
    ax3.plot(portfolio_df.index, portfolio_df['cash'], label='Cash', linewidth=2)
    ax3.plot(portfolio_df.index, portfolio_df['positions_value'], label='Positions', linewidth=2)
    ax3.set_title('Cash vs Positions Value')
    ax3.set_ylabel('Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Number of positions over time
    ax4 = axes[1, 1]
    ax4.plot(portfolio_df.index, portfolio_df['num_positions'], linewidth=2, color='purple')
    ax4.set_title('Number of Active Positions')
    ax4.set_ylabel('Number of Positions')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"strategy_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Performance plots saved to {plot_filename}")
    
    plt.show()


def main():
    """Main demonstration function."""
    print("="*60)
    print("ALGORITHMIC TRADING ML STRATEGY DEMONSTRATION")
    print("="*60)
    
    try:
        # Configuration
        SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']
        INITIAL_CAPITAL = 100000.0
        
        print(f"\nTrading symbols: {', '.join(SYMBOLS)}")
        print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
        
        # Step 1: Setup data and features
        print(f"\n{'-'*40}")
        print("STEP 1: Data Collection & Feature Engineering")
        print(f"{'-'*40}")
        
        collector, feature_engineer, featured_data = setup_data_and_features(SYMBOLS)
        
        # Step 2: Train ML model
        print(f"\n{'-'*40}")
        print("STEP 2: ML Model Training")
        print(f"{'-'*40}")
        
        model = train_ml_model(featured_data)
        
        # Step 3: Create and configure strategy
        print(f"\n{'-'*40}")
        print("STEP 3: Strategy Configuration")
        print(f"{'-'*40}")
        
        strategy = MLTradingStrategy(
            name="Demo_ML_Strategy",
            symbols=SYMBOLS,
            model=model,
            initial_capital=INITIAL_CAPITAL,
            max_positions=5,
            risk_per_trade=0.02,
            min_confidence=0.65,
            position_sizing_method="volatility",
            prediction_threshold=0.55,
            trend_filter=True,
            volatility_adjustment=True,
            stop_loss=0.05,
            take_profit=0.15
        )
        
        logger.info(f"Strategy configured: {strategy.name}")
        logger.info(f"Position sizing method: {strategy.position_sizing_method}")
        logger.info(f"Risk per trade: {strategy.risk_per_trade:.1%}")
        
        # Step 4: Run simulation
        print(f"\n{'-'*40}")
        print("STEP 4: Strategy Simulation")
        print(f"{'-'*40}")
        
        results = run_strategy_simulation(strategy, featured_data)
        
        # Step 5: Analyze results
        print(f"\n{'-'*40}")
        print("STEP 5: Results Analysis")
        print(f"{'-'*40}")
        
        analyze_results(results)
        
        print(f"\n{'='*60}")
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error in demonstration: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check logs for detailed error information.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)