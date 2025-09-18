#!/usr/bin/env python3
"""
Simple Trading Strategy Demo

This script demonstrates the trading strategy framework with mock data,
avoiding dependency issues while showing the complete workflow.
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict
from unittest.mock import Mock
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Use only built-in libraries for data structures
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataFrame:
    """Simple DataFrame-like class for demo purposes."""
    
    def __init__(self, data_dict, index=None):
        self.data = data_dict
        self.index = index or list(range(len(next(iter(data_dict.values())))))
        self.columns = list(data_dict.keys())
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        return SimpleDataFrame({k: v[key] for k, v in self.data.items()}, 
                              index=[self.index[key]] if isinstance(key, int) else [self.index[i] for i in key])
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def iloc(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
    def loc(self, idx_key):
        if idx_key in self.index:
            pos = self.index.index(idx_key)
            return {k: v[pos] for k, v in self.data.items()}
        return None


class MockTradingModel:
    """Mock ML model for demonstration."""
    
    def __init__(self):
        self.__class__.__name__ = 'MockRandomForest'
    
    def predict(self, data):
        """Generate mock predictions based on simple rules."""
        predictions = {}
        
        # Mock prediction logic based on simple moving average
        for symbol in data.index if hasattr(data, 'index') else ['AAPL', 'MSFT', 'GOOGL']:
            try:
                if hasattr(data, 'loc'):
                    symbol_data = data.loc[symbol] if symbol in data.index else None
                else:
                    symbol_data = data.get(symbol, {})
                
                if symbol_data:
                    close = symbol_data.get('close', 100)
                    sma_20 = symbol_data.get('sma_20', close)
                    
                    # Simple prediction: if price > SMA, predict upward movement
                    if close > sma_20:
                        predictions[symbol] = 0.7  # Bullish
                    elif close < sma_20 * 0.95:
                        predictions[symbol] = 0.3  # Bearish
                    else:
                        predictions[symbol] = 0.5  # Neutral
                else:
                    predictions[symbol] = 0.5
            except Exception:
                predictions[symbol] = 0.5
        
        return predictions


def generate_mock_market_data(symbols: List[str], days: int = 100) -> Dict:
    """Generate mock market data for demonstration."""
    logger.info(f"Generating mock market data for {len(symbols)} symbols over {days} days")
    
    import random
    random.seed(42)  # For reproducible results
    
    market_data = {}
    
    for symbol in symbols:
        # Generate realistic price data
        base_price = random.uniform(80, 300)  # Starting price
        prices = []
        
        for day in range(days):
            # Random walk with slight upward bias
            change = random.gauss(0.001, 0.02)  # Small positive drift, 2% daily volatility
            if prices:
                new_price = prices[-1] * (1 + change)
            else:
                new_price = base_price * (1 + change)
            
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Calculate technical indicators
        volumes = [random.randint(1000000, 5000000) for _ in range(days)]
        
        # Simple moving averages
        sma_20 = []
        sma_50 = []
        
        for i in range(days):
            if i >= 19:
                sma_20.append(sum(prices[i-19:i+1]) / 20)
            else:
                sma_20.append(sum(prices[:i+1]) / (i+1))
            
            if i >= 49:
                sma_50.append(sum(prices[i-49:i+1]) / 50)
            else:
                sma_50.append(sum(prices[:i+1]) / (i+1))
        
        # RSI (simplified)
        rsi_values = []
        for i in range(days):
            if i < 14:
                rsi_values.append(50.0)
            else:
                gains = []
                losses = []
                for j in range(i-13, i+1):
                    change = prices[j] - prices[j-1] if j > 0 else 0
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / 14 if gains else 0
                avg_loss = sum(losses) / 14 if losses else 0.01
                
                rs = avg_gain / avg_loss if avg_loss > 0 else 0
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        # ATR (simplified)
        atr_values = []
        for i in range(days):
            if i < 14:
                atr_values.append(prices[i] * 0.02)  # 2% of price as default
            else:
                true_ranges = []
                for j in range(i-13, i+1):
                    high = prices[j] * 1.01  # Mock high
                    low = prices[j] * 0.99   # Mock low
                    prev_close = prices[j-1] if j > 0 else prices[j]
                    
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    true_ranges.append(tr)
                
                atr_values.append(sum(true_ranges) / 14)
        
        # Create data structure
        dates = [(datetime.now() - timedelta(days=days-1-i)).strftime('%Y-%m-%d') for i in range(days)]
        
        market_data[symbol] = {
            'dates': dates,
            'close': prices,
            'volume': volumes,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi_14': rsi_values,
            'atr_14': atr_values,
            'ema_12': sma_20,  # Use SMA20 as proxy for EMA12
            'ema_26': sma_50,  # Use SMA50 as proxy for EMA26
        }
    
    logger.info("Mock market data generation completed")
    return market_data


# Simple versions of strategy classes for demo
class SignalType:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradingSignal:
    def __init__(self, symbol, signal_type, confidence, timestamp, price=None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.timestamp = timestamp
        self.price = price
        self.metadata = {}


class Position:
    def __init__(self, symbol, shares, entry_price, entry_time):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.current_price = entry_price
        self.unrealized_pnl = 0.0


class Trade:
    def __init__(self, symbol, side, shares, price, timestamp):
        self.symbol = symbol
        self.side = side
        self.shares = shares
        self.price = price
        self.timestamp = timestamp
        self.metadata = {}


class SimpleMLStrategy:
    """Simplified ML trading strategy for demonstration."""
    
    def __init__(self, name, symbols, model, initial_capital=100000):
        self.name = name
        self.symbols = symbols
        self.model = model
        self.initial_capital = initial_capital
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        
        # Configuration
        self.max_positions = 3
        self.risk_per_trade = 0.02
        self.min_confidence = 0.6
        
        logger.info(f"Initialized strategy: {name} with ${initial_capital:,.2f}")
    
    def generate_signals(self, market_data, timestamp):
        """Generate trading signals."""
        signals = []
        
        try:
            predictions = self.model.predict(market_data)
            
            for symbol in self.symbols:
                if symbol in predictions:
                    prediction = predictions[symbol]
                    
                    # Convert prediction to signal
                    if prediction > 0.6:
                        signal_type = SignalType.BUY
                        confidence = prediction
                    elif prediction < 0.4:
                        signal_type = SignalType.SELL
                        confidence = 1 - prediction
                    else:
                        continue  # Skip neutral signals
                    
                    if confidence >= self.min_confidence:
                        # Get current price
                        symbol_data = market_data.get(symbol, {})
                        price = symbol_data.get('close', 100)
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=signal_type,
                            confidence=confidence,
                            timestamp=timestamp,
                            price=price
                        )
                        
                        signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def execute_signal(self, signal, market_data):
        """Execute a trading signal."""
        try:
            if signal.signal_type == SignalType.BUY:
                return self._execute_buy(signal)
            elif signal.signal_type == SignalType.SELL:
                return self._execute_sell(signal)
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
        
        return None
    
    def _execute_buy(self, signal):
        """Execute buy signal."""
        # Check if we already have a position
        if signal.symbol in self.positions:
            return None
        
        # Check max positions
        if len(self.positions) >= self.max_positions:
            return None
        
        # Calculate position size
        risk_amount = self.cash * self.risk_per_trade
        shares = int(risk_amount / signal.price) * signal.confidence
        
        if shares < 1:
            return None
        
        cost = shares * signal.price
        if cost > self.cash:
            return None
        
        # Execute trade
        self.cash -= cost
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            shares=shares,
            entry_price=signal.price,
            entry_time=signal.timestamp
        )
        
        trade = Trade(
            symbol=signal.symbol,
            side="BUY",
            shares=shares,
            price=signal.price,
            timestamp=signal.timestamp
        )
        
        self.trades.append(trade)
        logger.info(f"BUY: {shares} shares of {signal.symbol} at ${signal.price:.2f}")
        
        return trade
    
    def _execute_sell(self, signal):
        """Execute sell signal."""
        if signal.symbol not in self.positions:
            return None
        
        position = self.positions[signal.symbol]
        
        # Execute trade
        proceeds = position.shares * signal.price
        self.cash += proceeds
        
        # Calculate P&L
        pnl = (signal.price - position.entry_price) * position.shares
        
        trade = Trade(
            symbol=signal.symbol,
            side="SELL",
            shares=position.shares,
            price=signal.price,
            timestamp=signal.timestamp
        )
        trade.metadata['realized_pnl'] = pnl
        
        self.trades.append(trade)
        del self.positions[signal.symbol]
        
        logger.info(f"SELL: {position.shares} shares of {signal.symbol} at ${signal.price:.2f}, P&L: ${pnl:.2f}")
        
        return trade
    
    def get_portfolio_value(self, market_data):
        """Calculate current portfolio value."""
        position_value = 0
        for symbol, position in self.positions.items():
            symbol_data = market_data.get(symbol, {})
            current_price = symbol_data.get('close', position.entry_price)
            position_value += position.shares * current_price
        
        return self.cash + position_value


def run_simulation(strategy, market_data, start_day=50):
    """Run strategy simulation."""
    logger.info("Starting strategy simulation...")
    
    portfolio_history = []
    days_to_simulate = len(next(iter(market_data.values()))['dates']) - start_day
    
    for day in range(start_day, start_day + days_to_simulate):
        # Get current market data
        current_data = {}
        timestamp = datetime.now() - timedelta(days=days_to_simulate-day)
        
        for symbol in strategy.symbols:
            if symbol in market_data:
                data = market_data[symbol]
                current_data[symbol] = {
                    'close': data['close'][day],
                    'sma_20': data['sma_20'][day],
                    'sma_50': data['sma_50'][day],
                    'rsi_14': data['rsi_14'][day],
                    'atr_14': data['atr_14'][day]
                }
        
        # Generate and execute signals
        signals = strategy.generate_signals(current_data, timestamp)
        
        for signal in signals:
            trade = strategy.execute_signal(signal, current_data)
        
        # Update portfolio tracking
        portfolio_value = strategy.get_portfolio_value(current_data)
        
        portfolio_history.append({
            'day': day,
            'date': timestamp.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': strategy.cash,
            'num_positions': len(strategy.positions)
        })
        
        if day % 10 == 0:
            logger.info(f"Day {day}: Portfolio Value: ${portfolio_value:,.2f}, Positions: {len(strategy.positions)}")
    
    logger.info(f"Simulation completed. Executed {len(strategy.trades)} trades.")
    return portfolio_history


def analyze_results(strategy, portfolio_history):
    """Analyze and display results."""
    print("\n" + "="*60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*60)
    
    initial_value = strategy.initial_capital
    final_value = portfolio_history[-1]['portfolio_value']
    total_return = (final_value / initial_value - 1) * 100
    
    print(f"Initial Capital: ${initial_value:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(strategy.trades)}")
    print(f"Final Cash: ${strategy.cash:,.2f}")
    print(f"Active Positions: {len(strategy.positions)}")
    
    # Trade analysis
    buy_trades = [t for t in strategy.trades if t.side == 'BUY']
    sell_trades = [t for t in strategy.trades if t.side == 'SELL']
    
    print(f"\nTrade Breakdown:")
    print(f"Buy Trades: {len(buy_trades)}")
    print(f"Sell Trades: {len(sell_trades)}")
    
    if sell_trades:
        total_pnl = sum(t.metadata.get('realized_pnl', 0) for t in sell_trades)
        profitable_trades = [t for t in sell_trades if t.metadata.get('realized_pnl', 0) > 0]
        win_rate = len(profitable_trades) / len(sell_trades) * 100
        
        print(f"Total Realized P&L: ${total_pnl:.2f}")
        print(f"Profitable Trades: {len(profitable_trades)} ({win_rate:.1f}%)")
    
    # Show portfolio progression
    print(f"\nPortfolio Progression (sample points):")
    for i in range(0, len(portfolio_history), max(1, len(portfolio_history)//10)):
        entry = portfolio_history[i]
        print(f"  {entry['date']}: ${entry['portfolio_value']:,.2f}")
    
    print("\n" + "="*60)


def main():
    """Main demo function."""
    print("="*60)
    print("SIMPLE TRADING STRATEGY DEMONSTRATION")
    print("="*60)
    
    try:
        # Configuration
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        initial_capital = 100000.0
        
        print(f"\nDemo Configuration:")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        
        # Generate mock data
        logger.info("Generating mock market data...")
        market_data = generate_mock_market_data(symbols, days=120)
        
        # Create mock model
        logger.info("Creating mock ML model...")
        model = MockTradingModel()
        
        # Initialize strategy
        logger.info("Initializing trading strategy...")
        strategy = SimpleMLStrategy(
            name="Simple_ML_Demo",
            symbols=symbols,
            model=model,
            initial_capital=initial_capital
        )
        
        # Run simulation
        logger.info("Running strategy simulation...")
        portfolio_history = run_simulation(strategy, market_data)
        
        # Analyze results
        logger.info("Analyzing results...")
        analyze_results(strategy, portfolio_history)
        
        print(f"\n{'='*60}")
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)