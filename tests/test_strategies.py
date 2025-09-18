"""
Tests for trading strategies module.

This module tests the trading strategy framework including base classes,
ML-based strategies, position management, and signal generation.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

from src.strategies import (
    BaseStrategy, MLTradingStrategy, MLEnsembleStrategy,
    TradingSignal, SignalType, Position, PositionType, Trade, PortfolioState
)


# Test data fixtures
@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'close': [100.0, 102.0, 101.0, 103.0, 105.0],
        'high': [101.0, 103.0, 102.0, 104.0, 106.0],
        'low': [99.0, 101.0, 100.0, 102.0, 104.0],
        'volume': [1000, 1100, 950, 1200, 1050],
        'sma_20': [100.0, 101.0, 101.0, 102.0, 102.5],
        'sma_50': [99.0, 99.5, 100.0, 100.5, 101.0],
        'ema_12': [100.0, 101.5, 101.2, 102.5, 103.8],
        'ema_26': [99.5, 100.8, 100.6, 101.7, 102.4],
        'rsi_14': [50.0, 55.0, 52.0, 58.0, 62.0],
        'atr_14': [1.0, 1.1, 1.05, 1.2, 1.15],
        'atr_50': [1.2, 1.2, 1.18, 1.15, 1.12]
    }, index=dates)
    data.index.name = 'date'
    return data


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.7, 0.3, 0.6, 0.8, 0.4]))
    model.__class__.__name__ = 'MockTradingModel'
    return model


class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime):
        """Generate test signals."""
        signals = []
        for symbol in self.symbols:
            if symbol in data.index:
                # Simple test logic: buy if price above SMA20, sell otherwise
                row = data.loc[symbol]
                price = row['close']
                sma_20 = row.get('sma_20', price)
                
                if price > sma_20:
                    signal_type = SignalType.BUY
                else:
                    signal_type = SignalType.SELL
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=0.7,
                    timestamp=timestamp,
                    price=price
                )
                signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal, current_price, volatility):
        """Calculate test position size."""
        risk_amount = self.portfolio.total_value * self.risk_per_trade
        return risk_amount / current_price


class TestTradingSignal:
    """Test TradingSignal class."""
    
    def test_signal_creation(self):
        """Test signal creation with valid data."""
        timestamp = datetime.now()
        signal = TradingSignal(
            symbol="AAPL",
            signal_type=SignalType.BUY,
            confidence=0.8,
            timestamp=timestamp,
            price=150.0
        )
        
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.timestamp == timestamp
        assert signal.price == 150.0
    
    def test_signal_validation(self):
        """Test signal validation for confidence bounds."""
        timestamp = datetime.now()
        
        # Test valid confidence
        signal = TradingSignal("AAPL", SignalType.BUY, 0.5, timestamp)
        assert signal.confidence == 0.5
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            TradingSignal("AAPL", SignalType.BUY, 1.5, timestamp)
        
        with pytest.raises(ValueError):
            TradingSignal("AAPL", SignalType.BUY, -0.1, timestamp)


class TestPosition:
    """Test Position class."""
    
    def test_position_creation(self):
        """Test position creation."""
        timestamp = datetime.now()
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            shares=100,
            entry_price=150.0,
            entry_time=timestamp
        )
        
        assert position.symbol == "AAPL"
        assert position.position_type == PositionType.LONG
        assert position.shares == 100
        assert position.entry_price == 150.0
        assert position.entry_time == timestamp
    
    def test_position_price_update(self):
        """Test position price updates and PnL calculation."""
        timestamp = datetime.now()
        position = Position(
            symbol="AAPL",
            position_type=PositionType.LONG,
            shares=100,
            entry_price=150.0,
            entry_time=timestamp
        )
        
        # Test long position update
        position.update_price(155.0, timestamp)
        assert position.current_price == 155.0
        assert position.unrealized_pnl == 500.0  # (155-150) * 100
        
        # Test short position
        short_position = Position(
            symbol="MSFT",
            position_type=PositionType.SHORT,
            shares=100,
            entry_price=300.0,
            entry_time=timestamp
        )
        
        short_position.update_price(295.0, timestamp)
        assert short_position.unrealized_pnl == 500.0  # (300-295) * 100


class TestPortfolioState:
    """Test PortfolioState class."""
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        timestamp = datetime.now()
        
        # Create positions
        position1 = Position("AAPL", PositionType.LONG, 100, 150.0, timestamp)
        position1.current_price = 155.0
        
        position2 = Position("MSFT", PositionType.LONG, 50, 300.0, timestamp)
        position2.current_price = 295.0
        
        # Create portfolio
        portfolio = PortfolioState(
            cash=10000.0,
            positions={"AAPL": position1, "MSFT": position2},
            total_value=0.0,
            timestamp=timestamp,
            trades_today=[]
        )
        
        # Test position value calculation
        expected_position_value = (155.0 * 100) + (295.0 * 50)  # 30250
        assert portfolio.get_position_value() == expected_position_value
        
        # Test total value update
        portfolio.update_total_value()
        assert portfolio.total_value == 10000.0 + expected_position_value


class TestBaseStrategy:
    """Test BaseStrategy class."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL", "MSFT"],
            initial_capital=100000.0,
            max_positions=5,
            risk_per_trade=0.02
        )
        
        assert strategy.name == "Test Strategy"
        assert strategy.symbols == ["AAPL", "MSFT"]
        assert strategy.initial_capital == 100000.0
        assert strategy.max_positions == 5
        assert strategy.risk_per_trade == 0.02
        assert strategy.portfolio.cash == 100000.0
        assert len(strategy.portfolio.positions) == 0
    
    def test_signal_generation(self, sample_market_data):
        """Test signal generation."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL"],
            initial_capital=100000.0
        )
        
        # Create multi-symbol data
        multi_data = pd.concat([sample_market_data.iloc[0:1]], keys=["AAPL"])
        timestamp = datetime.now()
        
        signals = strategy.generate_signals(multi_data, timestamp)
        
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert isinstance(signals[0].signal_type, SignalType)
    
    def test_position_entry_conditions(self):
        """Test position entry condition checks."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL", "MSFT"],
            initial_capital=100000.0,
            max_positions=2
        )
        
        timestamp = datetime.now()
        signal = TradingSignal("AAPL", SignalType.BUY, 0.8, timestamp, 150.0)
        
        # Test normal entry
        assert strategy.should_enter_position(signal, 150.0) is True
        
        # Add position and test duplicate entry
        position = Position("AAPL", PositionType.LONG, 100, 150.0, timestamp)
        strategy.portfolio.positions["AAPL"] = position
        
        assert strategy.should_enter_position(signal, 150.0) is False
        
        # Test max positions limit
        position2 = Position("MSFT", PositionType.LONG, 100, 300.0, timestamp)
        strategy.portfolio.positions["MSFT"] = position2
        
        signal_googl = TradingSignal("GOOGL", SignalType.BUY, 0.8, timestamp, 2800.0)
        assert strategy.should_enter_position(signal_googl, 2800.0) is False
    
    def test_position_exit_conditions(self):
        """Test position exit condition checks."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL"],
            initial_capital=100000.0,
            stop_loss=0.05,
            take_profit=0.15
        )
        
        timestamp = datetime.now()
        position = Position("AAPL", PositionType.LONG, 100, 150.0, timestamp)
        
        # Test normal price - no exit
        assert strategy.should_exit_position(position, 152.0, timestamp) is False
        
        # Test stop loss
        assert strategy.should_exit_position(position, 142.0, timestamp) is True
        
        # Test take profit
        assert strategy.should_exit_position(position, 173.0, timestamp) is True
        
        # Test max holding period
        old_timestamp = timestamp - timedelta(days=35)
        old_position = Position("AAPL", PositionType.LONG, 100, 150.0, old_timestamp)
        assert strategy.should_exit_position(old_position, 152.0, timestamp) is True
    
    def test_trade_execution(self, sample_market_data):
        """Test trade execution."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL"],
            initial_capital=100000.0
        )
        
        timestamp = datetime.now()
        signal = TradingSignal("AAPL", SignalType.BUY, 0.8, timestamp, 150.0)
        
        # Create market data row
        market_row = sample_market_data.iloc[0].copy()
        market_row['close'] = 150.0
        market_row['atr_14'] = 1.5
        
        # Execute trade
        trade = strategy.execute_signal(signal, market_row)
        
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.side == "BUY"
        assert trade.price == 150.0
        assert trade.shares > 0
        
        # Check portfolio state
        assert "AAPL" in strategy.portfolio.positions
        assert strategy.portfolio.cash < 100000.0
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        strategy = ConcreteStrategy(
            name="Test Strategy",
            symbols=["AAPL"],
            initial_capital=100000.0
        )
        
        # Add some performance history
        timestamps = [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)]
        values = [100000, 101000, 99500, 102000, 105000]
        
        for ts, val in zip(timestamps, values):
            strategy.performance_history.append({
                'timestamp': ts,
                'total_value': val,
                'cash': 50000,
                'positions_value': val - 50000,
                'num_positions': 2,
                'daily_return': (val / 100000) - 1
            })
        
        metrics = strategy.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['total_return'] == 0.05  # 5% return


class TestMLTradingStrategy:
    """Test MLTradingStrategy class."""
    
    def test_ml_strategy_initialization(self, mock_model):
        """Test ML strategy initialization."""
        strategy = MLTradingStrategy(
            name="ML Strategy",
            symbols=["AAPL", "MSFT"],
            model=mock_model,
            initial_capital=100000.0,
            position_sizing_method="volatility"
        )
        
        assert strategy.name == "ML Strategy"
        assert strategy.model == mock_model
        assert strategy.position_sizing_method == "volatility"
        assert strategy.prediction_threshold == 0.55  # default
    
    def test_ml_signal_generation(self, mock_model, sample_market_data):
        """Test ML signal generation."""
        strategy = MLTradingStrategy(
            name="ML Strategy",
            symbols=["AAPL"],
            model=mock_model,
            prediction_threshold=0.6
        )
        
        # Setup mock to return predictions
        mock_model.predict.return_value = pd.DataFrame({
            'prediction': [0.75]  # Above threshold, should generate BUY
        }, index=["AAPL"])
        
        # Create multi-symbol data
        multi_data = pd.concat([sample_market_data.iloc[0:1]], keys=["AAPL"])
        timestamp = datetime.now()
        
        signals = strategy.generate_signals(multi_data, timestamp)
        
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].confidence > 0
        
        # Test with prediction below threshold
        mock_model.predict.return_value = pd.DataFrame({
            'prediction': [0.5]  # Below threshold, should be HOLD (filtered out)
        }, index=["AAPL"])
        
        signals = strategy.generate_signals(multi_data, timestamp)
        assert len(signals) == 0
    
    def test_trend_filter(self, mock_model, sample_market_data):
        """Test trend filtering."""
        strategy = MLTradingStrategy(
            name="ML Strategy",
            symbols=["AAPL"],
            model=mock_model,
            trend_filter=True
        )
        
        # Create data with clear uptrend
        trend_data = sample_market_data.iloc[0:1].copy()
        trend_data.loc[:, 'close'] = 105.0
        trend_data.loc[:, 'sma_20'] = 102.0
        trend_data.loc[:, 'sma_50'] = 100.0
        trend_data.loc[:, 'ema_12'] = 104.0
        trend_data.loc[:, 'ema_26'] = 103.0
        
        # Test trend-aligned BUY signal
        signal_type = SignalType.BUY
        passes_filter = strategy._passes_trend_filter(trend_data.iloc[0], signal_type)
        assert passes_filter is True
        
        # Test trend-opposite SELL signal
        signal_type = SignalType.SELL
        passes_filter = strategy._passes_trend_filter(trend_data.iloc[0], signal_type)
        assert passes_filter is False
    
    def test_position_sizing_methods(self, mock_model):
        """Test different position sizing methods."""
        strategy = MLTradingStrategy(
            name="ML Strategy",
            symbols=["AAPL"],
            model=mock_model,
            position_sizing_method="fixed"
        )
        
        timestamp = datetime.now()
        signal = TradingSignal("AAPL", SignalType.BUY, 0.8, timestamp, 150.0)
        
        # Test fixed sizing
        shares_fixed = strategy.calculate_position_size(signal, 150.0, 1.5)
        assert shares_fixed > 0
        
        # Test volatility sizing
        strategy.position_sizing_method = "volatility"
        shares_vol = strategy.calculate_position_size(signal, 150.0, 1.5)
        assert shares_vol > 0
        
        # With higher volatility, should get fewer shares
        shares_vol_high = strategy.calculate_position_size(signal, 150.0, 3.0)
        assert shares_vol_high < shares_vol
    
    def test_confidence_adjustments(self, mock_model, sample_market_data):
        """Test confidence adjustments based on market conditions."""
        strategy = MLTradingStrategy(
            name="ML Strategy",
            symbols=["AAPL"],
            model=mock_model,
            volatility_adjustment=True,
            trend_filter=True
        )
        
        base_confidence = 0.7
        signal_type = SignalType.BUY
        
        # Test with low volatility (should boost confidence)
        market_data = sample_market_data.iloc[0].copy()
        market_data['atr_14'] = 0.5
        market_data['atr_50'] = 1.0  # Low volatility regime
        
        final_confidence = strategy._calculate_final_confidence(
            base_confidence, market_data, signal_type
        )
        assert final_confidence > base_confidence
        
        # Test with high volatility (should reduce confidence)
        market_data['atr_14'] = 2.0
        market_data['atr_50'] = 1.0  # High volatility regime
        
        final_confidence = strategy._calculate_final_confidence(
            base_confidence, market_data, signal_type
        )
        assert final_confidence < base_confidence


class TestMLEnsembleStrategy:
    """Test MLEnsembleStrategy class."""
    
    def test_ensemble_initialization(self):
        """Test ensemble strategy initialization."""
        models = [Mock() for _ in range(3)]
        for i, model in enumerate(models):
            model.__class__.__name__ = f'MockModel{i}'
        
        strategy = MLEnsembleStrategy(
            name="Ensemble Strategy",
            symbols=["AAPL"],
            models=models,
            weights=[0.4, 0.3, 0.3],
            consensus_threshold=0.7
        )
        
        assert len(strategy.models) == 3
        assert strategy.weights == [0.4, 0.3, 0.3]
        assert strategy.consensus_threshold == 0.7
    
    def test_consensus_calculation(self):
        """Test consensus calculation between models."""
        models = [Mock() for _ in range(3)]
        strategy = MLEnsembleStrategy(
            name="Ensemble Strategy",
            symbols=["AAPL"],
            models=models
        )
        
        # Create test predictions with different agreement levels
        predictions = [
            pd.DataFrame({'prediction': [0.8]}, index=["AAPL"]),  # High prediction
            pd.DataFrame({'prediction': [0.7]}, index=["AAPL"]),  # Medium prediction
            pd.DataFrame({'prediction': [0.75]}, index=["AAPL"])  # Medium prediction
        ]
        
        consensus_scores = strategy._calculate_consensus(predictions)
        assert "AAPL" in consensus_scores
        assert 0.0 <= consensus_scores["AAPL"] <= 1.0
        
        # Test with more disagreeing predictions
        predictions_disagree = [
            pd.DataFrame({'prediction': [0.9]}, index=["AAPL"]),
            pd.DataFrame({'prediction': [0.3]}, index=["AAPL"]),
            pd.DataFrame({'prediction': [0.6]}, index=["AAPL"])
        ]
        
        consensus_disagree = strategy._calculate_consensus(predictions_disagree)
        # Should have lower consensus due to disagreement
        assert consensus_disagree["AAPL"] < consensus_scores["AAPL"]


@pytest.fixture
def mock_strategies(mock_model):
    """Create mock strategies for integration testing."""
    concrete_strategy = ConcreteStrategy(
        name="Test Strategy",
        symbols=["AAPL", "MSFT"],
        initial_capital=100000.0
    )
    
    ml_strategy = MLTradingStrategy(
        name="ML Strategy",
        symbols=["AAPL", "MSFT"],
        model=mock_model,
        initial_capital=100000.0
    )
    
    return concrete_strategy, ml_strategy


class TestStrategyIntegration:
    """Integration tests for strategy framework."""
    
    def test_full_trading_workflow(self, mock_strategies, sample_market_data):
        """Test complete trading workflow."""
        concrete_strategy, ml_strategy = mock_strategies
        
        # Setup ML model predictions
        ml_strategy.model.predict.return_value = pd.DataFrame({
            'prediction': [0.75, 0.25]  # BUY AAPL, SELL MSFT
        }, index=["AAPL", "MSFT"])
        
        # Create multi-symbol market data
        multi_data = pd.concat({
            "AAPL": sample_market_data.iloc[0:1],
            "MSFT": sample_market_data.iloc[0:1]
        })
        
        timestamp = datetime.now()
        
        # Generate signals
        signals = ml_strategy.generate_signals(multi_data, timestamp)
        
        # Execute signals
        executed_trades = []
        for signal in signals:
            symbol_data = multi_data.loc[signal.symbol].iloc[0]
            trade = ml_strategy.execute_signal(signal, symbol_data)
            if trade:
                executed_trades.append(trade)
        
        # Verify trades were executed
        assert len(executed_trades) > 0
        assert ml_strategy.portfolio.cash < 100000.0  # Some cash used
        assert len(ml_strategy.portfolio.positions) > 0  # Positions created
        
        # Update portfolio with new prices
        new_prices = multi_data.copy()
        new_prices.loc[("AAPL", slice(None)), "close"] = 155.0  # Price increase
        new_prices.loc[("MSFT", slice(None)), "close"] = 295.0  # Price decrease
        
        ml_strategy.update_portfolio(new_prices, timestamp + timedelta(hours=1))
        
        # Verify portfolio update
        assert ml_strategy.portfolio.total_value != 100000.0
        assert len(ml_strategy.performance_history) > 0
    
    def test_strategy_performance_tracking(self, mock_strategies):
        """Test strategy performance tracking."""
        concrete_strategy, ml_strategy = mock_strategies
        
        # Simulate some trading activity
        timestamps = [datetime.now() - timedelta(days=i) for i in range(5, 0, -1)]
        
        for i, ts in enumerate(timestamps):
            ml_strategy.performance_history.append({
                'timestamp': ts,
                'total_value': 100000 + i * 1000,  # Increasing value
                'cash': 50000,
                'positions_value': 50000 + i * 1000,
                'num_positions': min(i + 1, 3),
                'daily_return': i * 0.01
            })
        
        # Test performance metrics
        metrics = ml_strategy.get_performance_metrics()
        
        assert metrics['total_return'] > 0
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['current_positions'] == len(ml_strategy.portfolio.positions)
        
        # Test model-specific metrics
        model_metrics = ml_strategy.get_model_performance_metrics()
        
        if ml_strategy.prediction_history:
            assert 'total_predictions' in model_metrics
            assert 'model_name' in model_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])