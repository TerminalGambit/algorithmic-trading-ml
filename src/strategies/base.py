"""
Base classes for trading strategies.

This module provides the foundational abstract classes that all trading strategies
must implement, defining the interface for signal generation, position management,
and strategy execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class PositionType(Enum):
    """Position types."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    price: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class Position:
    """Trading position information."""
    symbol: str
    position_type: PositionType
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def update_price(self, price: float, timestamp: datetime) -> None:
        """Update position with current market price."""
        self.current_price = price
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (price - self.entry_price) * self.shares
        elif self.position_type == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - price) * self.shares
        else:
            self.unrealized_pnl = 0.0


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: str  # "BUY" or "SELL"
    shares: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    trade_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    timestamp: datetime
    trades_today: List[Trade]
    
    def get_position_value(self) -> float:
        """Calculate total value of all positions."""
        return sum(pos.shares * (pos.current_price or pos.entry_price) 
                  for pos in self.positions.values())
    
    def update_total_value(self) -> None:
        """Update total portfolio value."""
        self.total_value = self.cash + self.get_position_value()


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All concrete strategy implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(
        self, 
        name: str,
        symbols: List[str],
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        **kwargs
    ):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            initial_capital: Starting capital
            max_positions: Maximum number of concurrent positions
            risk_per_trade: Maximum risk per trade as fraction of capital
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Initialize portfolio state
        self.portfolio = PortfolioState(
            cash=initial_capital,
            positions={},
            total_value=initial_capital,
            timestamp=datetime.now(),
            trades_today=[]
        )
        
        # Strategy parameters
        self.params = kwargs
        self.is_active = False
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.signal_history: List[TradingSignal] = []
        
        logger.info(f"Initialized strategy '{name}' with {len(symbols)} symbols")
    
    @abstractmethod
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        timestamp: datetime
    ) -> List[TradingSignal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data with OHLCV and features
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        signal: TradingSignal, 
        current_price: float,
        volatility: float
    ) -> float:
        """
        Calculate position size for a trading signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            volatility: Price volatility measure
            
        Returns:
            Number of shares to trade
        """
        pass
    
    def should_enter_position(
        self, 
        signal: TradingSignal,
        current_price: float
    ) -> bool:
        """
        Determine if should enter new position based on signal and current state.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            True if should enter position
        """
        # Check if already have position in this symbol
        if signal.symbol in self.portfolio.positions:
            logger.debug(f"Already have position in {signal.symbol}")
            return False
        
        # Check maximum positions limit
        if len(self.portfolio.positions) >= self.max_positions:
            logger.debug(f"Maximum positions limit reached: {self.max_positions}")
            return False
        
        # Check minimum confidence threshold
        min_confidence = self.params.get('min_confidence', 0.6)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal confidence {signal.confidence} below threshold {min_confidence}")
            return False
        
        return True
    
    def should_exit_position(
        self, 
        position: Position,
        current_price: float,
        timestamp: datetime
    ) -> bool:
        """
        Determine if should exit existing position.
        
        Args:
            position: Current position
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            True if should exit position
        """
        # Update position with current price
        position.update_price(current_price, timestamp)
        
        # Check stop loss
        stop_loss_pct = self.params.get('stop_loss', 0.05)  # 5% default
        if position.position_type == PositionType.LONG:
            if current_price <= position.entry_price * (1 - stop_loss_pct):
                logger.info(f"Stop loss triggered for {position.symbol} at {current_price}")
                return True
        elif position.position_type == PositionType.SHORT:
            if current_price >= position.entry_price * (1 + stop_loss_pct):
                logger.info(f"Stop loss triggered for {position.symbol} at {current_price}")
                return True
        
        # Check take profit
        take_profit_pct = self.params.get('take_profit', 0.15)  # 15% default
        if position.position_type == PositionType.LONG:
            if current_price >= position.entry_price * (1 + take_profit_pct):
                logger.info(f"Take profit triggered for {position.symbol} at {current_price}")
                return True
        elif position.position_type == PositionType.SHORT:
            if current_price <= position.entry_price * (1 - take_profit_pct):
                logger.info(f"Take profit triggered for {position.symbol} at {current_price}")
                return True
        
        # Check maximum holding period
        max_hold_days = self.params.get('max_hold_days', 30)
        hold_days = (timestamp - position.entry_time).days
        if hold_days >= max_hold_days:
            logger.info(f"Maximum hold period reached for {position.symbol}: {hold_days} days")
            return True
        
        return False
    
    def execute_signal(
        self, 
        signal: TradingSignal,
        current_data: pd.Series
    ) -> Optional[Trade]:
        """
        Execute trading signal by creating or closing positions.
        
        Args:
            signal: Trading signal to execute
            current_data: Current market data row
            
        Returns:
            Trade record if executed, None otherwise
        """
        current_price = signal.price or current_data.get('close', current_data.get('Close'))
        if current_price is None:
            logger.error(f"No price data available for {signal.symbol}")
            return None
        
        timestamp = signal.timestamp
        
        # Handle different signal types
        if signal.signal_type == SignalType.BUY:
            return self._execute_buy_signal(signal, current_price, current_data, timestamp)
        elif signal.signal_type == SignalType.SELL:
            return self._execute_sell_signal(signal, current_price, timestamp)
        else:  # HOLD
            return None
    
    def _execute_buy_signal(
        self,
        signal: TradingSignal,
        current_price: float,
        current_data: pd.Series,
        timestamp: datetime
    ) -> Optional[Trade]:
        """Execute buy signal."""
        if not self.should_enter_position(signal, current_price):
            return None
        
        # Calculate position size
        volatility = current_data.get('atr_14', 0.02)  # Use ATR or default
        shares = self.calculate_position_size(signal, current_price, volatility)
        
        if shares <= 0:
            logger.warning(f"Invalid position size calculated: {shares}")
            return None
        
        # Check if we have enough cash
        total_cost = shares * current_price
        if total_cost > self.portfolio.cash:
            logger.warning(f"Insufficient cash for trade: need ${total_cost:.2f}, have ${self.portfolio.cash:.2f}")
            return None
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            position_type=PositionType.LONG,
            shares=shares,
            entry_price=current_price,
            entry_time=timestamp,
            current_price=current_price,
            unrealized_pnl=0.0,
            metadata={'signal_confidence': signal.confidence}
        )
        
        # Update portfolio
        self.portfolio.positions[signal.symbol] = position
        self.portfolio.cash -= total_cost
        
        # Create trade record
        trade = Trade(
            symbol=signal.symbol,
            side="BUY",
            shares=shares,
            price=current_price,
            timestamp=timestamp,
            metadata={'signal_confidence': signal.confidence}
        )
        
        self.portfolio.trades_today.append(trade)
        
        logger.info(f"BUY executed: {shares:.2f} shares of {signal.symbol} at ${current_price:.2f}")
        return trade
    
    def _execute_sell_signal(
        self,
        signal: TradingSignal,
        current_price: float,
        timestamp: datetime
    ) -> Optional[Trade]:
        """Execute sell signal."""
        if signal.symbol not in self.portfolio.positions:
            logger.debug(f"No position to sell for {signal.symbol}")
            return None
        
        position = self.portfolio.positions[signal.symbol]
        
        # Close position
        total_proceeds = position.shares * current_price
        
        # Calculate realized PnL
        realized_pnl = (current_price - position.entry_price) * position.shares
        
        # Update portfolio
        self.portfolio.cash += total_proceeds
        del self.portfolio.positions[signal.symbol]
        
        # Create trade record
        trade = Trade(
            symbol=signal.symbol,
            side="SELL",
            shares=position.shares,
            price=current_price,
            timestamp=timestamp,
            metadata={
                'realized_pnl': realized_pnl,
                'hold_days': (timestamp - position.entry_time).days
            }
        )
        
        self.portfolio.trades_today.append(trade)
        
        logger.info(f"SELL executed: {position.shares:.2f} shares of {signal.symbol} at ${current_price:.2f}, PnL: ${realized_pnl:.2f}")
        return trade
    
    def update_portfolio(self, market_data: pd.DataFrame, timestamp: datetime) -> None:
        """
        Update portfolio with current market prices.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        # Update positions with current prices
        for symbol, position in self.portfolio.positions.items():
            if symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                position.update_price(current_price, timestamp)
        
        # Check for exit signals on existing positions
        positions_to_close = []
        for symbol, position in self.portfolio.positions.items():
            if symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                if self.should_exit_position(position, current_price, timestamp):
                    positions_to_close.append(symbol)
        
        # Close positions that meet exit criteria
        for symbol in positions_to_close:
            if symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                sell_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    confidence=1.0,  # Exit signals have full confidence
                    timestamp=timestamp,
                    price=current_price
                )
                self._execute_sell_signal(sell_signal, current_price, timestamp)
        
        # Update total portfolio value
        self.portfolio.timestamp = timestamp
        self.portfolio.update_total_value()
        
        # Record performance
        self.performance_history.append({
            'timestamp': timestamp,
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions_value': self.portfolio.get_position_value(),
            'num_positions': len(self.portfolio.positions),
            'daily_return': (self.portfolio.total_value / self.initial_capital) - 1
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics."""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change()
        
        # Basic metrics
        total_return = (df['total_value'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1  # Assume daily data
        
        # Risk metrics
        volatility = df['returns'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        rolling_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        total_trades = len([t for t in self.portfolio.trades_today if t.side == 'SELL'])
        winning_trades = len([t for t in self.portfolio.trades_today 
                            if t.side == 'SELL' and t.metadata.get('realized_pnl', 0) > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'current_positions': len(self.portfolio.positions),
            'cash_remaining': self.portfolio.cash
        }
    
    def start(self) -> None:
        """Start the strategy."""
        self.is_active = True
        logger.info(f"Strategy '{self.name}' started")
    
    def stop(self) -> None:
        """Stop the strategy."""
        self.is_active = False
        logger.info(f"Strategy '{self.name}' stopped")
    
    def reset(self) -> None:
        """Reset strategy to initial state."""
        self.portfolio = PortfolioState(
            cash=self.initial_capital,
            positions={},
            total_value=self.initial_capital,
            timestamp=datetime.now(),
            trades_today=[]
        )
        self.performance_history = []
        self.signal_history = []
        self.is_active = False
        
        logger.info(f"Strategy '{self.name}' reset to initial state")