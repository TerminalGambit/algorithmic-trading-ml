"""
ML-based trading strategy implementation.

This module provides concrete trading strategies that use machine learning
model predictions to generate trading signals with sophisticated risk management.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from .base import BaseStrategy, TradingSignal, SignalType
from ..ml.models.base import TradingModel

logger = logging.getLogger(__name__)


class MLTradingStrategy(BaseStrategy):
    """
    Machine learning-based trading strategy.
    
    Uses trained ML models to generate trading signals based on technical
    and fundamental features with confidence-based position sizing.
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        model: TradingModel,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        min_confidence: float = 0.65,
        position_sizing_method: str = "kelly",
        **kwargs
    ):
        """
        Initialize ML trading strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            model: Trained ML model for signal generation
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            risk_per_trade: Maximum risk per trade as fraction of capital
            min_confidence: Minimum confidence threshold for signals
            position_sizing_method: Method for position sizing (kelly, fixed, volatility)
            **kwargs: Additional strategy parameters
        """
        super().__init__(
            name=name,
            symbols=symbols,
            initial_capital=initial_capital,
            max_positions=max_positions,
            risk_per_trade=risk_per_trade,
            min_confidence=min_confidence,
            position_sizing_method=position_sizing_method,
            **kwargs
        )
        
        self.model = model
        self.position_sizing_method = position_sizing_method
        
        # Strategy-specific parameters with defaults
        self.prediction_threshold = kwargs.get('prediction_threshold', 0.55)
        self.confidence_scaling = kwargs.get('confidence_scaling', True)
        self.volatility_adjustment = kwargs.get('volatility_adjustment', True)
        self.trend_filter = kwargs.get('trend_filter', True)
        
        # Tracking variables
        self.prediction_history: List[Dict] = []
        self.signal_accuracy: Dict[str, List[bool]] = {}
        
        logger.info(f"Initialized ML strategy '{name}' with model: {model.__class__.__name__}")
    
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        timestamp: datetime
    ) -> List[TradingSignal]:
        """
        Generate trading signals using ML model predictions.
        
        Args:
            data: Market data with OHLCV and engineered features
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            # Get model predictions for all symbols
            predictions = self.model.predict(data)
            
            if predictions is None or len(predictions) == 0:
                logger.warning("No predictions received from model")
                return signals
            
            # Convert predictions to DataFrame if needed
            if isinstance(predictions, np.ndarray):
                predictions = pd.DataFrame(
                    predictions, 
                    index=data.index, 
                    columns=['prediction']
                )
            
            # Generate signals for each symbol
            for symbol in self.symbols:
                if symbol not in predictions.index:
                    logger.debug(f"No prediction available for {symbol}")
                    continue
                
                signal = self._generate_signal_for_symbol(
                    symbol=symbol,
                    prediction_data=predictions.loc[symbol],
                    market_data=data.loc[symbol] if symbol in data.index else None,
                    timestamp=timestamp
                )
                
                if signal:
                    signals.append(signal)
                    self.signal_history.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
        
        logger.info(f"Generated {len(signals)} signals at {timestamp}")
        return signals
    
    def _generate_signal_for_symbol(
        self,
        symbol: str,
        prediction_data: pd.Series,
        market_data: Optional[pd.Series],
        timestamp: datetime
    ) -> Optional[TradingSignal]:
        """Generate trading signal for a specific symbol."""
        if market_data is None:
            logger.debug(f"No market data available for {symbol}")
            return None
        
        # Extract prediction and confidence
        if isinstance(prediction_data, pd.Series):
            prediction = prediction_data.get('prediction', prediction_data.iloc[0])
            confidence = prediction_data.get('confidence', abs(prediction - 0.5) * 2)
        else:
            prediction = float(prediction_data)
            confidence = abs(prediction - 0.5) * 2
        
        # Apply prediction threshold
        if prediction > self.prediction_threshold:
            signal_type = SignalType.BUY
        elif prediction < (1 - self.prediction_threshold):
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Skip HOLD signals
        if signal_type == SignalType.HOLD:
            return None
        
        # Apply trend filter if enabled
        if self.trend_filter and not self._passes_trend_filter(market_data, signal_type):
            logger.debug(f"Signal for {symbol} filtered out by trend filter")
            return None
        
        # Calculate final confidence with adjustments
        final_confidence = self._calculate_final_confidence(
            base_confidence=confidence,
            market_data=market_data,
            signal_type=signal_type
        )
        
        # Check minimum confidence threshold
        if final_confidence < self.params.get('min_confidence', 0.65):
            logger.debug(f"Signal confidence {final_confidence:.3f} below threshold for {symbol}")
            return None
        
        # Get current price
        current_price = market_data.get('close', market_data.get('Close'))
        if current_price is None:
            logger.warning(f"No price data available for {symbol}")
            return None
        
        # Record prediction for tracking
        self.prediction_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'final_confidence': final_confidence,
            'signal_type': signal_type.value,
            'price': current_price
        })
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=final_confidence,
            timestamp=timestamp,
            price=current_price,
            metadata={
                'model_prediction': prediction,
                'base_confidence': confidence,
                'trend_aligned': self._is_trend_aligned(market_data, signal_type),
                'volatility_regime': self._get_volatility_regime(market_data)
            }
        )
        
        logger.debug(f"Generated {signal_type.value} signal for {symbol} with confidence {final_confidence:.3f}")
        return signal
    
    def _passes_trend_filter(self, market_data: pd.Series, signal_type: SignalType) -> bool:
        """Check if signal aligns with current trend."""
        if not self.trend_filter:
            return True
        
        # Use multiple trend indicators
        sma_20 = market_data.get('sma_20', None)
        sma_50 = market_data.get('sma_50', None)
        ema_12 = market_data.get('ema_12', None)
        ema_26 = market_data.get('ema_26', None)
        current_price = market_data.get('close', market_data.get('Close'))
        
        if None in [sma_20, sma_50, current_price]:
            logger.debug("Insufficient trend data, skipping trend filter")
            return True
        
        # Short-term trend (price vs SMA20)
        short_trend_up = current_price > sma_20
        
        # Medium-term trend (SMA20 vs SMA50)
        medium_trend_up = sma_20 > sma_50
        
        # MACD trend if available
        macd_trend_up = True
        if ema_12 is not None and ema_26 is not None:
            macd_trend_up = ema_12 > ema_26
        
        # Combine trend signals
        trend_score = sum([short_trend_up, medium_trend_up, macd_trend_up])
        
        # For BUY signals, require at least 2/3 uptrend indicators
        if signal_type == SignalType.BUY:
            return trend_score >= 2
        
        # For SELL signals, require at least 2/3 downtrend indicators
        elif signal_type == SignalType.SELL:
            return trend_score <= 1
        
        return True
    
    def _is_trend_aligned(self, market_data: pd.Series, signal_type: SignalType) -> bool:
        """Check if signal is aligned with trend (for metadata)."""
        return self._passes_trend_filter(market_data, signal_type)
    
    def _get_volatility_regime(self, market_data: pd.Series) -> str:
        """Determine current volatility regime."""
        atr_14 = market_data.get('atr_14', None)
        atr_50 = market_data.get('atr_50', None)
        
        if atr_14 is None:
            return 'unknown'
        
        if atr_50 is not None:
            volatility_ratio = atr_14 / atr_50
            if volatility_ratio > 1.5:
                return 'high'
            elif volatility_ratio < 0.7:
                return 'low'
        
        return 'medium'
    
    def _calculate_final_confidence(
        self,
        base_confidence: float,
        market_data: pd.Series,
        signal_type: SignalType
    ) -> float:
        """Calculate final confidence with adjustments."""
        confidence = base_confidence
        
        # Volatility adjustment
        if self.volatility_adjustment:
            volatility_regime = self._get_volatility_regime(market_data)
            if volatility_regime == 'high':
                confidence *= 0.9  # Reduce confidence in high volatility
            elif volatility_regime == 'low':
                confidence *= 1.05  # Slightly increase confidence in low volatility
        
        # Trend alignment bonus
        if self.trend_filter and self._is_trend_aligned(market_data, signal_type):
            confidence *= 1.1
        
        # RSI extremes adjustment
        rsi_14 = market_data.get('rsi_14', None)
        if rsi_14 is not None:
            if signal_type == SignalType.BUY and rsi_14 < 30:
                confidence *= 1.1  # Oversold boost for buy signals
            elif signal_type == SignalType.SELL and rsi_14 > 70:
                confidence *= 1.1  # Overbought boost for sell signals
            elif signal_type == SignalType.BUY and rsi_14 > 70:
                confidence *= 0.9  # Reduce buy confidence when overbought
            elif signal_type == SignalType.SELL and rsi_14 < 30:
                confidence *= 0.9  # Reduce sell confidence when oversold
        
        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, confidence))
    
    def calculate_position_size(
        self, 
        signal: TradingSignal, 
        current_price: float,
        volatility: float
    ) -> float:
        """
        Calculate position size using specified method.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            volatility: Price volatility (ATR or similar)
            
        Returns:
            Number of shares to trade
        """
        if self.position_sizing_method == "kelly":
            return self._calculate_kelly_position_size(signal, current_price, volatility)
        elif self.position_sizing_method == "volatility":
            return self._calculate_volatility_position_size(signal, current_price, volatility)
        else:  # fixed
            return self._calculate_fixed_position_size(signal, current_price)
    
    def _calculate_fixed_position_size(
        self, 
        signal: TradingSignal, 
        current_price: float
    ) -> float:
        """Calculate fixed percentage position size."""
        risk_amount = self.portfolio.total_value * self.risk_per_trade
        max_shares = risk_amount / current_price
        
        # Scale by confidence if enabled
        if self.confidence_scaling:
            max_shares *= signal.confidence
        
        return max_shares
    
    def _calculate_volatility_position_size(
        self,
        signal: TradingSignal,
        current_price: float,
        volatility: float
    ) -> float:
        """Calculate position size based on volatility (ATR-based)."""
        # Risk amount based on portfolio
        risk_amount = self.portfolio.total_value * self.risk_per_trade
        
        # Use ATR for position sizing (risk per share)
        atr_multiplier = 2.0  # Stop loss at 2x ATR
        risk_per_share = volatility * atr_multiplier
        
        if risk_per_share <= 0:
            return self._calculate_fixed_position_size(signal, current_price)
        
        # Calculate shares based on risk per share
        max_shares = risk_amount / risk_per_share
        
        # Scale by confidence if enabled
        if self.confidence_scaling:
            max_shares *= signal.confidence
        
        return max_shares
    
    def _calculate_kelly_position_size(
        self,
        signal: TradingSignal,
        current_price: float,
        volatility: float
    ) -> float:
        """Calculate position size using simplified Kelly criterion."""
        # Get historical performance for this strategy/symbol
        symbol_accuracy = self._get_symbol_accuracy(signal.symbol)
        
        if symbol_accuracy is None:
            # Fallback to volatility-based sizing for new symbols
            return self._calculate_volatility_position_size(signal, current_price, volatility)
        
        # Simplified Kelly calculation
        win_rate = symbol_accuracy
        avg_win = 0.08  # Assume 8% average win
        avg_loss = 0.04  # Assume 4% average loss (with stop loss)
        
        # Kelly fraction: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 2.0
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        
        # Cap Kelly fraction for risk management
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
        
        # Scale by confidence
        if self.confidence_scaling:
            kelly_fraction *= signal.confidence
        
        # Calculate shares
        position_value = self.portfolio.total_value * kelly_fraction
        shares = position_value / current_price
        
        return shares
    
    def _get_symbol_accuracy(self, symbol: str) -> Optional[float]:
        """Get historical accuracy for a symbol."""
        if symbol not in self.signal_accuracy:
            return None
        
        accuracies = self.signal_accuracy[symbol]
        if len(accuracies) < 5:  # Need minimum trade history
            return None
        
        return sum(accuracies) / len(accuracies)
    
    def update_signal_accuracy(self, symbol: str, was_profitable: bool) -> None:
        """Update signal accuracy tracking for a symbol."""
        if symbol not in self.signal_accuracy:
            self.signal_accuracy[symbol] = []
        
        self.signal_accuracy[symbol].append(was_profitable)
        
        # Keep only recent history (last 50 trades)
        if len(self.signal_accuracy[symbol]) > 50:
            self.signal_accuracy[symbol] = self.signal_accuracy[symbol][-50:]
    
    def get_model_performance_metrics(self) -> Dict:
        """Get ML model-specific performance metrics."""
        if not self.prediction_history:
            return {}
        
        df = pd.DataFrame(self.prediction_history)
        
        # Prediction distribution
        prediction_stats = df['prediction'].describe()
        confidence_stats = df['confidence'].describe()
        
        # Signal distribution
        signal_counts = df['signal_type'].value_counts()
        
        # Trend alignment rate
        trend_aligned_rate = df['metadata'].apply(
            lambda x: x.get('trend_aligned', False) if x else False
        ).mean() if 'metadata' in df.columns else 0.0
        
        return {
            'total_predictions': len(df),
            'avg_prediction': prediction_stats['mean'],
            'prediction_std': prediction_stats['std'],
            'avg_confidence': confidence_stats['mean'],
            'confidence_std': confidence_stats['std'],
            'signal_distribution': signal_counts.to_dict(),
            'trend_aligned_rate': trend_aligned_rate,
            'model_name': self.model.__class__.__name__
        }


class MLEnsembleStrategy(MLTradingStrategy):
    """
    Ensemble strategy using multiple ML models.
    
    Combines predictions from multiple models to generate more robust signals.
    """
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        models: List[TradingModel],
        weights: Optional[List[float]] = None,
        consensus_threshold: float = 0.6,
        **kwargs
    ):
        """
        Initialize ensemble ML strategy.
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            models: List of trained ML models
            weights: Optional weights for model ensemble (defaults to equal)
            consensus_threshold: Minimum consensus required for signal generation
            **kwargs: Additional strategy parameters
        """
        # Use first model as primary for base class
        super().__init__(name, symbols, models[0], **kwargs)
        
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.consensus_threshold = consensus_threshold
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        logger.info(f"Initialized ensemble strategy with {len(models)} models")
    
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        timestamp: datetime
    ) -> List[TradingSignal]:
        """Generate signals using ensemble of models."""
        signals = []
        
        try:
            # Get predictions from all models
            all_predictions = []
            for i, model in enumerate(self.models):
                predictions = model.predict(data)
                if predictions is not None:
                    if isinstance(predictions, np.ndarray):
                        predictions = pd.DataFrame(
                            predictions, 
                            index=data.index, 
                            columns=[f'prediction_{i}']
                        )
                    all_predictions.append(predictions * self.weights[i])
            
            if not all_predictions:
                logger.warning("No predictions received from any model")
                return signals
            
            # Combine predictions (weighted average)
            ensemble_predictions = pd.concat(all_predictions, axis=1).sum(axis=1)
            
            # Calculate consensus (agreement between models)
            consensus_scores = self._calculate_consensus(all_predictions)
            
            # Generate signals for each symbol
            for symbol in self.symbols:
                if symbol not in ensemble_predictions.index:
                    continue
                
                consensus = consensus_scores.get(symbol, 0.0)
                if consensus < self.consensus_threshold:
                    logger.debug(f"Insufficient consensus for {symbol}: {consensus:.3f}")
                    continue
                
                # Create combined prediction data
                prediction_data = pd.Series({
                    'prediction': ensemble_predictions[symbol],
                    'confidence': consensus,
                    'consensus': consensus
                })
                
                signal = self._generate_signal_for_symbol(
                    symbol=symbol,
                    prediction_data=prediction_data,
                    market_data=data.loc[symbol] if symbol in data.index else None,
                    timestamp=timestamp
                )
                
                if signal:
                    # Add ensemble metadata
                    signal.metadata = signal.metadata or {}
                    signal.metadata['ensemble_consensus'] = consensus
                    signal.metadata['num_models'] = len(self.models)
                    
                    signals.append(signal)
                    self.signal_history.append(signal)
        
        except Exception as e:
            logger.error(f"Error generating ensemble signals: {str(e)}")
            return []
        
        logger.info(f"Generated {len(signals)} ensemble signals at {timestamp}")
        return signals
    
    def _calculate_consensus(self, all_predictions: List[pd.DataFrame]) -> Dict[str, float]:
        """Calculate consensus scores between models."""
        consensus_scores = {}
        
        if len(all_predictions) < 2:
            return {symbol: 1.0 for symbol in all_predictions[0].index}
        
        for symbol in all_predictions[0].index:
            predictions = []
            for pred_df in all_predictions:
                if symbol in pred_df.index:
                    predictions.append(pred_df.loc[symbol].iloc[0])
            
            if len(predictions) < 2:
                consensus_scores[symbol] = 1.0
                continue
            
            # Calculate agreement as inverse of standard deviation
            pred_std = np.std(predictions)
            consensus = max(0.0, 1.0 - pred_std * 2)  # Scale and invert std
            consensus_scores[symbol] = consensus
        
        return consensus_scores