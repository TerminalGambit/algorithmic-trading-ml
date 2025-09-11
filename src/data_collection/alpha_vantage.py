"""Alpha Vantage data collection service.

This module handles market data collection from Alpha Vantage API with proper
rate limiting, error handling, and caching.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from config.settings import settings
from src.data_collection.cache_manager import cache_manager


class AlphaVantageCollector:
    """Alpha Vantage market data collector with rate limiting and caching."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Alpha Vantage collector.
        
        Args:
            api_key: Alpha Vantage API key. If None, uses settings.
        """
        self.api_key = api_key or settings.alpha_vantage_api_key.get_secret_value()
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request_time = 0
        self.min_request_interval = 12  # 5 requests per minute = 12 seconds between requests
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
    
    def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _make_request(self, params: Dict[str, str]) -> Dict:
        """Make API request with retry logic.
        
        Args:
            params: API request parameters
            
        Returns:
            JSON response from API
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        self._wait_for_rate_limit()
        
        params["apikey"] = self.api_key
        
        logger.debug(f"Making Alpha Vantage request with params: {params}")
        
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise requests.RequestException(f"API Error: {data['Error Message']}")
        
        if "Note" in data:
            logger.warning(f"API Note: {data['Note']}")
            raise requests.RequestException("API rate limit exceeded")
        
        return data
    
    def get_daily_data(
        self, 
        symbol: str, 
        outputsize: str = "compact"
    ) -> pd.DataFrame:
        """Get daily OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Check cache first
        cached_data = cache_manager.get(
            source="alpha_vantage",
            data_type="daily_data",
            function="TIME_SERIES_DAILY",
            symbol=symbol,
            outputsize=outputsize
        )
        
        if cached_data is not None:
            logger.info(f"Using cached daily data for {symbol}")
            return cached_data
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize
        }
        
        logger.info(f"Fetching daily data for {symbol} from API")
        
        try:
            data = self._make_request(params)
            
            # Extract time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                raise ValueError(f"No time series data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns to standard format (TIME_SERIES_DAILY format)
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Convert to numeric
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Add metadata
            df["symbol"] = symbol
            df["source"] = "alpha_vantage"
            df["timeframe"] = "daily"
            
            logger.info(f"Successfully fetched {len(df)} days of data for {symbol}")
            
            result_df = df[["symbol", "open", "high", "low", "close", "volume", "source", "timeframe"]]
            
            # Cache the result
            cache_manager.put(
                data=result_df,
                source="alpha_vantage",
                data_type="daily_data",
                metadata={"symbol": symbol, "rows": len(result_df)},
                function="TIME_SERIES_DAILY",
                symbol=symbol,
                outputsize=outputsize
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            raise
    
    def get_intraday_data(
        self, 
        symbol: str, 
        interval: str = "5min",
        month: Optional[str] = None
    ) -> pd.DataFrame:
        """Get intraday OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            interval: '1min', '5min', '15min', '30min', '60min'
            month: YYYY-MM format for historical data (optional)
            
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Determine cache data type based on interval
        data_type = f"intraday_{interval}"
        
        # Check cache first
        cached_data = cache_manager.get(
            source="alpha_vantage",
            data_type=data_type,
            function="TIME_SERIES_INTRADAY",
            symbol=symbol,
            interval=interval,
            month=month or "current"
        )
        
        if cached_data is not None:
            logger.info(f"Using cached {interval} intraday data for {symbol}")
            return cached_data
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval
        }
        
        if month:
            params["month"] = month
        
        logger.info(f"Fetching {interval} intraday data for {symbol} from API")
        
        try:
            data = self._make_request(params)
            
            # Extract time series data
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                raise ValueError(f"No intraday data found for {symbol}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ["open", "high", "low", "close", "volume"]
            
            # Convert to numeric
            df = df.astype(float)
            
            # Add metadata
            df["symbol"] = symbol
            df["source"] = "alpha_vantage"
            df["timeframe"] = interval
            
            logger.info(f"Successfully fetched {len(df)} {interval} bars for {symbol}")
            
            result_df = df[["symbol", "open", "high", "low", "close", "volume", "source", "timeframe"]]
            
            # Cache the result
            cache_manager.put(
                data=result_df,
                source="alpha_vantage",
                data_type=data_type,
                metadata={"symbol": symbol, "interval": interval, "rows": len(result_df)},
                function="TIME_SERIES_INTRADAY",
                symbol=symbol,
                interval=interval,
                month=month or "current"
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            raise
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company fundamental data overview.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with company fundamental data
        """
        # Check cache first
        cached_data = cache_manager.get(
            source="alpha_vantage",
            data_type="company_overview",
            function="OVERVIEW",
            symbol=symbol
        )
        
        if cached_data is not None:
            logger.info(f"Using cached company overview for {symbol}")
            return cached_data
        
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        
        logger.info(f"Fetching company overview for {symbol} from API")
        
        try:
            data = self._make_request(params)
            
            if not data or "Symbol" not in data:
                raise ValueError(f"No company overview found for {symbol}")
            
            logger.info(f"Successfully fetched company overview for {symbol}")
            
            # Cache the result
            cache_manager.put(
                data=data,
                source="alpha_vantage",
                data_type="company_overview",
                metadata={"symbol": symbol},
                function="OVERVIEW",
                symbol=symbol
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {str(e)}")
            raise
    
    def get_earnings(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get earnings data (annual and quarterly).
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Tuple of (annual_earnings_df, quarterly_earnings_df)
        """
        # Check cache first
        cached_data = cache_manager.get(
            source="alpha_vantage",
            data_type="earnings",
            function="EARNINGS",
            symbol=symbol
        )
        
        if cached_data is not None:
            logger.info(f"Using cached earnings data for {symbol}")
            return cached_data
        
        params = {
            "function": "EARNINGS",
            "symbol": symbol
        }
        
        logger.info(f"Fetching earnings data for {symbol} from API")
        
        try:
            data = self._make_request(params)
            
            if "annualEarnings" not in data or "quarterlyEarnings" not in data:
                raise ValueError(f"No earnings data found for {symbol}")
            
            # Convert to DataFrames
            annual_df = pd.DataFrame(data["annualEarnings"])
            quarterly_df = pd.DataFrame(data["quarterlyEarnings"])
            
            # Convert date columns
            if not annual_df.empty:
                annual_df["fiscalDateEnding"] = pd.to_datetime(annual_df["fiscalDateEnding"])
                annual_df = annual_df.sort_values("fiscalDateEnding")
            
            if not quarterly_df.empty:
                quarterly_df["fiscalDateEnding"] = pd.to_datetime(quarterly_df["fiscalDateEnding"])
                quarterly_df = quarterly_df.sort_values("fiscalDateEnding")
            
            logger.info(f"Successfully fetched earnings data for {symbol}")
            
            result = (annual_df, quarterly_df)
            
            # Cache the result
            cache_manager.put(
                data=result,
                source="alpha_vantage",
                data_type="earnings",
                metadata={
                    "symbol": symbol, 
                    "annual_rows": len(annual_df),
                    "quarterly_rows": len(quarterly_df)
                },
                function="EARNINGS",
                symbol=symbol
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {symbol}: {str(e)}")
            raise
