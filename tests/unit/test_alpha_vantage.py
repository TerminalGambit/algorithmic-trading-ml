"""Unit tests for Alpha Vantage data collector."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import requests

from src.data_collection.alpha_vantage import AlphaVantageCollector


class TestAlphaVantageCollector:
    """Test cases for AlphaVantageCollector."""

    def test_init_with_api_key(self):
        """Test collector initialization with API key."""
        collector = AlphaVantageCollector(api_key="test_key")
        assert collector.api_key == "test_key"
        assert collector.base_url == "https://www.alphavantage.co/query"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch("src.data_collection.alpha_vantage.settings") as mock_settings:
            mock_settings.alpha_vantage_api_key.get_secret_value.return_value = None
            
            with pytest.raises(ValueError, match="Alpha Vantage API key is required"):
                AlphaVantageCollector()

    @patch("time.time")
    @patch("time.sleep")
    def test_rate_limiting(self, mock_sleep, mock_time):
        """Test rate limiting functionality."""
        mock_time.side_effect = [0, 5]  # 5 seconds have passed
        
        collector = AlphaVantageCollector(api_key="test_key")
        collector.last_request_time = 0
        collector._wait_for_rate_limit()
        
        # Should sleep for (12 - 5) = 7 seconds
        mock_sleep.assert_called_once_with(7.0)

    @patch("requests.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_wait_for_rate_limit"):
            result = collector._make_request({"function": "TEST"})
        
        assert result == {"test": "data"}
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_make_request_api_error(self, mock_get):
        """Test API error handling."""
        # Mock error response
        mock_response = Mock()
        mock_response.json.return_value = {"Error Message": "Invalid symbol"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_wait_for_rate_limit"):
            with pytest.raises(requests.RequestException, match="API Error: Invalid symbol"):
                collector._make_request({"function": "TEST"})

    @patch("requests.get")
    def test_make_request_rate_limit_note(self, mock_get):
        """Test rate limit note handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.json.return_value = {"Note": "Rate limit exceeded"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_wait_for_rate_limit"):
            with pytest.raises(requests.RequestException, match="API rate limit exceeded"):
                collector._make_request({"function": "TEST"})

    def test_get_daily_data_success(self):
        """Test successful daily data retrieval."""
        # Mock API response
        mock_data = {
            "Time Series (Daily)": {
                "2023-01-01": {
                    "1. open": "100.0",
                    "2. high": "105.0", 
                    "3. low": "99.0",
                    "4. close": "104.0",
                    "5. adjusted close": "104.0",
                    "6. volume": "1000000",
                    "7. dividend amount": "0.0",
                    "8. split coefficient": "1.0"
                }
            }
        }
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            df = collector.get_daily_data("AAPL")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["open"] == 100.0
        assert df.iloc[0]["high"] == 105.0
        assert df.iloc[0]["low"] == 99.0
        assert df.iloc[0]["close"] == 104.0
        assert df.iloc[0]["volume"] == 1000000.0
        assert df.iloc[0]["source"] == "alpha_vantage"
        assert df.iloc[0]["timeframe"] == "daily"

    def test_get_daily_data_no_data(self):
        """Test daily data retrieval with no data."""
        mock_data = {"Meta Data": {}}  # No time series data
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            with pytest.raises(ValueError, match="No time series data found"):
                collector.get_daily_data("INVALID")

    def test_get_intraday_data_success(self):
        """Test successful intraday data retrieval."""
        # Mock API response
        mock_data = {
            "Time Series (5min)": {
                "2023-01-01 09:30:00": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.5", 
                    "4. close": "100.5",
                    "5. volume": "50000"
                }
            }
        }
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            df = collector.get_intraday_data("AAPL", interval="5min")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["open"] == 100.0
        assert df.iloc[0]["high"] == 101.0
        assert df.iloc[0]["timeframe"] == "5min"

    def test_get_company_overview_success(self):
        """Test successful company overview retrieval."""
        mock_data = {
            "Symbol": "AAPL",
            "Name": "Apple Inc.",
            "Exchange": "NASDAQ"
        }
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            result = collector.get_company_overview("AAPL")
        
        assert result == mock_data
        assert result["Symbol"] == "AAPL"
        assert result["Name"] == "Apple Inc."

    def test_get_company_overview_no_data(self):
        """Test company overview with no data."""
        mock_data = {}  # No symbol data
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            with pytest.raises(ValueError, match="No company overview found"):
                collector.get_company_overview("INVALID")

    def test_get_earnings_success(self):
        """Test successful earnings data retrieval."""
        mock_data = {
            "annualEarnings": [
                {
                    "fiscalDateEnding": "2022-12-31",
                    "reportedEPS": "2.50"
                }
            ],
            "quarterlyEarnings": [
                {
                    "fiscalDateEnding": "2022-12-31", 
                    "reportedEPS": "0.65"
                }
            ]
        }
        
        collector = AlphaVantageCollector(api_key="test_key")
        
        with patch.object(collector, "_make_request", return_value=mock_data):
            annual_df, quarterly_df = collector.get_earnings("AAPL")
        
        assert isinstance(annual_df, pd.DataFrame)
        assert isinstance(quarterly_df, pd.DataFrame)
        assert len(annual_df) == 1
        assert len(quarterly_df) == 1
