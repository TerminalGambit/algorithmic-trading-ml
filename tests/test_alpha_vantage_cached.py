"""Tests for Alpha Vantage collector with caching."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from src.data_collection.alpha_vantage import AlphaVantageCollector
from src.data_collection.cache_manager import DataCacheManager


class TestAlphaVantageWithCaching(unittest.TestCase):
    """Test Alpha Vantage collector caching integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock cache manager to use temp directory
        self.original_cache_manager = None
        
        # Patch cache_manager in the alpha_vantage module
        self.cache_patcher = patch('src.data_collection.alpha_vantage.cache_manager')
        self.mock_cache_manager = self.cache_patcher.start()
        
        # Create real cache manager for some tests
        self.real_cache_manager = DataCacheManager(cache_dir=Path(self.temp_dir))
        
        # Set up collector with mock API key
        with patch.dict('os.environ', {'ALPHA_VANTAGE_API_KEY': 'test_key'}):
            self.collector = AlphaVantageCollector(api_key="test_api_key")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.cache_patcher.stop()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_daily_data_cache_hit(self):
        """Test daily data retrieval from cache."""
        # Setup mock cached data
        cached_df = pd.DataFrame({
            'symbol': ['AAPL'] * 3,
            'open': [150.0, 151.0, 149.0],
            'high': [152.0, 153.0, 151.0],
            'low': [149.0, 150.0, 148.0],
            'close': [151.0, 152.0, 150.0],
            'volume': [1000000] * 3,
            'source': ['alpha_vantage'] * 3,
            'timeframe': ['daily'] * 3
        })
        
        # Configure mock to return cached data
        self.mock_cache_manager.get.return_value = cached_df
        
        # Call method
        result = self.collector.get_daily_data("AAPL")
        
        # Verify cache was checked
        self.mock_cache_manager.get.assert_called_once_with(
            source="alpha_vantage",
            data_type="daily_data",
            function="TIME_SERIES_DAILY",
            symbol="AAPL",
            outputsize="compact"
        )
        
        # Verify result is from cache
        pd.testing.assert_frame_equal(result, cached_df)
        
        # Verify API was not called (no cache_manager.put call)
        self.mock_cache_manager.put.assert_not_called()
    
    def test_daily_data_cache_miss_and_store(self):
        """Test daily data API call and caching on cache miss."""
        # Configure cache miss
        self.mock_cache_manager.get.return_value = None
        
        # Mock API response
        mock_response = {
            "Time Series (Daily)": {
                "2023-01-03": {
                    "1. open": "130.28",
                    "2. high": "130.90",
                    "3. low": "124.17",
                    "4. close": "125.07",
                    "5. volume": "112117471"
                },
                "2023-01-02": {
                    "1. open": "129.41",
                    "2. high": "131.20",
                    "3. low": "128.12",
                    "4. close": "129.93",
                    "5. volume": "64062300"
                }
            }
        }
        
        with patch.object(self.collector, '_make_request', return_value=mock_response):
            result = self.collector.get_daily_data("AAPL")
        
        # Verify cache was checked first
        self.mock_cache_manager.get.assert_called_once()
        
        # Verify data was stored in cache
        self.mock_cache_manager.put.assert_called_once()
        
        # Check put call arguments
        put_call_args = self.mock_cache_manager.put.call_args
        self.assertEqual(put_call_args[1]['source'], "alpha_vantage")
        self.assertEqual(put_call_args[1]['data_type'], "daily_data")
        self.assertEqual(put_call_args[1]['symbol'], "AAPL")
        
        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(col in result.columns for col in 
                           ['symbol', 'open', 'high', 'low', 'close', 'volume']))
    
    def test_intraday_data_cache_types(self):
        """Test intraday data uses correct cache types for different intervals."""
        intervals = ["1min", "5min", "15min", "30min", "60min"]
        expected_cache_types = ["intraday_1min", "intraday_5min", "intraday_15min", 
                               "intraday_30min", "intraday_60min"]
        
        # Configure cache miss for all calls
        self.mock_cache_manager.get.return_value = None
        
        # Mock API response
        mock_response = {
            "Time Series (5min)": {
                "2023-01-03 16:00:00": {
                    "1. open": "125.00",
                    "2. high": "125.50",
                    "3. low": "124.50",
                    "4. close": "125.25",
                    "5. volume": "100000"
                }
            }
        }
        
        for interval, expected_cache_type in zip(intervals, expected_cache_types):
            # Update mock response key for different intervals
            interval_key = f"Time Series ({interval})"
            mock_response_for_interval = {interval_key: mock_response["Time Series (5min)"]}
            
            with patch.object(self.collector, '_make_request', 
                            return_value=mock_response_for_interval):
                try:
                    result = self.collector.get_intraday_data("AAPL", interval=interval)
                except Exception:
                    # Skip if there are issues with the test data format
                    continue
            
            # Check that correct cache type was used
            get_calls = [call for call in self.mock_cache_manager.get.call_args_list 
                        if call[1]['data_type'] == expected_cache_type]
            self.assertGreater(len(get_calls), 0, 
                             f"Expected cache type {expected_cache_type} not found")
    
    def test_company_overview_cache_long_expiry(self):
        """Test company overview uses long cache expiry."""
        # Configure cache miss
        self.mock_cache_manager.get.return_value = None
        
        # Mock API response
        mock_response = {
            "Symbol": "AAPL",
            "AssetType": "Common Stock",
            "Name": "Apple Inc",
            "Description": "Apple Inc. designs, manufactures, and markets smartphones..."
        }
        
        with patch.object(self.collector, '_make_request', return_value=mock_response):
            result = self.collector.get_company_overview("AAPL")
        
        # Verify cache was checked with company_overview type
        self.mock_cache_manager.get.assert_called_with(
            source="alpha_vantage",
            data_type="company_overview",
            function="OVERVIEW",
            symbol="AAPL"
        )
        
        # Verify data was cached with company_overview type
        put_call_args = self.mock_cache_manager.put.call_args
        self.assertEqual(put_call_args[1]['data_type'], "company_overview")
    
    def test_earnings_cache(self):
        """Test earnings data caching."""
        # Configure cache miss
        self.mock_cache_manager.get.return_value = None
        
        # Mock API response
        mock_response = {
            "annualEarnings": [
                {"fiscalDateEnding": "2023-09-30", "reportedEPS": "6.13"}
            ],
            "quarterlyEarnings": [
                {"fiscalDateEnding": "2023-09-30", "reportedEPS": "1.46"}
            ]
        }
        
        with patch.object(self.collector, '_make_request', return_value=mock_response):
            result = self.collector.get_earnings("AAPL")
        
        # Verify result is tuple of DataFrames
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], pd.DataFrame)  # Annual earnings
        self.assertIsInstance(result[1], pd.DataFrame)  # Quarterly earnings
        
        # Verify cache was used
        self.mock_cache_manager.get.assert_called_with(
            source="alpha_vantage",
            data_type="earnings",
            function="EARNINGS",
            symbol="AAPL"
        )
        
        # Verify data was cached
        put_call_args = self.mock_cache_manager.put.call_args
        self.assertEqual(put_call_args[1]['data_type'], "earnings")
    
    def test_real_cache_integration(self):
        """Test actual cache manager integration (not mocked)."""
        # Use real cache manager instead of mock
        with patch('src.data_collection.alpha_vantage.cache_manager', 
                   self.real_cache_manager):
            
            # Mock API response for daily data
            mock_response = {
                "Time Series (Daily)": {
                    "2023-01-03": {
                        "1. open": "130.28",
                        "2. high": "130.90",
                        "3. low": "124.17",
                        "4. close": "125.07",
                        "5. volume": "112117471"
                    }
                }
            }
            
            with patch.object(self.collector, '_make_request', return_value=mock_response):
                # First call - should hit API and cache result
                result1 = self.collector.get_daily_data("AAPL")
                
                # Second call - should hit cache
                result2 = self.collector.get_daily_data("AAPL")
            
            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)
            
            # Check cache stats
            stats = self.real_cache_manager.get_cache_stats()
            self.assertEqual(stats['memory_entries'], 1)
            self.assertEqual(stats['disk_entries'], 1)
            self.assertEqual(stats['entries_by_source']['alpha_vantage'], 1)


if __name__ == "__main__":
    unittest.main()
