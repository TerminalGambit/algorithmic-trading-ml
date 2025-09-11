"""Tests for the cache manager module."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data_collection.cache_manager import CacheEntry, DataCacheManager


class TestCacheEntry(unittest.TestCase):
    """Test cases for CacheEntry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {"symbol": "AAPL", "price": 150.0}
        self.cache_key = "test_key_123"
        self.source = "test_source"
        self.metadata = {"test": "metadata"}
        
        # Create entry that expires in 1 hour
        self.entry = CacheEntry(
            data=self.test_data,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            cache_key=self.cache_key,
            source=self.source,
            metadata=self.metadata
        )
    
    def test_is_not_expired(self):
        """Test that fresh entry is not expired."""
        self.assertFalse(self.entry.is_expired())
    
    def test_is_expired(self):
        """Test that old entry is expired."""
        expired_entry = CacheEntry(
            data=self.test_data,
            timestamp=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
            cache_key=self.cache_key,
            source=self.source,
            metadata=self.metadata
        )
        self.assertTrue(expired_entry.is_expired())
    
    def test_age_minutes(self):
        """Test age calculation in minutes."""
        age = self.entry.age_minutes()
        self.assertGreaterEqual(age, 0)
        self.assertLess(age, 1)  # Should be less than 1 minute old
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry_dict = self.entry.to_dict()
        
        self.assertEqual(entry_dict["data"], self.test_data)
        self.assertEqual(entry_dict["cache_key"], self.cache_key)
        self.assertEqual(entry_dict["source"], self.source)
        self.assertEqual(entry_dict["metadata"], self.metadata)
        
        # Check that timestamps are ISO formatted
        self.assertIsInstance(entry_dict["timestamp"], str)
        self.assertIsInstance(entry_dict["expires_at"], str)
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        entry_dict = self.entry.to_dict()
        restored_entry = CacheEntry.from_dict(entry_dict)
        
        self.assertEqual(restored_entry.data, self.test_data)
        self.assertEqual(restored_entry.cache_key, self.cache_key)
        self.assertEqual(restored_entry.source, self.source)
        self.assertEqual(restored_entry.metadata, self.metadata)
        
        # Timestamps should be properly restored
        self.assertIsInstance(restored_entry.timestamp, datetime)
        self.assertIsInstance(restored_entry.expires_at, datetime)


class TestDataCacheManager(unittest.TestCase):
    """Test cases for DataCacheManager class."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = DataCacheManager(cache_dir=Path(self.temp_dir))
        
        self.test_data = pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "open": [150.0, 151.0, 149.0, 152.0, 150.5],
            "high": [152.0, 153.0, 151.0, 154.0, 152.5],
            "low": [149.0, 150.0, 148.0, 151.0, 149.5],
            "close": [151.0, 152.0, 150.0, 153.0, 151.5],
            "volume": [1000000] * 5
        })
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.cache_manager._generate_cache_key(
            "alpha_vantage", symbol="AAPL", function="TIME_SERIES_DAILY"
        )
        key2 = self.cache_manager._generate_cache_key(
            "alpha_vantage", symbol="AAPL", function="TIME_SERIES_DAILY"
        )
        key3 = self.cache_manager._generate_cache_key(
            "alpha_vantage", symbol="MSFT", function="TIME_SERIES_DAILY"
        )
        
        # Same parameters should generate same key
        self.assertEqual(key1, key2)
        
        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)
        
        # Keys should be hex strings
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 32)  # MD5 hex length
    
    def test_get_cache_policy(self):
        """Test cache policy retrieval."""
        self.assertEqual(
            self.cache_manager._get_cache_policy("real_time_quote"), 1
        )
        self.assertEqual(
            self.cache_manager._get_cache_policy("daily_data"), 60 * 4
        )
        self.assertEqual(
            self.cache_manager._get_cache_policy("unknown_type"), 60 * 6
        )
    
    def test_put_and_get_memory_cache(self):
        """Test storing and retrieving from memory cache."""
        source = "alpha_vantage"
        data_type = "daily_data"
        symbol = "AAPL"
        
        # Store data
        self.cache_manager.put(
            self.test_data, source, data_type, symbol=symbol
        )
        
        # Retrieve data
        retrieved = self.cache_manager.get(
            source, data_type, symbol=symbol
        )
        
        self.assertIsNotNone(retrieved)
        pd.testing.assert_frame_equal(retrieved, self.test_data)
    
    def test_put_and_get_disk_cache(self):
        """Test storing and retrieving from disk cache."""
        source = "alpha_vantage"
        data_type = "daily_data"
        symbol = "AAPL"
        
        # Store data
        self.cache_manager.put(
            self.test_data, source, data_type, symbol=symbol
        )
        
        # Clear memory cache to force disk retrieval
        self.cache_manager._memory_cache.clear()
        
        # Retrieve data (should come from disk)
        retrieved = self.cache_manager.get(
            source, data_type, symbol=symbol
        )
        
        self.assertIsNotNone(retrieved)
        pd.testing.assert_frame_equal(retrieved, self.test_data)
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        retrieved = self.cache_manager.get(
            "alpha_vantage", "daily_data", symbol="NONEXISTENT"
        )
        self.assertIsNone(retrieved)
    
    def test_invalidate(self):
        """Test cache invalidation."""
        source = "alpha_vantage"
        data_type = "daily_data"
        symbol = "AAPL"
        
        # Store data
        self.cache_manager.put(
            self.test_data, source, data_type, symbol=symbol
        )
        
        # Verify it's cached
        retrieved = self.cache_manager.get(source, data_type, symbol=symbol)
        self.assertIsNotNone(retrieved)
        
        # Invalidate
        self.cache_manager.invalidate(source, data_type, symbol=symbol)
        
        # Verify it's gone
        retrieved = self.cache_manager.get(source, data_type, symbol=symbol)
        self.assertIsNone(retrieved)
    
    def test_clear_expired(self):
        """Test clearing expired entries."""
        # Store data normally first
        self.cache_manager.put(
            self.test_data, "alpha_vantage", "daily_data", symbol="AAPL"
        )
        
        # Manually set expiration to past time by modifying the cached entry
        cache_key = self.cache_manager._generate_cache_key(
            "alpha_vantage", data_type="daily_data", symbol="AAPL"
        )
        
        # Get the entry from memory cache and modify expiration
        if cache_key in self.cache_manager._memory_cache:
            entry = self.cache_manager._memory_cache[cache_key]
            entry.expires_at = datetime.now() - timedelta(hours=1)  # Expired
            
            # Also update the disk cache
            self.cache_manager._save_to_disk(entry)
        
        # Now clear expired entries
        cleared_count = self.cache_manager.clear_expired()
        
        self.assertGreater(cleared_count, 0)
        
        # Verify entry is gone
        retrieved = self.cache_manager.get(
            "alpha_vantage", "daily_data", symbol="AAPL"
        )
        self.assertIsNone(retrieved)
    
    def test_clear_all(self):
        """Test clearing all cache entries."""
        # Store multiple entries
        self.cache_manager.put(
            self.test_data, "alpha_vantage", "daily_data", symbol="AAPL"
        )
        self.cache_manager.put(
            self.test_data, "alpha_vantage", "daily_data", symbol="MSFT"
        )
        
        # Verify entries exist
        self.assertIsNotNone(
            self.cache_manager.get("alpha_vantage", "daily_data", symbol="AAPL")
        )
        self.assertIsNotNone(
            self.cache_manager.get("alpha_vantage", "daily_data", symbol="MSFT")
        )
        
        # Clear all
        cleared_count = self.cache_manager.clear_all()
        
        # Should have cleared at least the 2 we added (may be more from disk)
        self.assertGreaterEqual(cleared_count, 2)
        
        # Verify both are gone
        self.assertIsNone(
            self.cache_manager.get("alpha_vantage", "daily_data", symbol="AAPL")
        )
        self.assertIsNone(
            self.cache_manager.get("alpha_vantage", "daily_data", symbol="MSFT")
        )
        
        # Verify cache is empty
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats['memory_entries'], 0)
        self.assertEqual(stats['disk_entries'], 0)
    
    def test_get_cache_stats(self):
        """Test cache statistics."""
        # Initially empty
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["memory_entries"], 0)
        self.assertEqual(stats["disk_entries"], 0)
        self.assertEqual(stats["total_size_bytes"], 0)
        
        # Add some data
        self.cache_manager.put(
            self.test_data, "alpha_vantage", "daily_data", symbol="AAPL"
        )
        
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["memory_entries"], 1)
        self.assertEqual(stats["disk_entries"], 1)
        self.assertGreater(stats["total_size_bytes"], 0)
        self.assertEqual(stats["entries_by_source"]["alpha_vantage"], 1)
    
    def test_memory_cache_size_limit(self):
        """Test memory cache size limiting."""
        # Set a small limit for testing
        self.cache_manager.max_memory_cache_size = 2
        
        # Add 3 entries
        for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL"]):
            self.cache_manager.put(
                self.test_data, "alpha_vantage", "daily_data", symbol=symbol
            )
        
        # Should only have 2 in memory (oldest evicted)
        self.assertEqual(len(self.cache_manager._memory_cache), 2)
        
        # But all should be on disk
        stats = self.cache_manager.get_cache_stats()
        self.assertEqual(stats["disk_entries"], 3)
    
    def test_corrupted_cache_file_handling(self):
        """Test handling of corrupted cache files."""
        # Create a corrupted cache file
        cache_key = "corrupted_key"
        cache_file = self.cache_manager._get_cache_file_path(cache_key)
        
        with open(cache_file, 'w') as f:
            f.write("invalid json content")
        
        # Should return None and clean up corrupted file
        entry = self.cache_manager._load_from_disk(cache_key)
        self.assertIsNone(entry)
        self.assertFalse(cache_file.exists())


if __name__ == "__main__":
    unittest.main()
