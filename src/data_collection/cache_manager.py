"""Intelligent caching system for API data to minimize redundant requests.

This module provides a smart caching layer that:
- Stores API responses locally with configurable expiration
- Validates data freshness based on request type
- Manages cache size and cleanup
- Provides transparent caching for data collectors
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from loguru import logger

from config.settings import settings


@dataclass
class CacheEntry:
    """Represents a cached API response with metadata."""
    
    data: Any
    timestamp: datetime
    expires_at: datetime
    cache_key: str
    source: str
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at
    
    def age_minutes(self) -> float:
        """Get age of cache entry in minutes."""
        return (datetime.now() - self.timestamp).total_seconds() / 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Handle pandas DataFrame serialization
        data_to_store = self.data
        data_type = "raw"
        
        if hasattr(self.data, 'to_json'):  # pandas DataFrame/Series
            data_to_store = self.data.to_json(orient='records', date_format='iso')
            data_type = "pandas_df"
        
        return {
            "data": data_to_store,
            "data_type": data_type,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "cache_key": self.cache_key,
            "source": self.source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create cache entry from dictionary."""
        # Handle pandas DataFrame deserialization
        stored_data = data["data"]
        data_type = data.get("data_type", "raw")
        
        if data_type == "pandas_df" and isinstance(stored_data, str):
            import pandas as pd
            from io import StringIO
            stored_data = pd.read_json(StringIO(stored_data), orient='records')
        
        return cls(
            data=stored_data,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            cache_key=data["cache_key"],
            source=data["source"],
            metadata=data["metadata"]
        )


class DataCacheManager:
    """Manages intelligent caching of API data with smart expiration policies."""
    
    # Cache expiration policies (in minutes)
    CACHE_POLICIES = {
        "real_time_quote": 1,          # Real-time quotes: 1 minute
        "daily_data": 60 * 4,          # Daily data: 4 hours (markets close)
        "intraday_1min": 2,            # 1-minute intraday: 2 minutes
        "intraday_5min": 6,            # 5-minute intraday: 6 minutes
        "intraday_15min": 16,          # 15-minute intraday: 16 minutes
        "intraday_30min": 31,          # 30-minute intraday: 31 minutes
        "intraday_60min": 61,          # 1-hour intraday: 61 minutes
        "weekly_data": 60 * 24,        # Weekly data: 24 hours
        "monthly_data": 60 * 24 * 7,   # Monthly data: 7 days
        "company_overview": 60 * 24 * 30,  # Company data: 30 days
        "earnings": 60 * 24 * 7,       # Earnings: 7 days
        "news": 60,                    # News: 1 hour
        "default": 60 * 6              # Default: 6 hours
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to data/cache.
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, CacheEntry] = {}
        self.max_memory_cache_size = 100
        
        logger.info(f"DataCacheManager initialized with cache_dir: {self.cache_dir}")
    
    def _generate_cache_key(self, source: str, **params) -> str:
        """Generate a unique cache key for the request.
        
        Args:
            source: Data source name (e.g., 'alpha_vantage')
            **params: Request parameters
            
        Returns:
            Unique cache key as hex string
        """
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        key_data = f"{source}_{sorted_params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_policy(self, data_type: str) -> int:
        """Get cache expiration time in minutes for data type.
        
        Args:
            data_type: Type of data being cached
            
        Returns:
            Expiration time in minutes
        """
        return self.CACHE_POLICIES.get(data_type, self.CACHE_POLICIES["default"])
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_disk(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Cache entry if found and valid, None otherwise
        """
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            
            # Check if expired
            if entry.is_expired():
                logger.debug(f"Cache entry expired, removing: {cache_key}")
                cache_file.unlink()
                return None
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry {cache_key}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
            return None
    
    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Save cache entry to disk.
        
        Args:
            entry: Cache entry to save
        """
        cache_file = self._get_cache_file_path(entry.cache_key)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save cache entry {entry.cache_key}: {e}")
    
    def get(self, source: str, data_type: str = "default", **params) -> Optional[Any]:
        """Get cached data if available and fresh.
        
        Args:
            source: Data source name
            data_type: Type of data for cache policy
            **params: Request parameters
            
        Returns:
            Cached data if available and fresh, None otherwise
        """
        cache_key = self._generate_cache_key(source, data_type=data_type, **params)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if not entry.is_expired():
                logger.debug(f"Cache HIT (memory): {cache_key} ({entry.age_minutes():.1f}m old)")
                return entry.data
            else:
                # Remove expired entry from memory
                del self._memory_cache[cache_key]
        
        # Check disk cache
        entry = self._load_from_disk(cache_key)
        if entry:
            # Add to memory cache
            if len(self._memory_cache) < self.max_memory_cache_size:
                self._memory_cache[cache_key] = entry
            
            logger.debug(f"Cache HIT (disk): {cache_key} ({entry.age_minutes():.1f}m old)")
            return entry.data
        
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def put(self, data: Any, source: str, data_type: str = "default", 
            metadata: Optional[Dict[str, Any]] = None, **params) -> None:
        """Store data in cache with appropriate expiration.
        
        Args:
            data: Data to cache
            source: Data source name
            data_type: Type of data for cache policy
            metadata: Additional metadata to store
            **params: Request parameters used to generate cache key
        """
        cache_key = self._generate_cache_key(source, data_type=data_type, **params)
        
        # Calculate expiration
        expire_minutes = self._get_cache_policy(data_type)
        expires_at = datetime.now() + timedelta(minutes=expire_minutes)
        
        # Create cache entry
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            expires_at=expires_at,
            cache_key=cache_key,
            source=source,
            metadata=metadata or {}
        )
        
        # Store in memory cache
        if len(self._memory_cache) >= self.max_memory_cache_size:
            # Remove oldest entry
            oldest_key = min(self._memory_cache.keys(), 
                           key=lambda k: self._memory_cache[k].timestamp)
            del self._memory_cache[oldest_key]
        
        self._memory_cache[cache_key] = entry
        
        # Store on disk
        self._save_to_disk(entry)
        
        logger.debug(f"Cache STORE: {cache_key} (expires in {expire_minutes}m)")
    
    def invalidate(self, source: str, data_type: str = "default", **params) -> None:
        """Invalidate specific cache entry.
        
        Args:
            source: Data source name
            data_type: Type of data
            **params: Request parameters
        """
        cache_key = self._generate_cache_key(source, data_type=data_type, **params)
        
        # Remove from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Remove from disk
        cache_file = self._get_cache_file_path(cache_key)
        if cache_file.exists():
            cache_file.unlink()
        
        logger.debug(f"Cache INVALIDATE: {cache_key}")
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        # Clear expired memory cache entries
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            cleared_count += 1
        
        # Clear expired disk cache entries
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry.from_dict(data)
                if entry.is_expired():
                    cache_file.unlink()
                    cleared_count += 1
                    
            except Exception:
                # Remove corrupted files
                cache_file.unlink()
                cleared_count += 1
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} expired cache entries")
        
        return cleared_count
    
    def clear_all(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        # Clear memory cache
        cleared_count += len(self._memory_cache)
        self._memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            cleared_count += 1
        
        logger.info(f"Cleared all {cleared_count} cache entries")
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        disk_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in disk_files)
        
        # Count entries by source
        sources = {}
        for cache_file in disk_files:
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                source = data.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            except:
                pass
        
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "entries_by_source": sources,
            "cache_dir": str(self.cache_dir)
        }


# Global cache manager instance
cache_manager = DataCacheManager()
