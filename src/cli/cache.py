#!/usr/bin/env python3
"""Cache management CLI utilities.

This script provides command-line tools for managing the data cache,
including viewing statistics, clearing expired entries, and full cache cleanup.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.cache_manager import cache_manager
from loguru import logger


def show_cache_stats() -> None:
    """Display cache statistics."""
    stats = cache_manager.get_cache_stats()
    
    print("\n📊 Cache Statistics")
    print("=" * 50)
    print(f"Memory entries: {stats['memory_entries']}")
    print(f"Disk entries: {stats['disk_entries']}")
    print(f"Total size: {stats['total_size_mb']} MB ({stats['total_size_bytes']} bytes)")
    print(f"Cache directory: {stats['cache_dir']}")
    
    if stats['entries_by_source']:
        print("\nEntries by source:")
        for source, count in stats['entries_by_source'].items():
            print(f"  {source}: {count}")
    else:
        print("\nNo cache entries found.")
    print()


def clear_expired() -> None:
    """Clear expired cache entries."""
    print("🧹 Clearing expired cache entries...")
    cleared_count = cache_manager.clear_expired()
    
    if cleared_count > 0:
        print(f"✅ Cleared {cleared_count} expired entries")
    else:
        print("✅ No expired entries found")
    
    # Show updated stats
    show_cache_stats()


def clear_all() -> None:
    """Clear all cache entries."""
    print("⚠️  This will clear ALL cache entries!")
    confirmation = input("Are you sure? (y/N): ").strip().lower()
    
    if confirmation in ['y', 'yes']:
        print("🧹 Clearing all cache entries...")
        cleared_count = cache_manager.clear_all()
        print(f"✅ Cleared {cleared_count} entries")
    else:
        print("❌ Operation cancelled")
    
    # Show updated stats
    show_cache_stats()


def invalidate_symbol(symbol: str, data_type: Optional[str] = None) -> None:
    """Invalidate cache entries for a specific symbol.
    
    Args:
        symbol: Stock symbol to invalidate
        data_type: Specific data type to invalidate (optional)
    """
    if data_type:
        print(f"🗑️  Invalidating {data_type} cache for {symbol}...")
        cache_manager.invalidate(
            source="alpha_vantage",
            data_type=data_type,
            symbol=symbol
        )
        print(f"✅ Invalidated {data_type} cache for {symbol}")
    else:
        # Invalidate all data types for this symbol
        data_types = ["daily_data", "intraday_1min", "intraday_5min", 
                     "company_overview", "earnings"]
        
        print(f"🗑️  Invalidating all cache entries for {symbol}...")
        
        for dt in data_types:
            try:
                cache_manager.invalidate(
                    source="alpha_vantage", 
                    data_type=dt,
                    symbol=symbol
                )
            except Exception as e:
                logger.debug(f"No {dt} cache found for {symbol}: {e}")
        
        print(f"✅ Invalidated all cache entries for {symbol}")
    
    # Show updated stats
    show_cache_stats()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cache management utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.cache stats           # Show cache statistics
  python -m src.cli.cache clear-expired   # Clear expired entries
  python -m src.cli.cache clear-all       # Clear all entries
  python -m src.cli.cache invalidate AAPL # Invalidate all AAPL data
  python -m src.cli.cache invalidate AAPL --type daily_data # Invalidate specific type
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    subparsers.add_parser('stats', help='Show cache statistics')
    
    # Clear expired command
    subparsers.add_parser('clear-expired', help='Clear expired cache entries')
    
    # Clear all command
    subparsers.add_parser('clear-all', help='Clear all cache entries')
    
    # Invalidate command
    invalidate_parser = subparsers.add_parser('invalidate', help='Invalidate cache for symbol')
    invalidate_parser.add_argument('symbol', help='Stock symbol to invalidate')
    invalidate_parser.add_argument(
        '--type', 
        dest='data_type',
        help='Specific data type to invalidate (daily_data, company_overview, etc.)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'stats':
            show_cache_stats()
        elif args.command == 'clear-expired':
            clear_expired()
        elif args.command == 'clear-all':
            clear_all()
        elif args.command == 'invalidate':
            invalidate_symbol(args.symbol, args.data_type)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
