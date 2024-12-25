import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional


class Cache:
    """
    A lightweight file-based cache for storing frequently used data.
    """

    def __init__(self, cache_dir: str = "./cache", default_expiry: int = 3600):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files.
            default_expiry: Default expiration time for cache items (in seconds).
        """
        self.cache_dir = Path(cache_dir)
        self.default_expiry = timedelta(seconds=default_expiry)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        """Generate a file path for the given cache key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.pkl"

    def set(self, key: str, value: Any, expiry: Optional[int] = None):
        """
        Save a value to the cache.

        Args:
            key: Cache key.
            value: Value to store.
            expiry: Expiration time for this key (in seconds). Defaults to `default_expiry`.
        """
        expiry_time = datetime.now() + timedelta(seconds=expiry or self.default_expiry.total_seconds())
        data = {
            "value": value,
            "expiry_time": expiry_time
        }
        with open(self._cache_path(key), "wb") as f:
            pickle.dump(data, f)

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key.

        Returns:
            The cached value, or None if the key is not found or expired.
        """
        cache_file = self._cache_path(key)
        if not cache_file.exists():
            return None

        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        if datetime.now() > data["expiry_time"]:
            self.delete(key)
            return None

        return data["value"]

    def delete(self, key: str):
        """
        Remove a value from the cache.

        Args:
            key: Cache key.
        """
        cache_file = self._cache_path(key)
        if cache_file.exists():
            os.remove(cache_file)

    def clear(self):
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            os.remove(cache_file)

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if the key exists and is valid, False otherwise.
        """
        return self.get(key) is not None
