# utils/cache.py
"""
Cache utility module. Attempts to use Redis for caching but gracefully falls back
to no caching if the ``redis`` package is unavailable or not functioning.

This change wraps the import of ``redis`` in a try/except block so that a
missing or incompatible redis package does not crash the entire application. When
``redis`` is not available, the ``RedisCache`` class will disable caching and
log a warning.  This is particularly important for environments where the
``redis`` client library is either not installed or not compatible with the
running Python version.
"""

import logging
import requests

try:
    import redis  # type: ignore
except Exception:
    # If redis is not installed or is incompatible, set to None and handle gracefully below
    redis = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RedisCache:
    """Simple caching wrapper around Redis.

    If the ``redis`` module is unavailable or fails to connect, caching is
    disabled and all ``get`` calls return ``None`` while ``set`` calls are
    no-ops.  A TTL can be provided to control how long values remain in
    Redis; when disabled, this value is ignored.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        # If redis module is missing, disable caching
        if redis is None:
            logger.warning("redis library not available or incompatible; caching disabled")
            self.redis_client = None
            return
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            if not self.redis_client.ping():
                raise Exception("Unable to connect to Redis server")
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.redis_client = None

    def get(self, key: str):
        """Retrieve a value from the cache.

        Returns ``None`` if caching is disabled or the key is not present.
        """
        if not getattr(self, 'redis_client', None):
            logger.debug("Redis client not available; cache GET skipped")
            return None
        try:
            value = self.redis_client.get(key)
            if value:
                logger.info(f"Cache HIT for key: {key}")
                return value.decode('utf-8')
            logger.info(f"Cache MISS for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Store a value in the cache.

        If caching is disabled, this function simply returns.  A custom TTL
        may be provided; otherwise the default TTL from initialization is used.
        """
        if not getattr(self, 'redis_client', None):
            logger.debug("Redis client not available; cache SET skipped")
            return
        expire_time = ttl if ttl is not None else self.ttl_seconds
        try:
            self.redis_client.set(key, value, ex=expire_time)
            logger.info(f"Cache SET for key: {key} with TTL {expire_time} seconds")
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")

def fetch_url_with_cache(url: str, cache: RedisCache, headers=None, timeout=30, retries=1):
    cache_key = f"rss_cache:{url}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data

    # לא נמצא במטמון, נטען מהאינטרנט
    logging.info(f"Fetching RSS feed from internet: {url}")
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            cache.set(cache_key, response.text)
            return response.text
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt == retries:
                logging.error(f"Failed to fetch {url} after {retries+1} attempts")
                raise
