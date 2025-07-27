# utils/cache.py
import redis
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0, ttl_seconds=600):
        self.ttl_seconds = ttl_seconds
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            if not self.redis_client.ping():
                raise Exception("Unable to connect to Redis")
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            self.redis_client = None

    def get(self, key: str):
        if not self.redis_client:
            logger.warning("Redis client not available, skipping cache")
            return None
        try:
            value = self.redis_client.get(key)
            if value:
                logger.info(f"Cache HIT for key: {key}")
                return value.decode('utf-8')
            else:
                logger.info(f"Cache MISS for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key, value, ttl=None):
        if not self.redis_client:
            logger.warning("Redis client not available, skipping set")
            return
        try:
            expire_time = ttl if ttl is not None else self.ttl_seconds
            self.redis_client.set(key, value, ex=expire_time)
            logger.info(f"Cache SET for key: {key} with TTL {expire_time} seconds")
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")

        try:
            self.redis_client.set(key, value, ex=self.ttl_seconds)
            logger.info(f"Cache SET for key: {key} with TTL {self.ttl_seconds} seconds")
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
