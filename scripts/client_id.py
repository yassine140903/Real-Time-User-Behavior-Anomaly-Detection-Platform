import redis
from src.config import REDIS_HOST, REDIS_PORT

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
keys = r.keys("profile:client:*")
print(keys[0])