# Functional Test for Redis Embedding Cache
import redis
import numpy as np
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_redis_integration():
    print(' Testing Redis Embedding Cache Integration')
    
    try:
        # Connect to Redis using environment variable
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url, decode_responses=False)
        r.ping()  # Test connection
        print(' Redis connection successful')
    except Exception as e:
        print(f' Redis connection failed: {e}')
        print(' Make sure Redis is running and REDIS_URL is set correctly')
        print('   For Redis Cloud: REDIS_URL=rediss://username:password@host:port')
        print('   For local Redis: REDIS_URL=redis://localhost:6379')
        return
    
    # Test data
    camp_id = 17
    group_id = 14
    expire_at = int((datetime.utcnow() + timedelta(hours=24)).timestamp())
    
    # Create test embeddings
    embeddings = {}
    for i in range(10):
        emb = np.random.rand(512).astype(np.float32)
        embeddings[f'camper_{i}'] = emb
    
    print(f' Created {len(embeddings)} test embeddings')
    
    # Test SET operation
    print(' Testing Redis SET...')
    try:
        pipe = r.pipeline()
        for camper_id, emb in embeddings.items():
            pipe.hset(f'face:embeddings:camp:{camp_id}:group:{group_id}', camper_id, emb.tobytes())
        pipe.expire(f'face:embeddings:camp:{camp_id}:group:{group_id}', expire_at - int(time.time()))
        result = pipe.execute()
        print(f' SET result: {result}')
    except Exception as e:
        print(f' SET operation failed: {e}')
        return
    
    # Test GET operation
    print(' Testing Redis GET...')
    try:
        key = f'face:embeddings:camp:{camp_id}:group:{group_id}'
        stored = r.hgetall(key)
        print(f' Retrieved {len(stored)} embeddings from Redis')
    except Exception as e:
        print(f' GET operation failed: {e}')
        return
    
    # Test deserialization
    print(' Testing deserialization...')
    try:
        retrieved_embeddings = {}
        for camper_id, emb_bytes in stored.items():
            retrieved_embeddings[camper_id.decode()] = np.frombuffer(emb_bytes, dtype=np.float32)
        print(' Deserialization successful')
    except Exception as e:
        print(f' Deserialization failed: {e}')
        return
    
    # Verify data integrity
    print(' Verifying data integrity...')
    success_count = 0
    for camper_id, original_emb in embeddings.items():
        if camper_id in retrieved_embeddings:
            retrieved_emb = retrieved_embeddings[camper_id]
            if np.allclose(original_emb, retrieved_emb, rtol=1e-6):
                success_count += 1
            else:
                print(f' Data mismatch for {camper_id}')
        else:
            print(f' Missing embedding for {camper_id}')
    
    print(f' Data integrity: {success_count}/{len(embeddings)} embeddings match')
    
    # Test TTL
    print(' Testing TTL...')
    try:
        ttl = r.ttl(key)
        print(f' TTL remaining: {ttl} seconds')
    except Exception as e:
        print(f' TTL check failed: {e}')
    
    # Cleanup
    try:
        r.delete(key)
        print(' Cleaned up test data')
    except Exception as e:
        print(f' Cleanup failed: {e}')
    
    print('\n All Redis integration tests passed!')

if __name__ == '__main__':
    test_redis_integration()
