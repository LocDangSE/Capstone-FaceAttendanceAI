"""
Embedding Cache Service
Stores and retrieves pre-computed face embeddings for fast recognition
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from deepface import DeepFace
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import redis

from config.settings import settings, get_embedding_path
from utils import get_now

# Initialize Redis client
REDIS_URL = os.getenv('REDIS_URL', 'sample-string')
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZATION: Singleton Model Cache (Eliminates cold start)
# ============================================================================
_model_cache = {}
_model_lock = threading.Lock()

def _get_or_load_model(model_name: str):
    """Singleton pattern for DeepFace model to avoid repeated loading"""
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                logger.info(f"🔥 Loading model {model_name} (first time only)...")
                # Trigger model loading by calling DeepFace.build_model (correct API)
                try:
                    from deepface.commons import functions
                    model_obj = functions.build_model(model_name)
                    _model_cache[model_name] = model_obj
                    logger.info(f"✅ Model {model_name} loaded and cached via build_model")
                except Exception as e:
                    # Fallback: Use DeepFace.represent to trigger model loading
                    logger.info(f"Using fallback method to load {model_name}: {e}")
                    import traceback
                    logger.debug(f"Fallback reason traceback: {traceback.format_exc()}")
                    try:
                        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                        DeepFace.represent(img_path=dummy_img, 
                                          model_name=model_name, 
                                          enforce_detection=False)
                        _model_cache[model_name] = True
                        logger.info(f"✅ Model {model_name} loaded and cached via represent")
                    except Exception as fallback_error:
                        logger.error(f"❌ Failed to load model {model_name}: {fallback_error}")
                        logger.error(f"Model loading traceback: {traceback.format_exc()}")
                        raise
    return _model_cache.get(model_name)


class EmbeddingCache:
    """
    Cache for face embeddings to avoid recomputation
    Stores 512D vectors for each camper
    """
    
    def __init__(self, preload: bool = True):
        """
        Initialize embedding cache
        
        Args:
            preload: Whether to load all embeddings from disk on startup
        """
        self.cache: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.model_name = settings.DEEPFACE_MODEL
        self.distance_metric = settings.DEEPFACE_DISTANCE_METRIC
        
        # OPTIMIZATION: Preload model at initialization to avoid first-request latency
        _get_or_load_model(self.model_name)
        
        logger.info(f"EmbeddingCache initialized with model: {self.model_name}")
        
        if preload and settings.CACHE_PRELOAD:
            self._load_all_embeddings()
    
    def _load_all_embeddings(self):
        """Load all embeddings from disk on startup"""
        try:
            embeddings_dir = settings.EMBEDDINGS_FOLDER
            if not embeddings_dir.exists():
                logger.warning(f"Embeddings folder not found: {embeddings_dir}")
                return
            
            # Load all .npy files
            embedding_files = list(embeddings_dir.glob("*.npy"))
            loaded_count = 0
            
            for embedding_file in embedding_files:
                try:
                    camper_id = embedding_file.stem
                    embedding = np.load(str(embedding_file))
                    self.cache[camper_id] = embedding
                    
                    # Load metadata if exists
                    metadata_file = embedding_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            self.metadata[camper_id] = json.load(f)
                    
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load embedding {embedding_file}: {e}")
            
            logger.info(f"✅ Preloaded {loaded_count} embeddings from disk")
        
        except Exception as e:
            logger.error(f"Error preloading embeddings: {e}")
    
    def _load_embeddings_from_folder(self, folder_path: Path):
        """
        Load embeddings from a specific activity schedule folder (OPTIMIZED: Parallel processing)
        Used for selective loading instead of loading all embeddings
        
        Args:
            folder_path: Path to activity schedule database folder
        """
        try:
            if not folder_path.exists():
                logger.warning(f"Folder not found: {folder_path}")
                return
            
            # Get all image files in folder
            image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.jpeg")) + list(folder_path.glob("*.png"))
            
            if not image_files:
                logger.warning(f"No image files found in {folder_path}")
                return
            
            logger.info(f"📥 Generating embeddings for {len(image_files)} images from {folder_path}")
            
            # OPTIMIZATION: Generate embeddings in parallel during load (eager loading)
            import re
            
            def process_single_image(image_file):
                try:
                    filename = image_file.stem
                    if filename.startswith('avatar_'):
                        match = re.match(r'avatar_(\d+)', filename)
                        camper_id = match.group(1) if match else filename
                    else:
                        camper_id = filename
                    
                    # Generate embedding immediately (eager loading for fast recognition)
                    if camper_id not in self.cache:
                        embedding = self.generate_embedding(str(image_file))
                        if embedding is not None:
                            return (camper_id, embedding, str(folder_path))
                    return None
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {image_file}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return None
            
            # Use ThreadPoolExecutor for parallel generation (1 worker to avoid TensorFlow deadlock on Windows)
            results = []
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = [executor.submit(process_single_image, img) for img in image_files]
                    
                    for idx, future in enumerate(futures):
                        try:
                            result = future.result(timeout=60)
                            results.append(result)
                            if (idx + 1) % 3 == 0 or (idx + 1) == len(image_files):
                                logger.info(f"Progress: {idx + 1}/{len(image_files)} embeddings generated")
                        except Exception as e:
                            logger.error(f"Exception generating embedding {idx}: {e}")
                            results.append(None)
            except Exception as e:
                logger.error(f"ThreadPoolExecutor error: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Update cache with results
            loaded_count = 0
            for result in results:
                if result:
                    camper_id, embedding, source = result
                    self.cache[camper_id] = embedding
                    self.metadata[camper_id] = {
                        'camper_id': camper_id,
                        'model': self.model_name,
                        'cached_at': get_now().isoformat(),
                        'source': source
                    }
                    loaded_count += 1
            
            logger.info(f"✅ Generated and cached {loaded_count}/{len(image_files)} embeddings")
        
        except Exception as e:
            logger.error(f"Error loading embeddings from folder: {e}")
    
    def get_embedding(self, camper_id: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for a camper
        
        Args:
            camper_id: Camper identifier
            
        Returns:
            512D numpy array or None if not cached
        """
        return self.cache.get(camper_id)
    
    def set_embedding(
        self,
        camper_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None,
        save_to_disk: bool = True
    ) -> bool:
        """
        Store embedding in cache and optionally save to disk
        
        Args:
            camper_id: Camper identifier
            embedding: 512D numpy array
            metadata: Optional metadata dictionary
            save_to_disk: Whether to persist to disk
            
        Returns:
            Success boolean
        """
        try:
            # Validate embedding shape
            if embedding.shape != (512,) and embedding.shape != (1, 512):
                logger.error(f"Invalid embedding shape: {embedding.shape}, expected (512,)")
                return False
            
            # Ensure 1D array
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            # Store in memory
            self.cache[camper_id] = embedding
            
            # Store metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'camper_id': camper_id,
                'model': self.model_name,
                'cached_at': get_now().isoformat(),
                'embedding_shape': embedding.shape
            })
            self.metadata[camper_id] = metadata
            
            # Save to disk
            if save_to_disk:
                embedding_path = get_embedding_path(camper_id)
                np.save(str(embedding_path), embedding)
                
                # Save metadata
                metadata_path = embedding_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.debug(f"Embedding saved for camper {camper_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting embedding for {camper_id}: {e}")
            return False
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate face embedding from image using DeepFace (uses singleton cached model)
        
        Args:
            image_path: Path to face image
            
        Returns:
            512D numpy array or None on failure
        """
        try:
            logger.debug(f"Generating embedding for: {image_path}")
            
            # OPTIMIZATION: Ensure model is loaded (singleton pattern)
            _get_or_load_model(self.model_name)
            
            # Use DeepFace to generate embedding
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend=settings.DEEPFACE_DETECTOR
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                logger.error("No embedding generated")
                return None
            
            # Extract embedding vector
            embedding = np.array(embedding_objs[0]['embedding'])
            
            logger.debug(f"Embedding generated: shape={embedding.shape}")
            return embedding
        
        except KeyboardInterrupt:
            # Re-raise keyboard interrupt to allow graceful shutdown
            logger.warning(f"Keyboard interrupt during embedding generation for {image_path}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            import traceback
            logger.error(f"Embedding generation traceback: {traceback.format_exc()}")
            return None
    
    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Distance value (lower = more similar)
        """
        try:
            if self.distance_metric == 'cosine':
                # Cosine distance: 1 - cosine_similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                cosine_similarity = dot_product / (norm1 * norm2)
                distance = 1 - cosine_similarity
            
            elif self.distance_metric == 'euclidean':
                # Euclidean distance
                distance = np.linalg.norm(embedding1 - embedding2)
            
            elif self.distance_metric == 'euclidean_l2':
                # L2 normalized Euclidean distance
                distance = np.linalg.norm(embedding1 - embedding2) / len(embedding1)
            
            else:
                logger.error(f"Unknown distance metric: {self.distance_metric}")
                distance = float('inf')
            
            return float(distance)
        
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return float('inf')
    
    def find_best_match(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float, float]:
        """
        Find best matching camper from cached embeddings (OPTIMIZED: Early exit + vectorization)
        
        Args:
            query_embedding: Query embedding vector
            threshold: Maximum distance threshold (uses config default if None)
            
        Returns:
            Tuple of (camper_id, distance, confidence)
            Returns (None, inf, 0.0) if no match found
        """
        try:
            if threshold is None:
                threshold = settings.CONFIDENCE_THRESHOLD
            
            if not self.cache:
                logger.warning("Cache is empty, no campers to match against")
                return None, float('inf'), 0.0
            
            # OPTIMIZATION: Vectorized comparison for speed
            camper_ids = list(self.cache.keys())
            cached_embeddings = np.array([self.cache[cid] for cid in camper_ids])
            
            # Vectorized distance calculation
            if self.distance_metric == 'cosine':
                # Batch cosine distance
                query_norm = np.linalg.norm(query_embedding)
                cached_norms = np.linalg.norm(cached_embeddings, axis=1)
                dot_products = np.dot(cached_embeddings, query_embedding)
                cosine_similarities = dot_products / (cached_norms * query_norm + 1e-8)
                distances = 1 - cosine_similarities
            elif self.distance_metric == 'euclidean':
                # Batch Euclidean distance
                distances = np.linalg.norm(cached_embeddings - query_embedding, axis=1)
            else:
                # Fallback to original implementation
                distances = np.array([self.compare_embeddings(query_embedding, emb) 
                                     for emb in cached_embeddings])
            
            # Find best match
            best_idx = np.argmin(distances)
            best_distance = float(distances[best_idx])
            best_camper_id = camper_ids[best_idx]
            
            # OPTIMIZATION: Early exit for high-confidence matches (70% of threshold)
            high_confidence_threshold = threshold * 0.7
            if best_distance < high_confidence_threshold:
                confidence = 1.0 - best_distance
                logger.debug(f"⚡ Early exit - High confidence match: {best_camper_id} (distance={best_distance:.4f})")
                return best_camper_id, best_distance, confidence
            
            # Check if best match is within threshold
            if best_distance <= threshold:
                confidence = 1.0 - best_distance
                logger.info(f"✅ Match found: camper_id={best_camper_id} (distance={best_distance:.4f}, confidence={confidence:.4f}, threshold={threshold})")
                return best_camper_id, best_distance, confidence
            else:
                logger.warning(f"❌ No match: best_distance={best_distance:.4f} exceeds threshold={threshold} (best_camper_id={best_camper_id})")
                return None, best_distance, 0.0
        
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None, float('inf'), 0.0
    
    def batch_find_matches(
        self,
        query_embeddings: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> List[Tuple[Optional[str], float, float]]:
        """
        Find best matches for multiple embeddings efficiently
        
        Args:
            query_embeddings: List of query embedding vectors
            threshold: Maximum distance threshold
            
        Returns:
            List of (camper_id, distance, confidence) tuples
        """
        results = []
        for embedding in query_embeddings:
            match = self.find_best_match(embedding, threshold)
            results.append(match)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'total_cached': len(self.cache),
            'campers': list(self.cache.keys()),
            'model': self.model_name,
            'distance_metric': self.distance_metric,
            'embeddings_folder': str(settings.EMBEDDINGS_FOLDER)
        }
    
    def get_redis_stats(self) -> Dict:
        """
        Get Redis statistics for face embeddings
        
        Returns:
            Dictionary with Redis statistics
        """
        try:
            # Get all face embedding keys
            pattern = "face:embeddings:camp:*:group:*"
            keys = redis_client.keys(pattern)
            
            redis_stats = {
                'total_keys': len(keys),
                'keys': [key.decode() if isinstance(key, bytes) else key for key in keys],
                'camp_groups': {}
            }
            
            # Get details for each key
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                # Parse camp and group from key
                parts = key_str.split(':')
                if len(parts) >= 5:
                    camp_id = parts[3]
                    group_id = parts[5]
                    
                    # Get hash length (number of embeddings)
                    hash_len = redis_client.hlen(key)
                    # Get TTL
                    ttl = redis_client.ttl(key)
                    
                    if camp_id not in redis_stats['camp_groups']:
                        redis_stats['camp_groups'][camp_id] = {}
                    
                    redis_stats['camp_groups'][camp_id][group_id] = {
                        'embeddings_count': hash_len,
                        'ttl_seconds': ttl,
                        'expires_at': None if ttl == -1 else int(datetime.utcnow().timestamp()) + ttl
                    }
            
            return redis_stats
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                'error': str(e),
                'total_keys': 0,
                'keys': [],
                'camp_groups': {}
            }
    
    def clear_cache(self, camper_id: Optional[str] = None):
        """
        Clear cache (all or specific camper)
        
        Args:
            camper_id: Camper to remove (None = clear all)
        """
        if camper_id:
            if camper_id in self.cache:
                del self.cache[camper_id]
            if camper_id in self.metadata:
                del self.metadata[camper_id]
            logger.info(f"Cleared cache for camper: {camper_id}")
        else:
            self.cache.clear()
            self.metadata.clear()
            logger.info("Cleared entire cache")
    
    def delete_embedding(self, camper_id: str, delete_from_disk: bool = True) -> bool:
        """
        Delete embedding from cache and optionally from disk
        
        Args:
            camper_id: Camper identifier
            delete_from_disk: Whether to delete disk files
            
        Returns:
            Success boolean
        """
        try:
            # Remove from memory
            if camper_id in self.cache:
                del self.cache[camper_id]
            if camper_id in self.metadata:
                del self.metadata[camper_id]
            
            # Remove from disk
            if delete_from_disk:
                embedding_path = get_embedding_path(camper_id)
                if embedding_path.exists():
                    embedding_path.unlink()
                
                metadata_path = embedding_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"Deleted embedding for camper {camper_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting embedding for {camper_id}: {e}")
            return False
        
    def set_embeddings_redis(self, camp_id: int, group_id: int, embeddings: Dict[str, np.ndarray], expire_at: int):
        """
        Store all embeddings for a camp/group in Redis as a HASH, set TTL to expire_at (unix timestamp).
        """
        key = f"face:embeddings:camp:{camp_id}:group:{group_id}"
        pipe = redis_client.pipeline()
        for camper_id, embedding in embeddings.items():
            pipe.hset(key, camper_id, embedding.tobytes())
        ttl_seconds = max(0, expire_at - int(datetime.utcnow().timestamp()))
        pipe.expire(key, ttl_seconds)
        pipe.execute()
        logger.info(f"✅ Redis embeddings set for {key}, TTL {ttl_seconds}s")

    def get_embeddings_redis(self, camp_id: int, group_id: int) -> Dict[str, np.ndarray]:
        """
        Fetch all embeddings for a camp/group from Redis HASH, decode to numpy arrays.
        """
        key = f"face:embeddings:camp:{camp_id}:group:{group_id}"
        result = redis_client.hgetall(key)
        embeddings = {}
        for camper_id, emb_bytes in result.items():
            embeddings[camper_id.decode() if isinstance(camper_id, bytes) else camper_id] = np.frombuffer(emb_bytes, dtype=np.float32)
        logger.info(f"✅ Redis embeddings fetched for {key}: {len(embeddings)} campers")
        return embeddings

    def fetch_embeddings_for_recognition(self, camp_id: int, group_id: int, use_hot_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Optimized recognition read path: fetch embeddings from Redis, lazy decode, use RAM hot cache if enabled.
        """
        cache_key = f"camp_{camp_id}_group_{group_id}"
        # RAM hot cache
        if use_hot_cache and hasattr(self, 'hot_cache') and cache_key in self.hot_cache:
            logger.info(f"🔥 RAM hot cache hit for {cache_key}")
            return self.hot_cache[cache_key]
        # Redis fetch
        embeddings = self.get_embeddings_redis(camp_id, group_id)
        if use_hot_cache:
            if not hasattr(self, 'hot_cache'):
                self.hot_cache = {}
            self.hot_cache[cache_key] = embeddings
        return embeddings
