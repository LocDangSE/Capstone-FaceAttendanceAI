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

from config.settings import settings, get_embedding_path
from utils import get_now

logger = logging.getLogger(__name__)


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
        Load embeddings from a specific activity schedule folder
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
            loaded_count = 0
            
            logger.info(f"📥 Loading embeddings from {folder_path} ({len(image_files)} images)")
            
            for image_file in image_files:
                try:
                    # Extract camper ID from filename
                    # Format: avatar_21_avatar_uuid.jpg -> camper_id = 21
                    # Or: 21.jpg -> camper_id = 21
                    filename = image_file.stem
                    if filename.startswith('avatar_'):
                        # Extract number after first 'avatar_'
                        import re
                        match = re.match(r'avatar_(\d+)', filename)
                        if match:
                            camper_id = match.group(1)
                        else:
                            camper_id = filename
                    else:
                        camper_id = filename
                    
                    # Generate embedding if not cached
                    if camper_id not in self.cache:
                        embedding = self.generate_embedding(str(image_file))
                        if embedding is not None:
                            self.cache[camper_id] = embedding
                            self.metadata[camper_id] = {
                                'camper_id': camper_id,
                                'model': self.model_name,
                                'cached_at': get_now().isoformat(),
                                'source': str(folder_path)
                            }
                            loaded_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to load embedding for {image_file}: {e}")
            
            logger.info(f"✅ Loaded {loaded_count} embeddings into cache")
        
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
        Generate face embedding from image using DeepFace
        
        Args:
            image_path: Path to face image
            
        Returns:
            512D numpy array or None on failure
        """
        try:
            logger.debug(f"Generating embedding for: {image_path}")
            
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
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
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
        Find best matching camper from cached embeddings
        
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
            
            best_camper_id = None
            best_distance = float('inf')
            
            # Compare against all cached embeddings
            for camper_id, cached_embedding in self.cache.items():
                distance = self.compare_embeddings(query_embedding, cached_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_camper_id = camper_id
            
            # Check if best match is within threshold
            if best_distance <= threshold:
                confidence = 1.0 - best_distance
                logger.debug(f"Best match: {best_camper_id} (distance={best_distance:.4f}, confidence={confidence:.4f})")
                return best_camper_id, best_distance, confidence
            else:
                logger.debug(f"Best distance {best_distance:.4f} exceeds threshold {threshold}")
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
