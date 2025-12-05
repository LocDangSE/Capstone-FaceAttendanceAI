"""
Lazy-Loading Embedding Cache (Railway-Optimized)
Minimizes memory usage by loading embeddings on-demand and using disk caching
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from datetime import datetime
import asyncio
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger(__name__)


class LazyEmbeddingCache:
    """
    Memory-efficient embedding cache for Railway's limited resources.
    Features:
    - Lazy loading: Only loads embeddings when requested
    - LRU eviction: Keeps only N most recently used embeddings in memory
    - Disk-first: Always reads from disk, minimal memory footprint
    - Batch processing: Loads embeddings in small batches
    """
    
    def __init__(self, max_memory_items: int = 50):
        """
        Initialize lazy embedding cache
        
        Args:
            max_memory_items: Maximum number of embeddings to keep in memory (LRU)
        """
        self.model = settings.DEEPFACE_MODEL
        self.distance_metric = settings.DEEPFACE_DISTANCE_METRIC
        self.embedding_dir = Path("embeddings")
        self.embedding_dir.mkdir(exist_ok=True)
        
        # LRU cache in memory (OrderedDict for O(1) access and eviction)
        self.memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_memory_items = max_memory_items
        
        # Metadata cache (lightweight, always in memory)
        self.metadata: Dict[str, dict] = {}
        
        # Track cache statistics
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_loaded": 0
        }
        
        logger.info(f"✅ LazyEmbeddingCache initialized (max_memory={max_memory_items})")
    
    def _get_embedding_path(self, camper_id: str) -> Tuple[Path, Path]:
        """Get file paths for embedding and metadata"""
        npy_path = self.embedding_dir / f"{camper_id}.npy"
        json_path = self.embedding_dir / f"{camper_id}.json"
        return npy_path, json_path
    
    def _evict_if_needed(self):
        """Evict oldest item from memory cache if over limit"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove least recently used (first item in OrderedDict)
            evicted_id = next(iter(self.memory_cache))
            del self.memory_cache[evicted_id]
            self.stats["evictions"] += 1
            logger.debug(f"🗑️  Evicted {evicted_id} from memory cache")
    
    def get(self, camper_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for camper (lazy load from disk if not in memory)
        
        Args:
            camper_id: Unique camper identifier
            
        Returns:
            Embedding vector or None if not found
        """
        # Check memory cache first
        if camper_id in self.memory_cache:
            self.stats["memory_hits"] += 1
            # Move to end (mark as recently used)
            self.memory_cache.move_to_end(camper_id)
            return self.memory_cache[camper_id]
        
        # Load from disk
        npy_path, _ = self._get_embedding_path(camper_id)
        if npy_path.exists():
            try:
                embedding = np.load(str(npy_path))
                
                # Add to memory cache with LRU eviction
                self._evict_if_needed()
                self.memory_cache[camper_id] = embedding
                
                self.stats["disk_hits"] += 1
                logger.debug(f"📂 Loaded {camper_id} from disk")
                return embedding
            except Exception as e:
                logger.error(f"Failed to load embedding for {camper_id}: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(self, camper_id: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        """
        Save embedding to disk and optionally add to memory cache
        
        Args:
            camper_id: Unique camper identifier
            embedding: Face embedding vector
            metadata: Optional metadata (name, group, etc.)
        """
        npy_path, json_path = self._get_embedding_path(camper_id)
        
        try:
            # Save to disk first (primary storage)
            np.save(str(npy_path), embedding)
            
            # Save metadata
            if metadata:
                with open(json_path, 'w') as f:
                    json.dump(metadata, f)
                self.metadata[camper_id] = metadata
            
            # Add to memory cache with eviction
            self._evict_if_needed()
            self.memory_cache[camper_id] = embedding
            
            self.stats["total_loaded"] += 1
            logger.debug(f"💾 Saved {camper_id} to disk and memory")
            
        except Exception as e:
            logger.error(f"Failed to save embedding for {camper_id}: {e}")
    
    def get_batch(self, camper_ids: List[str], batch_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Get multiple embeddings in batches (memory-efficient)
        
        Args:
            camper_ids: List of camper IDs
            batch_size: Number of embeddings to load per batch
            
        Returns:
            Dictionary of camper_id -> embedding
        """
        results = {}
        
        for i in range(0, len(camper_ids), batch_size):
            batch = camper_ids[i:i + batch_size]
            for camper_id in batch:
                embedding = self.get(camper_id)
                if embedding is not None:
                    results[camper_id] = embedding
        
        return results
    
    async def get_batch_async(self, camper_ids: List[str], batch_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Async version of get_batch for non-blocking I/O
        """
        results = {}
        
        for i in range(0, len(camper_ids), batch_size):
            batch = camper_ids[i:i + batch_size]
            # Load batch in background
            await asyncio.sleep(0)  # Yield to event loop
            
            for camper_id in batch:
                embedding = self.get(camper_id)
                if embedding is not None:
                    results[camper_id] = embedding
        
        return results
    
    def list_cached_ids(self) -> List[str]:
        """Get list of all camper IDs with cached embeddings on disk"""
        return [
            p.stem for p in self.embedding_dir.glob("*.npy")
        ]
    
    def get_metadata(self, camper_id: str) -> Optional[dict]:
        """Get metadata for a camper"""
        if camper_id in self.metadata:
            return self.metadata[camper_id]
        
        _, json_path = self._get_embedding_path(camper_id)
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    self.metadata[camper_id] = metadata
                    return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata for {camper_id}: {e}")
        
        return None
    
    def preload_metadata_only(self):
        """
        Load only metadata (not embeddings) for all cached campers.
        This is memory-efficient and allows fast lookups.
        """
        count = 0
        for json_path in self.embedding_dir.glob("*.json"):
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    camper_id = json_path.stem
                    self.metadata[camper_id] = metadata
                    count += 1
            except Exception as e:
                logger.error(f"Failed to preload metadata from {json_path}: {e}")
        
        logger.info(f"📋 Preloaded metadata for {count} campers")
    
    def clear_memory(self):
        """Clear memory cache (keep disk cache intact)"""
        count = len(self.memory_cache)
        self.memory_cache.clear()
        logger.info(f"🧹 Cleared {count} embeddings from memory (disk cache preserved)")
    
    def clear_all(self):
        """Clear both memory and disk cache"""
        # Clear memory
        self.memory_cache.clear()
        self.metadata.clear()
        
        # Clear disk
        count = 0
        for file in self.embedding_dir.glob("*"):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file}: {e}")
        
        logger.info(f"🗑️  Cleared {count} files from disk cache")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "memory_cached": len(self.memory_cache),
            "disk_cached": len(list(self.embedding_dir.glob("*.npy"))),
            "max_memory_items": self.max_memory_items,
            "metadata_loaded": len(self.metadata),
            **self.stats
        }
    
    def optimize_memory(self):
        """
        Optimize memory usage by keeping only most frequently accessed embeddings
        """
        if len(self.memory_cache) > self.max_memory_items // 2:
            # Keep only half, remove least recently used
            to_remove = len(self.memory_cache) - (self.max_memory_items // 2)
            for _ in range(to_remove):
                if self.memory_cache:
                    removed_id = next(iter(self.memory_cache))
                    del self.memory_cache[removed_id]
                    self.stats["evictions"] += 1
            
            logger.info(f"♻️  Optimized memory: removed {to_remove} embeddings")
