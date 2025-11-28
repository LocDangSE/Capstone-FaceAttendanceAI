"""
Optimized Image Processing Service
High-performance image preprocessing and face detection
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from deepface import DeepFace

from config.settings import settings

logger = logging.getLogger(__name__)


class OptimizedImageProcessor:
    """
    Optimized service for image preprocessing with performance improvements:
    - Smart resolution reduction
    - Early face detection before embedding
    - Batch processing support
    - Memory-efficient operations
    """
    
    def __init__(self):
        """Initialize optimized image processor"""
        self.resize_width = settings.IMAGE_RESIZE_WIDTH
        self.resize_height = settings.IMAGE_RESIZE_HEIGHT
        self.quality = settings.IMAGE_QUALITY
        self.detector_backend = settings.DEEPFACE_DETECTOR
        
        # Optimization: Cache detector for reuse
        self._detector_cache = None
        
        logger.info(f"OptimizedImageProcessor initialized: {self.resize_width}x{self.resize_height}, quality={self.quality}")
    
    def preprocess_image(self, image_path: str, output_path: Optional[str] = None, target_size: Optional[Tuple[int, int]] = None) -> str:
        """
        Optimized image preprocessing with early resizing
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image
            target_size: Optional (width, height) tuple for custom sizing
            
        Returns:
            Path to processed image
        """
        try:
            logger.debug(f"Preprocessing image: {image_path}")
            
            # Use OpenCV for faster loading
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            original_height, original_width = img.shape[:2]
            
            # Calculate target dimensions
            if target_size:
                new_width, new_height = target_size
            else:
                aspect_ratio = original_width / original_height
                
                if aspect_ratio > 1:  # Wider than tall
                    new_width = self.resize_width
                    new_height = int(new_width / aspect_ratio)
                else:  # Taller than wide
                    new_height = self.resize_height
                    new_width = int(new_height * aspect_ratio)
            
            # Resize with efficient interpolation
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Determine output path
            if output_path is None:
                output_path = image_path
            
            # Save with OpenCV (faster than PIL for JPEG)
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            
            logger.debug(f"Image preprocessed: {original_width}x{original_height} -> {new_width}x{new_height}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def smart_resize_for_detection(self, image_path: str, max_dimension: int = 640) -> str:
        """
        Smart resize specifically optimized for face detection
        Reduces resolution while preserving face detectability
        
        Args:
            image_path: Path to input image
            max_dimension: Maximum width or height
            
        Returns:
            Path to resized image
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            height, width = img.shape[:2]
            
            # Only resize if image is larger than max_dimension
            if max(height, width) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(image_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                logger.debug(f"Smart resize: {width}x{height} -> {new_width}x{new_height}")
            
            return image_path
        
        except Exception as e:
            logger.error(f"Error in smart resize: {e}")
            return image_path
    
    def crop_face_region(
        self,
        image_path: str,
        face_region: Dict[str, int],
        padding: float = 0.1,
        output_path: Optional[str] = None
    ) -> str:
        """
        Optimized face cropping with OpenCV
        
        Args:
            image_path: Path to source image
            face_region: Dictionary with 'x', 'y', 'width', 'height'
            padding: Percentage of padding to add
            output_path: Path to save cropped face
            
        Returns:
            Path to cropped face image
        """
        try:
            # Load with OpenCV (faster)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            h, w = img.shape[:2]
            
            # Extract face region with padding
            x = face_region['x']
            y = face_region['y']
            face_w = face_region['width']
            face_h = face_region['height']
            
            # Calculate padding
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)
            
            # Apply padding with boundary checks
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + face_w + pad_w)
            y2 = min(h, y + face_h + pad_h)
            
            # Crop face (numpy slicing is very fast)
            face_img = img[y1:y2, x1:x2]
            
            # Generate output path if not provided
            if output_path is None:
                base_name = Path(image_path).stem
                output_path = str(settings.TEMP_FOLDER / f"{base_name}_face.jpg")
            
            # Save cropped face
            cv2.imwrite(output_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.debug(f"Face cropped: ({x1},{y1}) to ({x2},{y2})")
            return output_path
        
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            raise
    
    def extract_faces_from_frame(
        self,
        image_path: str,
        min_confidence: float = 0.9
    ) -> List[Dict]:
        """
        Optimized face extraction with smart preprocessing
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum detection confidence threshold
            
        Returns:
            List of face dictionaries with region and confidence
        """
        try:
            logger.debug(f"Extracting faces from: {image_path}")
            
            # Smart resize for faster detection
            processed_path = self.smart_resize_for_detection(image_path, max_dimension=800)
            
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(
                img_path=processed_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Filter and format results
            detected_faces = []
            for idx, face_obj in enumerate(faces):
                confidence = face_obj.get('confidence', 0)
                
                if confidence >= min_confidence:
                    face_data = {
                        'index': idx,
                        'confidence': round(confidence, 4),
                        'region': {
                            'x': int(face_obj['facial_area']['x']),
                            'y': int(face_obj['facial_area']['y']),
                            'width': int(face_obj['facial_area']['w']),
                            'height': int(face_obj['facial_area']['h'])
                        },
                        'face_array': face_obj['face']  # Normalized face array
                    }
                    detected_faces.append(face_data)
            
            logger.info(f"Extracted {len(detected_faces)} face(s) with confidence >= {min_confidence}")
            return detected_faces
        
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            return []
    
    def validate_single_face(self, image_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Optimized validation for single face images
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, message, face_data)
        """
        try:
            faces = self.extract_faces_from_frame(image_path, min_confidence=0.7)
            
            if not faces:
                return False, "No face detected in image", None
            
            if len(faces) > 1:
                return False, f"Multiple faces detected ({len(faces)}). Please upload a photo with a single face.", None
            
            face = faces[0]
            if face['confidence'] < 0.9:
                return False, f"Face confidence too low ({face['confidence']:.2f}). Please upload a clearer photo.", None
            
            return True, "Valid single face detected", face
        
        except Exception as e:
            logger.error(f"Error validating single face: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def batch_preprocess_images(self, image_paths: List[str], output_dir: Path) -> List[str]:
        """
        Batch preprocess multiple images efficiently
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save processed images
            
        Returns:
            List of processed image paths
        """
        processed_paths = []
        
        for i, img_path in enumerate(image_paths):
            try:
                output_path = output_dir / f"processed_{i}_{Path(img_path).name}"
                processed = self.preprocess_image(img_path, str(output_path))
                processed_paths.append(processed)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        logger.info(f"Batch processed {len(processed_paths)}/{len(image_paths)} images")
        return processed_paths
    
    def estimate_face_quality(self, face_region: Dict) -> float:
        """
        Estimate face quality based on size and position
        
        Args:
            face_region: Face region dictionary
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        try:
            width = face_region['width']
            height = face_region['height']
            
            # Larger faces generally have better quality
            size_score = min(1.0, (width * height) / (200 * 200))
            
            # Face should be roughly square for best quality
            aspect_ratio = width / height if height > 0 else 1.0
            aspect_score = 1.0 - abs(1.0 - aspect_ratio)
            
            quality = (size_score * 0.7) + (aspect_score * 0.3)
            return round(quality, 3)
        
        except Exception as e:
            logger.error(f"Error estimating face quality: {e}")
            return 0.5
