"""
Image Processing Service
Handles image preprocessing, optimization, and face extraction
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


class ImageProcessor:
    """
    Service for optimizing images before face recognition
    Provides preprocessing, face cropping, and extraction methods
    """
    
    def __init__(self):
        """Initialize Image Processor with configuration"""
        self.resize_width = settings.IMAGE_RESIZE_WIDTH
        self.resize_height = settings.IMAGE_RESIZE_HEIGHT
        self.quality = settings.IMAGE_QUALITY
        self.detector_backend = settings.DEEPFACE_DETECTOR
        
        logger.info(f"ImageProcessor initialized: {self.resize_width}x{self.resize_height}, quality={self.quality}")
    
    def preprocess_image(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess image for optimal recognition performance
        - Resize to target dimensions (maintain aspect ratio)
        - Convert to RGB
        - Compress to JPEG at specified quality
        
        Args:
            image_path: Path to input image
            output_path: Path to save processed image (if None, overwrites input)
            
        Returns:
            Path to processed image
        """
        try:
            logger.debug(f"Preprocessing image: {image_path}")
            
            # Load image with PIL
            img = Image.open(image_path)
            
            # Convert to RGB (remove alpha channel if present)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate aspect ratio and resize
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            if aspect_ratio > 1:  # Wider than tall
                new_width = self.resize_width
                new_height = int(new_width / aspect_ratio)
            else:  # Taller than wide
                new_height = self.resize_height
                new_width = int(new_height * aspect_ratio)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Determine output path
            if output_path is None:
                output_path = image_path
            
            # Save with compression
            img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
            
            logger.debug(f"Image preprocessed: {original_width}x{original_height} -> {new_width}x{new_height}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def crop_face_region(
        self,
        image_path: str,
        face_region: Dict[str, int],
        padding: float = 0.1,
        output_path: Optional[str] = None
    ) -> str:
        """
        Crop face region from image with optional padding
        
        Args:
            image_path: Path to source image
            face_region: Dictionary with 'x', 'y', 'width', 'height'
            padding: Percentage of padding to add (0.1 = 10%)
            output_path: Path to save cropped face (if None, generates temp path)
            
        Returns:
            Path to cropped face image
        """
        try:
            # Load image
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
            
            # Crop face
            face_img = img[y1:y2, x1:x2]
            
            # Generate output path if not provided
            if output_path is None:
                base_name = Path(image_path).stem
                output_path = str(settings.TEMP_FOLDER / f"{base_name}_face.jpg")
            
            # Save cropped face
            cv2.imwrite(output_path, face_img)
            
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
        Extract all faces from an image with high accuracy
        Uses RetinaFace detector for better accuracy
        
        Args:
            image_path: Path to input image
            min_confidence: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            List of face dictionaries with region and confidence
        """
        try:
            logger.debug(f"Extracting faces from: {image_path}")
            
            # Use DeepFace for face detection
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Filter by confidence and format results
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
        Validate that image contains exactly one face
        Used for student registration
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (is_valid, message, face_data)
        """
        try:
            faces = self.extract_faces_from_frame(image_path, min_confidence=0.5)
            
            if len(faces) == 0:
                return False, "No face detected in the image", None
            elif len(faces) > 1:
                return False, f"Multiple faces detected ({len(faces)}). Please use image with single face", None
            else:
                return True, "Single face detected", faces[0]
        
        except Exception as e:
            logger.error(f"Error validating face: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def enhance_image(self, image_path: str) -> str:
        """
        Enhance image quality for better recognition
        - Adjust brightness and contrast
        - Apply sharpening
        
        Args:
            image_path: Path to image
            
        Returns:
            Path to enhanced image (overwrites original)
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Save enhanced image
            cv2.imwrite(image_path, enhanced)
            
            logger.debug(f"Image enhanced: {image_path}")
            return image_path
        
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            # Return original path if enhancement fails
            return image_path
