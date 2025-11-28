"""
Input Validation Utilities
Helper functions for validating user inputs
"""

import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
import logging
from werkzeug.datastructures import FileStorage

from config.settings import settings

logger = logging.getLogger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        
    Returns:
        True if extension is allowed
    """
    if not filename:
        return False
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in settings.ALLOWED_EXTENSIONS


def validate_image_file(file: FileStorage) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not file or file.filename == '':
            return False, "No file provided"
        
        # Check file extension
        if not allowed_file(file.filename):
            allowed = ', '.join(settings.ALLOWED_EXTENSIONS)
            return False, f"File type not allowed. Allowed types: {allowed}"
        
        # Check file size (if we can read it)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > settings.IMAGE_MAX_SIZE:
            max_mb = settings.IMAGE_MAX_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_mb:.1f}MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, None
    
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False, f"File validation error: {str(e)}"


def validate_uuid(uuid_string: str) -> Tuple[bool, Optional[str]]:
    """
    Validate UUID format
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        uuid.UUID(uuid_string)
        return True, None
    except ValueError:
        return False, "Invalid UUID format"
    except Exception as e:
        return False, f"UUID validation error: {str(e)}"


def validate_camper_id(camper_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate camper ID format (should be UUID)
    
    Args:
        camper_id: Camper ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not camper_id or not camper_id.strip():
        return False, "Camper ID cannot be empty"
    
    if len(camper_id) > 100:
        return False, "Camper ID too long (max 100 characters)"
    
    # Check for invalid characters (optional - adjust as needed)
    # Allow alphanumeric, hyphens, underscores
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', camper_id):
        return False, "Camper ID can only contain letters, numbers, hyphens, and underscores"
    
    return True, None


def validate_activity_schedule_id(activity_schedule_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate activity schedule ID format (should be UUID)
    
    Args:
        activity_schedule_id: Activity schedule ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not activity_schedule_id or not activity_schedule_id.strip():
        return False, "Activity schedule ID cannot be empty"
    
    if len(activity_schedule_id) > 100:
        return False, "Activity schedule ID too long (max 100 characters)"
    
    return True, None


def validate_confidence_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
    """
    Validate confidence threshold value
    
    Args:
        threshold: Threshold value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not 0.0 <= threshold <= 1.0:
        return False, "Confidence threshold must be between 0.0 and 1.0"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory components
    filename = os.path.basename(filename)
    
    # Remove any non-alphanumeric characters except dots, hyphens, underscores
    import re
    filename = re.sub(r'[^\w\.-]', '_', filename)
    
    return filename


def validate_session_id(session_id: str) -> Tuple[bool, Optional[str]]:
    """
    Validate session ID format
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Session IDs should be UUIDs
    return validate_uuid(session_id)
