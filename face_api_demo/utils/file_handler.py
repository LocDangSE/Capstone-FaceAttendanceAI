"""
File Handler Utilities
Helper functions for file operations
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple
import logging
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from config.settings import settings
from utils.validators import sanitize_filename

logger = logging.getLogger(__name__)


class FileHandler:
    """Handler for file operations"""
    
    @staticmethod
    def save_uploaded_file(
        file: FileStorage,
        folder = None,
        custom_filename: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Save uploaded file with unique name
        
        Args:
            file: Uploaded file object
            folder: Destination folder (default: TEMP_FOLDER) - can be Path or str
            custom_filename: Custom filename (if None, generates unique name)
            
        Returns:
            Tuple of (file_path, error_message)
        """
        try:
            if not file or file.filename == '':
                return None, "No file provided"
            
            # Determine destination folder
            if folder is None:
                folder = settings.TEMP_FOLDER
            elif isinstance(folder, str):
                # Convert string to Path relative to TEMP_FOLDER
                folder = settings.TEMP_FOLDER / folder
            elif not isinstance(folder, Path):
                folder = Path(folder)
            
            # Ensure folder exists
            folder.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            if custom_filename:
                filename = sanitize_filename(custom_filename)
            else:
                # Generate unique filename
                file_ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
                filename = f"{uuid.uuid4()}.{file_ext}"
            
            # Save file
            file_path = str(folder / filename)
            file.save(file_path)
            
            logger.debug(f"File saved: {file_path}")
            return file_path, None
        
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return None, str(e)
    
    @staticmethod
    def cleanup_file(file_path: Optional[str]):
        """
        Delete a file safely
        
        Args:
            file_path: Path to file to delete
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"File deleted: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")
    
    @staticmethod
    def cleanup_folder(folder_path: Path, recursive: bool = False):
        """
        Delete a folder and its contents
        
        Args:
            folder_path: Path to folder
            recursive: Whether to delete recursively
        """
        try:
            if folder_path.exists():
                if recursive:
                    shutil.rmtree(str(folder_path))
                else:
                    folder_path.rmdir()
                logger.debug(f"Folder deleted: {folder_path}")
        except Exception as e:
            logger.warning(f"Could not delete folder {folder_path}: {e}")
    
    @staticmethod
    def copy_file(source: str, destination: str) -> Tuple[bool, Optional[str]]:
        """
        Copy a file
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Ensure destination directory exists
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            logger.debug(f"File copied: {source} -> {destination}")
            return True, None
        
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return False, str(e)
    
    @staticmethod
    def move_file(source: str, destination: str) -> Tuple[bool, Optional[str]]:
        """
        Move a file
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Ensure destination directory exists
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(source, destination)
            logger.debug(f"File moved: {source} -> {destination}")
            return True, None
        
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return False, str(e)
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """
        Get file size in bytes
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting file size: {e}")
            return 0
    
    @staticmethod
    def ensure_directory(directory: Path):
        """
        Ensure directory exists
        
        Args:
            directory: Directory path
        """
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def cleanup_temp_folder(older_than_hours: Optional[int] = None) -> Tuple[int, int, Optional[str]]:
        """
        Clean up files in temp folder
        
        Args:
            older_than_hours: Only delete files older than this many hours (None = delete all)
        
        Returns:
            Tuple of (deleted_count, failed_count, error_message)
        """
        try:
            temp_folder = settings.TEMP_FOLDER
            if not temp_folder.exists():
                logger.warning(f"Temp folder does not exist: {temp_folder}")
                return 0, 0, "Temp folder does not exist"
            
            deleted_count = 0
            failed_count = 0
            
            # Calculate cutoff time if filtering by age
            import time
            cutoff_time = None
            if older_than_hours is not None:
                cutoff_time = time.time() - (older_than_hours * 3600)
            
            # Delete files in temp folder
            for item in temp_folder.iterdir():
                try:
                    # Check age if filtering
                    if cutoff_time is not None:
                        item_mtime = item.stat().st_mtime
                        if item_mtime > cutoff_time:
                            continue  # Skip newer files
                    
                    if item.is_file():
                        item.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted temp file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(str(item))
                        deleted_count += 1
                        logger.debug(f"Deleted temp directory: {item.name}")
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to delete {item.name}: {e}")
            
            logger.info(f"Temp folder cleanup: {deleted_count} deleted, {failed_count} failed")
            return deleted_count, failed_count, None
        
        except Exception as e:
            logger.error(f"Error cleaning temp folder: {e}")
            return 0, 0, str(e)
