"""
Supabase Service
Clean integration for cloud storage management
"""

import os
import logging
from supabase import create_client, Client
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class SupabaseService:
    """
    Service for Supabase cloud storage operations
    Handles file uploads, downloads, and session management
    """
    
    def __init__(self):
        """Initialize Supabase service"""
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.bucket = settings.SUPABASE_BUCKET
        self.enabled = settings.SUPABASE_ENABLED
        self.client: Optional[Client] = None
        
        if self.enabled and self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                logger.info(f"✅ Supabase client initialized (bucket: {self.bucket})")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase: {e}")
                self.enabled = False
        else:
            logger.info("ℹ️  Supabase storage disabled or not configured")
    
    def is_enabled(self) -> bool:
        """Check if Supabase is enabled and configured"""
        return self.enabled and self.client is not None
    
    def upload_file(
        self,
        local_path: str,
        remote_path: str,
        content_type: str = "image/jpeg"
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Upload a file to Supabase Storage
        
        Args:
            local_path: Path to local file
            remote_path: Destination path in bucket (e.g., "students/uuid.jpg")
            content_type: MIME type of file
            
        Returns:
            Tuple of (success, public_url, error_message)
        """
        if not self.is_enabled():
            return False, None, "Supabase storage is not enabled"
        
        try:
            # Read file
            with open(local_path, 'rb') as f:
                file_data = f.read()
            
            # Delete existing file if present
            try:
                self.client.storage.from_(self.bucket).remove([remote_path])
            except:
                pass  # File doesn't exist, continue
            
            # Upload to Supabase
            response = self.client.storage.from_(self.bucket).upload(
                path=remote_path,
                file=file_data,
                file_options={
                    "content-type": content_type,
                    "upsert": "true"
                }
            )
            
            # Get public URL
            public_url = self.get_public_url(remote_path)
            
            logger.info(f"✅ Uploaded to Supabase: {remote_path}")
            return True, public_url, None
        
        except Exception as e:
            error_msg = f"Failed to upload to Supabase: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def download_file(
        self,
        remote_path: str,
        local_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Download a file from Supabase Storage
        
        Args:
            remote_path: Path to file in bucket
            local_path: Local destination path
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_enabled():
            return False, "Supabase storage is not enabled"
        
        try:
            # Download file
            file_data = self.client.storage.from_(self.bucket).download(remote_path)
            
            if not file_data:
                return False, f"File not found: {remote_path}"
            
            # Save to local path
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            
            logger.info(f"✅ Downloaded from Supabase: {remote_path} -> {local_path}")
            return True, None
        
        except Exception as e:
            error_msg = f"Failed to download from Supabase: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
    
    def delete_file(self, remote_path: str) -> Tuple[bool, Optional[str]]:
        """
        Delete a file from Supabase Storage
        
        Args:
            remote_path: Path to file in bucket
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_enabled():
            return False, "Supabase storage is not enabled"
        
        try:
            self.client.storage.from_(self.bucket).remove([remote_path])
            logger.info(f"✅ Deleted from Supabase: {remote_path}")
            return True, None
        
        except Exception as e:
            error_msg = f"Failed to delete from Supabase: {str(e)}"
            logger.warning(f"⚠️  {error_msg}")
            return False, error_msg
    
    def get_public_url(self, remote_path: str) -> str:
        """
        Get public URL for a file in storage
        
        Args:
            remote_path: Path to file in bucket
            
        Returns:
            Public URL string
        """
        if not self.is_enabled():
            return ""
        
        try:
            public_url = self.client.storage.from_(self.bucket).get_public_url(remote_path)
            return public_url
        except Exception as e:
            logger.error(f"❌ Failed to get public URL: {e}")
            return ""
    
    def list_files(self, folder: str = "") -> List[Dict]:
        """
        List files in a folder
        
        Args:
            folder: Folder path in bucket
            
        Returns:
            List of file objects
        """
        if not self.is_enabled():
            return []
        
        try:
            files = self.client.storage.from_(self.bucket).list(folder)
            return files
        except Exception as e:
            logger.error(f"❌ Failed to list files: {e}")
            return []
    
    def save_camper_face(
        self,
        local_path: str,
        camper_id: str,
        filename: str
    ) -> Tuple[str, bool]:
        """
        Save camper face image to Supabase
        
        Args:
            local_path: Path to local temporary file
            camper_id: Camper ID (UUID)
            filename: Desired filename
            
        Returns:
            Tuple of (public_url, success)
            
        Raises:
            Exception: If Supabase is not enabled or upload fails
        """
        if not self.is_enabled():
            error_msg = "❌ Supabase storage is not enabled. Please configure SUPABASE_ENABLED=true in .env"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        remote_path = f"campers/{camper_id}/{filename}"
        
        logger.info(f"📤 Uploading camper face to Supabase: {remote_path}")
        success, public_url, error = self.upload_file(
            local_path=local_path,
            remote_path=remote_path,
            content_type="image/jpeg"
        )
        
        if success and public_url:
            logger.info(f"✅ Successfully saved to Supabase: {public_url}")
            return public_url, True
        else:
            error_msg = f"❌ FAILED to upload to Supabase: {error}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def sync_activity_schedule_campers(
        self,
        activity_schedule_id: str,
        local_folder: Path
    ) -> Tuple[int, str]:
        """
        Sync activity schedule face database from Supabase to local storage
        Downloads all camper images for an activity schedule
        
        Args:
            activity_schedule_id: Activity schedule identifier (UUID)
            local_folder: Local folder to save images
            
        Returns:
            Tuple of (camper_count, message)
        """
        if not self.is_enabled():
            return 0, "Supabase not enabled"
        
        try:
            logger.info(f"📥 Syncing activity schedule {activity_schedule_id} from Supabase...")
            
            # Create activity schedule folder
            local_folder.mkdir(parents=True, exist_ok=True)
            
            # List all entries in campers/
            items = self.list_files("campers/")
            
            IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
            
            # Separate folder-style and root-level images
            folder_camper_ids = [
                item.get('name', '') for item in items
                if item.get('name') and not item.get('name', '').lower().endswith(IMAGE_EXTS)
                and '.' not in item.get('name', '')
            ]
            
            root_images = [
                item.get('name', '') for item in items
                if item.get('name', '').lower().endswith(IMAGE_EXTS)
            ]
            
            if not folder_camper_ids and not root_images:
                return 0, "No campers found in Supabase"
            
            logger.info(f"  Found {len(folder_camper_ids)} camper folders and {len(root_images)} root images")
            
            success_count = 0
            
            # Download from camper folders
            for camper_id in folder_camper_ids:
                try:
                    camper_files = self.list_files(f"campers/{camper_id}/")
                    image_files = [
                        f for f in camper_files
                        if f.get('name', '').lower().endswith(IMAGE_EXTS)
                    ]
                    
                    if not image_files:
                        continue
                    
                    # Download first image
                    image_file = image_files[0]['name']
                    remote_path = f"campers/{camper_id}/{image_file}"
                    
                    # Save directly in activity schedule folder with camper ID as filename
                    local_path = str(local_folder / f"{camper_id}.jpg")
                    
                    success, error = self.download_file(remote_path, local_path)
                    if success:
                        success_count += 1
                        logger.info(f"  ✅ Downloaded {camper_id}.jpg")
                
                except Exception as e:
                    logger.error(f"  ❌ Error downloading camper {camper_id}: {e}")
            
            # Download root images
            for filename in root_images:
                try:
                    base = os.path.basename(filename)
                    name_wo_ext, _ = os.path.splitext(base)
                    # Use the full filename (without extension) as camper_id - it should be a GUID
                    camper_id = name_wo_ext
                    
                    remote_path = f"campers/{filename}"
                    # Save directly in activity schedule folder with camper ID as filename
                    local_path = str(local_folder / f"{camper_id}.jpg")
                    
                    success, error = self.download_file(remote_path, local_path)
                    if success:
                        success_count += 1
                        logger.info(f"  ✅ Downloaded {camper_id}.jpg from root")
                
                except Exception as e:
                    logger.error(f"  ❌ Error downloading root image {filename}: {e}")
            
            message = f"Synced {success_count} campers from Supabase"
            logger.info(f"  ✅ {message}")
            return success_count, message
        
        except Exception as e:
            error_msg = f"Sync error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return 0, error_msg
    
    def sync_specific_campers(
        self,
        camper_ids: List[str],
        local_folder: Path
    ) -> Tuple[int, str]:
        """
        Download face images for specific campers only (selective sync)
        Used for activity-specific attendance to avoid downloading all campers
        
        Args:
            camper_ids: List of camper IDs (UUIDs) enrolled in the activity
            local_folder: Local folder to save images
            
        Returns:
            Tuple of (downloaded_count, message)
        """
        if not self.is_enabled():
            return 0, "Supabase not enabled"
        
        try:
            logger.info(f"📥 Downloading faces for {len(camper_ids)} specific campers...")
            
            # Create folder
            local_folder.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            failed_campers = []
            
            for camper_id in camper_ids:
                try:
                    # Try to download from camper's folder first
                    camper_folder = f"campers/{camper_id}/"
                    camper_files = self.list_files(camper_folder)
                    
                    IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
                    image_files = [
                        f for f in camper_files
                        if f.get('name', '').lower().endswith(IMAGE_EXTS)
                    ]
                    
                    if image_files:
                        # Download first image from folder
                        image_file = image_files[0]['name']
                        remote_path = f"{camper_folder}{image_file}"
                        local_path = str(local_folder / f"{camper_id}.jpg")
                        
                        success, error = self.download_file(remote_path, local_path)
                        if success:
                            success_count += 1
                            logger.info(f"  ✅ {camper_id}.jpg")
                        else:
                            failed_campers.append(camper_id)
                            logger.warning(f"  ⚠️ Failed to download {camper_id}: {error}")
                    else:
                        # Try direct file: campers/{camper_id}.jpg
                        remote_path = f"campers/{camper_id}.jpg"
                        local_path = str(local_folder / f"{camper_id}.jpg")
                        
                        success, error = self.download_file(remote_path, local_path)
                        if success:
                            success_count += 1
                            logger.info(f"  ✅ {camper_id}.jpg (direct)")
                        else:
                            failed_campers.append(camper_id)
                            logger.warning(f"  ⚠️ Camper {camper_id} not found in Supabase")
                
                except Exception as e:
                    failed_campers.append(camper_id)
                    logger.error(f"  ❌ Error downloading camper {camper_id}: {e}")
            
            message = f"Downloaded {success_count}/{len(camper_ids)} campers"
            if failed_campers:
                message += f" ({len(failed_campers)} failed: {', '.join(failed_campers[:5])}{'...' if len(failed_campers) > 5 else ''})"
            
            logger.info(f"  ✅ {message}")
            return success_count, message
        
        except Exception as e:
            error_msg = f"Sync error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return 0, error_msg
    
    def sync_camp_core_group(
        self,
        camp_id: str,
        camper_ids: List[str],
        local_folder: Path
    ) -> Tuple[int, str]:
        """
        Download core CamperGroup faces (used for all core activities in the camp)
        New structure: camp_id/camper_group_*/avatar_camperid_*.jpg
        
        Args:
            camp_id: Camp identifier
            camper_ids: List of all camper IDs in the camp's core group (not used in new structure)
            local_folder: Local folder to save images (e.g., face_database/{camp_id}/camper_groups)
            
        Returns:
            Tuple of (downloaded_count, message)
        """
        logger.info(f"📥 Downloading core group for Camp {camp_id} from cloud structure: camp_{camp_id}/camper_group_*/avatar_*")
        
        # Create folder
        local_folder.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        failed = []
        
        try:
            # List folders under camp_id (e.g., camp_17/)
            camp_folder_items = self.list_files(f"camp_{camp_id}/")
            
            # Find all camper_group folders (e.g., camper_group_14, camper_group_15)
            camper_group_folders = [
                item.get('name', '') for item in camp_folder_items
                if item.get('name', '').startswith('camper_group_')
            ]
            
            if not camper_group_folders:
                logger.warning(f"  ⚠️ No camper_group folders found under camp_{camp_id}/")
                return 0, "No camper_group folders found in cloud"
            
            logger.info(f"  Found {len(camper_group_folders)} camper_group folders")
            
            # Download from each camper_group folder
            for group_folder in camper_group_folders:
                try:
                    # List files in camper_group folder
                    group_path = f"camp_{camp_id}/{group_folder}/"
                    group_files = self.list_files(group_path)
                    
                    # Filter for avatar images (avatar_21_*.jpg)
                    avatar_files = [
                        f for f in group_files
                        if f.get('name', '').startswith('avatar_') and 
                           f.get('name', '').lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    
                    logger.info(f"  Processing {group_folder}: {len(avatar_files)} avatars")
                    
                    for avatar_file in avatar_files:
                        try:
                            filename = avatar_file.get('name', '')
                            # Extract camper ID from avatar_21_xxx.jpg format
                            import re
                            match = re.match(r'avatar_(\d+)_.*\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
                            
                            if not match:
                                logger.warning(f"    ⚠️ Skipping invalid filename format: {filename}")
                                continue
                            
                            camper_id = match.group(1)
                            
                            # Download from cloud
                            remote_path = f"{group_path}{filename}"
                            local_path = str(local_folder / f"{camper_id}.jpg")
                            
                            success, error = self.download_file(remote_path, local_path)
                            if success:
                                success_count += 1
                                logger.info(f"    ✅ Downloaded camper {camper_id}")
                            else:
                                failed.append(f"{camper_id} ({filename})")
                                logger.warning(f"    ⚠️ Failed: {camper_id} - {error}")
                        
                        except Exception as e:
                            logger.error(f"    ❌ Error processing {avatar_file.get('name', '')}: {e}")
                            failed.append(str(avatar_file.get('name', '')))
                
                except Exception as e:
                    logger.error(f"  ❌ Error processing group folder {group_folder}: {e}")
            
            message = f"Core group: {success_count} campers downloaded"
            if failed:
                message += f" ({len(failed)} failed)"
            
            logger.info(f"  ✅ {message}")
            return success_count, message
        
        except Exception as e:
            error_msg = f"Error syncing core group: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            return 0, error_msg
    
    def sync_optional_activity(
        self,
        camp_id: str,
        activity_id: str,
        camper_ids: List[str],
        local_folder: Path
    ) -> Tuple[int, str]:
        """
        Download faces for an optional activity (CamperActivity)
        New structure: camp_{campId}/camperactivity_{activityScheduleId}/avatar_{camperId}_{guid}.jpg
        
        Args:
            camp_id: Camp identifier
            activity_id: Activity schedule identifier (activityScheduleId)
            camper_ids: List of camper IDs enrolled in this optional activity (not used - downloads all from folder)
            local_folder: Local folder to save images (e.g., face_database/camp_{campId}/camper_activities/{activityId})
            
        Returns:
            Tuple of (downloaded_count, message)
        """
        logger.info(f"📥 Downloading optional activity {activity_id} for Camp {camp_id} from cloud structure: camp_{camp_id}/camperactivity_{activity_id}/avatar_*")
        
        # Create folder
        local_folder.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        failed = []
        
        try:
            # Download from camperactivity folder
            # Structure: camp_{camp_id}/camperactivity_{activity_id}/avatar_{camper_id}_{guid}.jpg
            activity_folder_path = f"camp_{camp_id}/camperactivity_{activity_id}/"
            
            # List files in camperactivity folder
            activity_files = self.list_files(activity_folder_path)
            
            if not activity_files:
                logger.warning(f"  ⚠️ No files found in {activity_folder_path}")
                return 0, f"No files found in camperactivity_{activity_id}"
            
            # Filter for avatar images (avatar_21_*.jpg)
            avatar_files = [
                f for f in activity_files
                if f.get('name', '').startswith('avatar_') and 
                   f.get('name', '').lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            logger.info(f"  Found {len(avatar_files)} avatars in camperactivity_{activity_id}")
            
            for avatar_file in avatar_files:
                try:
                    filename = avatar_file.get('name', '')
                    # Extract camper ID from avatar_21_xxx.jpg format
                    import re
                    match = re.match(r'avatar_(\d+)_.*\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
                    
                    if not match:
                        logger.warning(f"    ⚠️ Skipping invalid filename format: {filename}")
                        continue
                    
                    camper_id = match.group(1)
                    
                    # Download from cloud
                    remote_path = f"{activity_folder_path}{filename}"
                    local_path = str(local_folder / f"{camper_id}.jpg")
                    
                    success, error = self.download_file(remote_path, local_path)
                    if success:
                        success_count += 1
                        logger.info(f"    ✅ Downloaded camper {camper_id}")
                    else:
                        failed.append(f"{camper_id} ({filename})")
                        logger.warning(f"    ⚠️ Failed: {camper_id} - {error}")
                
                except Exception as e:
                    logger.error(f"    ❌ Error processing {avatar_file.get('name', '')}: {e}")
                    failed.append(str(avatar_file.get('name', '')))
            
            message = f"Optional activity {activity_id}: {success_count} campers downloaded"
            if failed:
                message += f" ({len(failed)} failed)"
            
            logger.info(f"  ✅ {message}")
            return success_count, message
        
        except Exception as e:
            error_msg = f"Sync error for activity {activity_id}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return 0, error_msg
