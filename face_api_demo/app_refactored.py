"""
Face Recognition API Service - Production Ready
Integrated with ASP.NET Camp Management System
Folder structure: attendance-sessions/camp_{campId}/camper_group_{groupId}/{camperId}/photo.jpg
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from config import settings, setup_logging
from services import ImageProcessor, EmbeddingCache, FaceRecognitionService, SupabaseService
from models.schemas import (
    RegisterCamperResponse,
    CheckAttendanceResponse,
    DetectFacesResponse,
    HealthResponse,
    CacheStats,
    DetectedFace,
    RecognizedCamper,
    FaceRegion
)
from utils import FileHandler, validate_image_file, validate_camper_id, get_now

# Setup logging
logger = setup_logging(log_level="INFO" if not settings.FLASK_DEBUG else "DEBUG")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
face_service = FaceRecognitionService()
supabase_service = SupabaseService()
file_handler = FileHandler()

# Global state for loaded camps
loaded_camps: Dict[int, Dict] = {}  # campId -> {face_count, load_time, groups: []}

logger.info("="*60)
logger.info("🚀 Face Recognition API Service Starting...")
logger.info(f"📦 Model: {settings.DEEPFACE_MODEL}")
logger.info(f"🎯 Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
logger.info(f"📁 Face DB Root: {settings.DATABASE_FOLDER}")
logger.info("="*60)


# ============================================================================
# ENDPOINT 1: Health Check
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system statistics"""
    try:
        cache_stats = face_service.get_cache_statistics()
        
        response = HealthResponse(
            status="healthy",
            service="Face Recognition API",
            model=settings.DEEPFACE_MODEL,
            cache_stats=CacheStats(
                total_campers=cache_stats.get('total_campers', 0),
                memory_usage_mb=cache_stats.get('memory_usage_mb', 0),
                last_updated=cache_stats.get('last_updated')
            ),
            timestamp=get_now().isoformat()
        )
        
        # Add loaded camps info
        response_dict = response.dict()
        response_dict['loaded_camps'] = len(loaded_camps)
        response_dict['loaded_camp_ids'] = list(loaded_camps.keys())
        
        return jsonify(response_dict), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": get_now().isoformat()
        }), 500


# ============================================================================
# ENDPOINT 2: Load Camp Face Database  
# ============================================================================

@app.route('/api/face-db/load/<int:camp_id>', methods=['POST'])
def load_camp_face_db(camp_id: int):
    """
    Load face database for a specific camp from Supabase attendance-sessions bucket
    Downloads faces from: attendance-sessions/camp_{campId}/camper_group_{groupId}/{camperId}/
    Loads embeddings into memory for fast recognition
    
    Expected Supabase structure:
    - attendance-sessions/camp_{campId}/camper_group_{groupId}/{camperId}/photo.jpg
    - attendance-sessions/camp_{campId}/camperactivity_{activityId}/{camperId}/photo.jpg
    """
    try:
        request_data = request.json if request.json else {}
        force_reload = request_data.get('force_reload', False)
        
        # Check if already loaded
        if camp_id in loaded_camps and not force_reload:
            return jsonify({
                "success": True,
                "message": f"Camp {camp_id} face database already loaded",
                "camp_id": camp_id,
                "face_count": loaded_camps[camp_id]['face_count'],
                "loaded_at": loaded_camps[camp_id]['load_time']
            }), 200
        
        if not supabase_service.is_enabled():
            return jsonify({
                "success": False,
                "message": "Supabase is not enabled. Cannot download face database.",
                "camp_id": camp_id,
                "face_count": 0
            }), 400
        
        logger.info(f"📥 Loading face database for Camp {camp_id}")
        
        # Define local storage path
        local_camp_path = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        local_camp_path.mkdir(parents=True, exist_ok=True)
        
        # Download all faces from Supabase attendance-sessions bucket
        # Path pattern: attendance-sessions/camp_{campId}/camper_group_{groupId}/{camperId}/
        supabase_camp_prefix = f"camp_{camp_id}"
        
        total_downloaded = 0
        groups_loaded = []
        
        try:
            # First, verify the camp folder exists by listing root
            root_folders = supabase_service.client.storage.from_(settings.SUPABASE_BUCKET).list()
            logger.info(f"🗂️  Root level has {len(root_folders)} items:")
            for folder in root_folders:
                logger.info(f"   - Name: '{folder.get('name')}', ID: {folder.get('id')}")
            camp_folder_exists = any(f.get('name') == supabase_camp_prefix for f in root_folders)
            logger.info(f"📁 Looking for camp folder '{supabase_camp_prefix}', exists: {camp_folder_exists}")
            
            # List all items inside the camp folder
            files = supabase_service.client.storage.from_(settings.SUPABASE_BUCKET).list(supabase_camp_prefix)
            logger.info(f"📋 Supabase list returned {len(files)} items inside '{supabase_camp_prefix}'")
            for f in files:
                logger.debug(f"  - {f.get('name', 'unnamed')} (id: {f.get('id', 'no-id')})")
            
            # Process camper_group folders
            for item in files:
                if item['name'].startswith('camper_group_'):
                    group_id = item['name'].replace('camper_group_', '')
                    group_prefix = f"{supabase_camp_prefix}/{item['name']}"
                    logger.info(f"Processing group: {item['name']} (prefix: {group_prefix})")
                    
                    # List items in this group (now expecting flat files like avatar_16_xxx.jpg)
                    try:
                        group_items = supabase_service.client.storage.from_(settings.SUPABASE_BUCKET).list(group_prefix)
                        logger.debug(f"Found {len(group_items)} items in {group_prefix}")
                        
                        for item_in_group in group_items:
                            filename = item_in_group['name']
                            
                            # Check if it's an image file (not a folder marker)
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                # Extract camper_id from filename: avatar_16_xxx.jpg -> 16
                                if filename.startswith('avatar_'):
                                    parts = filename.split('_')
                                    if len(parts) >= 2:
                                        camper_id = parts[1]
                                    else:
                                        logger.warning(f"Unexpected filename format: {filename}")
                                        continue
                                else:
                                    logger.warning(f"Filename doesn't start with 'avatar_': {filename}")
                                    continue
                                
                                # Download the photo (flat structure in camp folder)
                                remote_path = f"{group_prefix}/{filename}"
                                local_file = local_camp_path / filename
                                local_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                logger.debug(f"Downloading: {remote_path} -> {local_file}")
                                success, error = supabase_service.download_file(remote_path, str(local_file))
                                if success:
                                    total_downloaded += 1
                                    logger.debug(f"✅ Downloaded camper {camper_id} photo")
                                else:
                                    logger.error(f"❌ Failed to download {remote_path}: {error}")
                                        
                    except Exception as e:
                        logger.warning(f"Error processing group {group_id}: {e}")
                
                # Process camperactivity folders
                elif item['name'].startswith('camperactivity_'):
                    activity_id = item['name'].replace('camperactivity_', '')
                    activity_prefix = f"{supabase_camp_prefix}{item['name']}/"
                    
                    try:
                        campers = supabase_service.client.storage.from_(settings.SUPABASE_BUCKET).list(activity_prefix)
                        
                        for camper_folder in campers:
                            if camper_folder['id']:
                                camper_id = camper_folder['name']
                                camper_prefix = f"{activity_prefix}{camper_id}/"
                                
                                photos = supabase_service.client.storage.from_(settings.SUPABASE_BUCKET).list(camper_prefix)
                                
                                for photo in photos:
                                    if photo['name'].lower().endswith(('.jpg', '.jpeg', '.png')):
                                        remote_path = f"{camper_prefix}{photo['name']}"
                                        local_file = local_camp_path / f"activity_{activity_id}" / camper_id / photo['name']
                                        local_file.parent.mkdir(parents=True, exist_ok=True)
                                        
                                        success, error = supabase_service.download_file(remote_path, str(local_file))
                                        if success:
                                            total_downloaded += 1
                    
                    except Exception as e:
                        logger.warning(f"Error processing activity {activity_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error downloading from Supabase: {e}")
            return jsonify({
                "success": False,
                "message": f"Failed to download face database: {str(e)}",
                "camp_id": camp_id,
                "face_count": 0
            }), 500
        
        # Load embeddings into cache
        face_service.embedding_cache._load_embeddings_from_folder(local_camp_path)
        
        # Update loaded camps registry
        loaded_camps[camp_id] = {
            "face_count": total_downloaded,
            "load_time": get_now().isoformat(),
            "groups": groups_loaded
        }
        
        logger.info(f"✅ Loaded {total_downloaded} faces for Camp {camp_id}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully loaded {total_downloaded} faces for camp {camp_id}",
            "camp_id": camp_id,
            "face_count": total_downloaded,
            "timestamp": get_now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error loading camp face database: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": str(e),
            "camp_id": camp_id,
            "face_count": 0
        }), 500


# ============================================================================
# ENDPOINT 3: Unload Camp Face Database
# ============================================================================

@app.route('/api/face-db/unload/<int:camp_id>', methods=['DELETE'])
def unload_camp_face_db(camp_id: int):
    """
    Unload face database for a specific camp from memory
    Deletes local files and clears embeddings cache
    """
    try:
        if camp_id not in loaded_camps:
            return jsonify({
                "success": True,
                "message": f"Camp {camp_id} face database is not loaded",
                "camp_id": camp_id
            }), 200
        
        logger.info(f"🗑️  Unloading face database for Camp {camp_id}")
        
        # Delete local files
        local_camp_path = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        if local_camp_path.exists():
            shutil.rmtree(local_camp_path)
            logger.info(f"   Deleted local files: {local_camp_path}")
        
        # Remove from loaded camps
        face_count = loaded_camps[camp_id]['face_count']
        del loaded_camps[camp_id]
        
        # Clear cache (partial clear for this camp)
        # Note: Current implementation clears entire cache - can be optimized
        face_service.embedding_cache.clear_cache()
        
        logger.info(f"✅ Unloaded Camp {camp_id}")
        
        return jsonify({
            "success": True,
            "message": f"Successfully unloaded camp {camp_id}",
            "camp_id": camp_id,
            "face_count": face_count,
            "timestamp": get_now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error unloading camp face database: {e}")
        return jsonify({
            "success": False,
            "message": str(e),
            "camp_id": camp_id
        }), 500


# ============================================================================
# ENDPOINT 4: Get Face Database Statistics
# ============================================================================

@app.route('/api/face-db/stats', methods=['GET'])
def get_face_db_stats():
    """Get statistics about loaded camps"""
    try:
        stats = {
            "success": True,
            "loaded_camps": {
                camp_id: {
                    "face_count": info['face_count'],
                    "loaded_at": info['load_time']
                }
                for camp_id, info in loaded_camps.items()
            },
            "total_camps": len(loaded_camps),
            "total_faces": sum(info['face_count'] for info in loaded_camps.values()),
            "cache_stats": face_service.get_cache_statistics()
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        logger.error(f"Error getting face DB stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# ENDPOINT 5: Recognize Faces for Activity Schedule
# ============================================================================

@app.route('/api/recognition/recognize/<int:activity_schedule_id>', methods=['POST'])
def recognize_faces(activity_schedule_id: int):
    """
    Recognize faces in uploaded photo for a specific activity schedule
    Returns list of recognized campers with confidence scores
    """
    temp_file = None
    try:
        # Validate image
        if 'photo' not in request.files:
            return jsonify({
                "success": False,
                "message": "No photo file provided",
                "activity_schedule_id": activity_schedule_id,
                "recognized_campers": [],
                "total_faces_detected": 0,
                "matched_faces": 0
            }), 400
        
        # Get optional parameters
        confidence_threshold = request.form.get('confidence_threshold')
        if confidence_threshold:
            try:
                confidence_threshold = float(confidence_threshold)
            except:
                confidence_threshold = settings.CONFIDENCE_THRESHOLD
        else:
            confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Validate file
        file = request.files['photo']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({
                "success": False,
                "message": error_msg,
                "activity_schedule_id": activity_schedule_id,
                "recognized_campers": [],
                "total_faces_detected": 0,
                "matched_faces": 0
            }), 400
        
        # Save uploaded file
        temp_file, error = file_handler.save_uploaded_file(file)
        if error:
            return jsonify({
                "success": False,
                "message": error,
                "activity_schedule_id": activity_schedule_id,
                "recognized_campers": [],
                "total_faces_detected": 0,
                "matched_faces": 0
            }), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Perform recognition
        start_time = datetime.now()
        result = face_service.check_attendance(
            image_path=temp_file,
            session_id=session_id,
            preprocess=True,
            save_results=True
        )
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if not result['success']:
            return jsonify({
                "success": False,
                "message": result.get('error', 'Recognition failed'),
                "activity_schedule_id": activity_schedule_id,
                "recognized_campers": [],
                "total_faces_detected": 0,
                "matched_faces": 0
            }), 500
        
        # Format response
        recognized_campers = []
        for camper in result.get('recognized_campers', []):
            recognized_campers.append({
                "camper_id": int(camper['camper_id']) if camper['camper_id'].isdigit() else 0,
                "camper_name": "",  # Will be filled by .NET service
                "confidence": 1.0 - camper['distance'],  # Convert distance to confidence
                "camper_group_id": 0,  # Will be filled by .NET service
                "bounding_box": [
                    camper['face_region']['x'],
                    camper['face_region']['y'],
                    camper['face_region']['width'],
                    camper['face_region']['height']
                ]
            })
        
        return jsonify({
            "success": True,
            "message": f"Recognized {len(recognized_campers)} camper(s)",
            "activity_schedule_id": activity_schedule_id,
            "session_id": session_id,
            "recognized_campers": recognized_campers,
            "total_faces_detected": result['total_faces_detected'],
            "matched_faces": len(recognized_campers),
            "processing_time_ms": int(processing_time),
            "timestamp": get_now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Error in recognize_faces: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": str(e),
            "activity_schedule_id": activity_schedule_id,
            "recognized_campers": [],
            "total_faces_detected": 0,
            "matched_faces": 0
        }), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


# ============================================================================
# Legacy Endpoints (Keep for backward compatibility)
# ============================================================================

@app.route('/api/face/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image"""
    temp_file = None
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": error_msg}), 400
        
        temp_file, error = file_handler.save_uploaded_file(file)
        if error:
            return jsonify({"success": False, "error": error}), 400
        
        faces = face_service.image_processor.extract_faces_from_frame(temp_file, min_confidence=0.5)
        
        detected_faces = [
            {
                "face_id": str(uuid.uuid4()),
                "confidence": face['confidence'],
                "region": face['region']
            }
            for face in faces
        ]
        
        return jsonify({
            "success": True,
            "message": f"Detected {len(detected_faces)} face(s)",
            "detected_faces": detected_faces,
            "total_faces": len(detected_faces)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in detect_faces: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear embedding cache"""
    try:
        camper_id = request.json.get('camperId') if request.json else None
        face_service.clear_cache(camper_id)
        
        message = f"Cache cleared for camper {camper_id}" if camper_id else "Cache cleared"
        
        return jsonify({"success": True, "message": message}), 200
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("✅ All services initialized successfully")
    logger.info(f"🌐 Starting Flask server on {settings.FLASK_HOST}:{settings.FLASK_PORT}")
    logger.info("="*60)
    
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG
    )
