"""
Face Recognition API Service - Refactored
Clean, modular Flask application with proper separation of concerns
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import uuid

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
from utils import FileHandler, validate_image_file, validate_camper_id, validate_activity_schedule_id, get_now

# Setup logging
logger = setup_logging(log_level="INFO" if not settings.FLASK_DEBUG else "DEBUG")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize services
face_service = FaceRecognitionService()
supabase_service = SupabaseService()
file_handler = FileHandler()

logger.info("="*60)
logger.info("🚀 Face Recognition API Service Starting...")
logger.info(f"📦 Model: {settings.DEEPFACE_MODEL}")
logger.info(f"📏 Distance Metric: {settings.DEEPFACE_DISTANCE_METRIC}")
logger.info(f"🔍 Detector: {settings.DEEPFACE_DETECTOR}")
logger.info(f"🎯 Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
logger.info(f"⚡ FPS Limit: {settings.RECOGNITION_FPS_LIMIT}")
logger.info(f"💾 Cache Preload: {settings.CACHE_PRELOAD}")
logger.info("="*60)


# ============================================================================
# ENDPOINT 1: Health Check
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with cache statistics"""
    try:
        cache_stats = face_service.get_cache_statistics()
        
        response = HealthResponse(
            status="healthy",
            service="Face Recognition API",
            model=settings.DEEPFACE_MODEL,
            cache_stats=CacheStats(**cache_stats),
            timestamp=get_now().isoformat()
        )
        
        return jsonify(response.dict()), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": get_now().isoformat()
        }), 500


# ============================================================================
# ENDPOINT 2: Detect Faces
# ============================================================================

@app.route('/api/face/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image"""
    temp_file = None
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify(DetectFacesResponse(
                success=False,
                error="No image file provided",
                detected_faces=[],
                total_faces=0
            ).dict()), 400
        
        # Validate file
        file = request.files['image']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify(DetectFacesResponse(
                success=False,
                error=error_msg,
                detected_faces=[],
                total_faces=0
            ).dict()), 400
        
        # Save uploaded file
        temp_file, error = file_handler.save_uploaded_file(file)
        if error:
            return jsonify(DetectFacesResponse(
                success=False,
                error=error,
                detected_faces=[],
                total_faces=0
            ).dict()), 400
        
        # Extract faces
        faces = face_service.image_processor.extract_faces_from_frame(temp_file, min_confidence=0.5)
        
        if not faces:
            return jsonify(DetectFacesResponse(
                success=False,
                message="No faces detected in the image",
                detected_faces=[],
                total_faces=0
            ).dict()), 200
        
        # Format response
        detected_faces = [
            DetectedFace(
                face_id=str(uuid.uuid4()),
                confidence=face['confidence'],
                region=FaceRegion(**face['region'])
            )
            for face in faces
        ]
        
        response = DetectFacesResponse(
            success=True,
            message=f"Detected {len(detected_faces)} face(s)",
            detected_faces=detected_faces,
            total_faces=len(detected_faces)
        )
        
        return jsonify(response.dict()), 200
    
    except Exception as e:
        logger.error(f"Error in detect_faces: {e}")
        return jsonify(DetectFacesResponse(
            success=False,
            error=str(e),
            detected_faces=[],
            total_faces=0
        ).dict()), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


# ============================================================================
# ENDPOINT 3: Register Camper
# ============================================================================

@app.route('/api/attendance/camper/register', methods=['POST'])
def register_camper():
    """Register a camper's face for attendance"""
    temp_file = None
    try:
        # Validate image
        if 'image' not in request.files:
            return jsonify(RegisterCamperResponse(
                success=False,
                error="No image file provided",
                camper_id=""
            ).dict()), 400
        
        # Validate camper ID
        camper_id = request.form.get('camperId')
        is_valid, error_msg = validate_camper_id(camper_id)
        if not is_valid:
            return jsonify(RegisterCamperResponse(
                success=False,
                error=error_msg,
                camper_id=camper_id or ""
            ).dict()), 400
        
        # Validate file
        file = request.files['image']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify(RegisterCamperResponse(
                success=False,
                error=error_msg,
                camper_id=camper_id
            ).dict()), 400
        
        # Save uploaded file
        temp_file, error = file_handler.save_uploaded_file(file)
        if error:
            return jsonify(RegisterCamperResponse(
                success=False,
                error=error,
                camper_id=camper_id
            ).dict()), 400
        
        # Register camper
        result = face_service.register_camper(
            image_path=temp_file,
            camper_id=camper_id,
            preprocess=True
        )
        
        if not result['success']:
            return jsonify(RegisterCamperResponse(
                success=False,
                error=result.get('error', 'Registration failed'),
                camper_id=camper_id
            ).dict()), 400
        
        # Upload to Supabase if enabled
        face_url = None
        if supabase_service.is_enabled():
            try:
                from datetime import datetime
                filename = f"{camper_id}_{get_now().strftime('%Y%m%d_%H%M%S')}.jpg"
                face_url, _ = supabase_service.save_camper_face(
                    local_path=temp_file,
                    camper_id=camper_id,
                    filename=filename
                )
                logger.info(f"✅ Uploaded to Supabase: {face_url}")
            except Exception as e:
                logger.warning(f"⚠️ Supabase upload failed: {e}")
        
        # Build response
        response = RegisterCamperResponse(
            success=True,
            message="Camper registered successfully",
            camper_id=camper_id,
            embedding_shape=result.get('embedding_shape'),
            processing_time=result.get('processing_time')
        )
        
        return jsonify(response.dict()), 200
    
    except Exception as e:
        logger.error(f"Error in register_camper: {e}")
        return jsonify(RegisterCamperResponse(
            success=False,
            error=str(e),
            camper_id=request.form.get('camperId', '')
        ).dict()), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


# ============================================================================
# ENDPOINT 4: Check Attendance
# ============================================================================

@app.route('/api/attendance/check', methods=['POST'])
def check_attendance():
    """Check attendance by recognizing faces in uploaded image"""
    temp_file = None
    try:
        # Validate image
        if 'image' not in request.files:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message="No image file provided",
                error="No image file provided",
                session_id=str(uuid.uuid4()),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 400
        
        # Get parameters
        camp_id = request.form.get('campId')
        activity_type = request.form.get('activityType', 'core')  # 'core' or 'optional'
        activity_id = request.form.get('activityId')  # Required for optional activities
        
        # Validate parameters
        if not camp_id:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message="Camp ID is required",
                error="Camp ID is required",
                session_id=str(uuid.uuid4()),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 400
        
        if activity_type == 'optional' and not activity_id:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message="Activity ID is required for optional activities",
                error="Activity ID is required for optional activities",
                session_id=str(uuid.uuid4()),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 400
        
        # Validate file
        file = request.files['image']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message=error_msg,
                error=error_msg,
                session_id=str(uuid.uuid4()),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 400
        
        # Save uploaded file
        temp_file, error = file_handler.save_uploaded_file(file)
        if error:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message=error,
                error=error,
                session_id=str(uuid.uuid4()),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 400
        
        # Load appropriate embeddings based on activity type
        from config.settings import settings
        camp_db_path = settings.DATABASE_FOLDER / camp_id
        
        if activity_type == 'core':
            # Load core group embeddings
            core_path = camp_db_path / "camper_groups"
            if not core_path.exists():
                return jsonify(CheckAttendanceResponse(
                    success=False,
                    message=f"Core group database not found for camp {camp_id}. Please run preload first.",
                    error="Database not preloaded",
                    session_id=str(uuid.uuid4()),
                    recognized_campers=[],
                    total_faces_detected=0
                ).dict()), 400
            
            # Ensure embeddings are loaded
            face_service.embedding_cache._load_embeddings_from_folder(core_path)
        else:
            # Load optional activity embeddings
            activity_path = camp_db_path / "camper_activities" / activity_id
            if not activity_path.exists():
                return jsonify(CheckAttendanceResponse(
                    success=False,
                    message=f"Optional activity database not found for activity {activity_id}. Please run preload first.",
                    error="Database not preloaded",
                    session_id=str(uuid.uuid4()),
                    recognized_campers=[],
                    total_faces_detected=0
                ).dict()), 400
            
            # Ensure embeddings are loaded
            face_service.embedding_cache._load_embeddings_from_folder(activity_path)
        
        # Check attendance via face recognition
        result = face_service.check_attendance(
            image_path=temp_file,
            session_id=None,
            preprocess=True,
            save_results=True
        )
        
        if not result['success']:
            return jsonify(CheckAttendanceResponse(
                success=False,
                message=result.get('error', 'Attendance check failed'),
                error=result.get('error'),
                session_id=result.get('session_id', str(uuid.uuid4())),
                recognized_campers=[],
                total_faces_detected=0
            ).dict()), 500
        
        # Build response
        recognized_campers = [
            RecognizedCamper(
                camper_id=camper['camper_id'],
                distance=camper['distance'],
                face_region=FaceRegion(**camper['face_region'])
            )
            for camper in result.get('recognized_campers', [])
        ]
        
        response = CheckAttendanceResponse(
            success=True,
            message=result.get('message', 'Attendance check complete'),
            session_id=result['session_id'],
            recognized_campers=recognized_campers,
            total_faces_detected=result['total_faces_detected'],
            total_recognized=result.get('total_recognized', len(recognized_campers)),
            processing_time=result.get('processing_time'),
            timestamp=result.get('timestamp')
        )
        
        return jsonify(response.dict()), 200
    
    except Exception as e:
        logger.error(f"Error in check_attendance: {e}")
        return jsonify(CheckAttendanceResponse(
            success=False,
            message=str(e),
            error=str(e),
            session_id=str(uuid.uuid4()),
            recognized_campers=[],
            total_faces_detected=0
        ).dict()), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


# ============================================================================
# ENDPOINT 5: Get Session Results
# ============================================================================

@app.route('/api/session/<session_id>/results', methods=['GET'])
def get_session_results(session_id):
    """Get results for a specific session"""
    try:
        results = face_service.get_session_results(session_id)
        
        if results is None:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "results": results
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting session results: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# ENDPOINT 6: Cache Statistics
# ============================================================================

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = face_service.get_cache_statistics()
        return jsonify({
            "success": True,
            "cache_stats": stats
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# ENDPOINT 7: Preload Camp Day (for Hangfire - Start of Day)
# ============================================================================

@app.route('/api/attendance/preload-camp-day', methods=['POST'])
def preload_camp_day():
    """
    Preload all face data for a camp's activities for the entire day
    Downloads: CamperGroup (core) + all CamperActivities scheduled today
    Called by Hangfire at start of day
    """
    try:
        request_data = request.json if request.json else {}
        camp_id = request_data.get('campId')
        camper_group_ids = request_data.get('camperGroupIds', [])  # Core activity campers
        optional_activities = request_data.get('optionalActivities', [])  # List of {activityId, camperIds}
        
        if not camp_id:
            return jsonify({
                "success": False,
                "error": "campId is required"
            }), 400
        
        if not supabase_service.is_enabled():
            return jsonify({
                "success": False,
                "error": "Supabase is not enabled"
            }), 400
        
        logger.info(f"📥 Start-of-day preload for Camp {camp_id}")
        logger.info(f"   - CamperGroup: {len(camper_group_ids)} campers")
        logger.info(f"   - Optional Activities: {len(optional_activities)} activities")
        
        from config.settings import settings
        camp_db_path = settings.DATABASE_FOLDER / camp_id
        
        # Download CamperGroup faces (core activities - reused all day)
        core_path = camp_db_path / "camper_groups"
        core_count, core_message = supabase_service.sync_camp_core_group(
            camp_id=camp_id,
            camper_ids=camper_group_ids,
            local_folder=core_path
        )
        
        # Download optional activity faces
        optional_results = []
        for activity in optional_activities:
            activity_id = activity.get('activityId')
            camper_ids = activity.get('camperIds', [])
            
            if activity_id and camper_ids:
                activity_path = camp_db_path / "camper_activities" / activity_id
                count, message = supabase_service.sync_optional_activity(
                    camp_id=camp_id,
                    activity_id=activity_id,
                    camper_ids=camper_ids,
                    local_folder=activity_path
                )
                optional_results.append({
                    "activity_id": activity_id,
                    "downloaded": count,
                    "requested": len(camper_ids)
                })
        
        # Load all embeddings into cache
        face_service.embedding_cache.clear_cache()  # Clear old cache
        face_service.embedding_cache._load_embeddings_from_folder(core_path)
        for activity in optional_activities:
            activity_id = activity.get('activityId')
            if activity_id:
                activity_path = camp_db_path / "camper_activities" / activity_id
                face_service.embedding_cache._load_embeddings_from_folder(activity_path)
        
        total_campers = core_count + sum(r['downloaded'] for r in optional_results)
        
        return jsonify({
            "success": True,
            "message": f"Preloaded {total_campers} campers for camp {camp_id}",
            "camp_id": camp_id,
            "core_campers": {
                "downloaded": core_count,
                "requested": len(camper_group_ids)
            },
            "optional_activities": optional_results,
            "total_campers_loaded": total_campers
        }), 200
    
    except Exception as e:
        logger.error(f"Error preloading camp day: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# ENDPOINT 8: Cleanup Camp Day (for Hangfire - End of Day)
# ============================================================================

@app.route('/api/attendance/cleanup-camp-day', methods=['POST'])
def cleanup_camp_day():
    """
    Delete all face data for a camp at end of day
    Called by Hangfire at end of day
    """
    try:
        request_data = request.json if request.json else {}
        camp_id = request_data.get('campId')
        
        if not camp_id:
            return jsonify({
                "success": False,
                "error": "campId is required"
            }), 400
        
        # Delete camp database folder
        from config.settings import settings
        import shutil
        
        camp_db_path = settings.DATABASE_FOLDER / camp_id
        
        if camp_db_path.exists():
            shutil.rmtree(camp_db_path)
            logger.info(f"🗑️  End-of-day cleanup: Deleted camp database {camp_db_path}")
            
            # Clear embeddings cache
            face_service.embedding_cache.clear_cache()
            
            return jsonify({
                "success": True,
                "message": f"End-of-day cleanup complete for camp {camp_id}",
                "deleted_path": str(camp_db_path)
            }), 200
        else:
            return jsonify({
                "success": True,
                "message": f"No database found for camp {camp_id}",
                "deleted_path": None
            }), 200
    
    except Exception as e:
        logger.error(f"Error cleaning up camp day: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# ENDPOINT 9: Clear Cache
# ============================================================================

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear embedding cache"""
    try:
        camper_id = request.json.get('camperId') if request.json else None
        face_service.clear_cache(camper_id)
        
        message = f"Cache cleared for camper {camper_id}" if camper_id else "Cache cleared"
        
        return jsonify({
            "success": True,
            "message": message
        }), 200
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


# ============================================================================
# Startup Initialization
# ============================================================================

def initialize_database_on_startup():
    """
    Initialize database on server startup
    Note: Face databases are now loaded per activity schedule via /api/attendance/preload
    This function just checks if any data exists for development purposes
    """
    try:
        logger.info("="*60)
        logger.info("🔄 Checking local face database...")
        
        # Check face_database folder
        face_db_path = settings.DATABASE_FOLDER
        
        # Count existing images
        has_images = any(
            f.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            for f in face_db_path.rglob('*')
            if f.is_file()
        )
        
        if has_images:
            logger.info(f"✅ Local database found with images")
            logger.info(f"ℹ️  Face databases are loaded per activity schedule via /api/attendance/preload")
        else:
            logger.info(f"ℹ️  Local database empty - face data will be loaded via /api/attendance/preload")
            logger.info(f"ℹ️  This is expected for production deployment")
        
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"❌ Error checking database: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Initialize database on startup
    initialize_database_on_startup()
    
    logger.info("="*60)
    logger.info("✅ All services initialized successfully")
    logger.info(f"🌐 Starting Flask server on {settings.FLASK_HOST}:{settings.FLASK_PORT}")
    logger.info("="*60)
    
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG
    )
