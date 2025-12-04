"""
Face Recognition API Service - Production Ready
Optimized for latency with Spectree Swagger integration
"""

from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from datetime import datetime
import logging
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import os

from config import settings, setup_logging
from services import SupabaseService
from middleware import init_jwt_middleware, require_auth

# Import optimized services
if settings.USE_OPTIMIZED_PROCESSOR:
    from services.image_processor_optimized import OptimizedImageProcessor as ImageProcessor
    logger = logging.getLogger(__name__)
    logger.info("✅ Using OptimizedImageProcessor")
else:
    from services import ImageProcessor

if settings.USE_OPTIMIZED_CACHE:
    from services.embedding_cache_optimized import OptimizedEmbeddingCache as EmbeddingCache
    logger = logging.getLogger(__name__)
    logger.info("✅ Using OptimizedEmbeddingCache with FAISS")
else:
    from services import EmbeddingCache

from services import FaceRecognitionService
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

# Configure CORS with explicit settings
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False)

# Note: Spectree removed due to Pydantic v2 compatibility issues
# Using manual OpenAPI documentation instead

# Initialize JWT middleware
app.config['JWT_SECRET_KEY'] = settings.JWT_SECRET_KEY
app.config['JWT_ALGORITHM'] = settings.JWT_ALGORITHM
app.config['JWT_ISSUER'] = settings.JWT_ISSUER
app.config['JWT_AUDIENCE'] = settings.JWT_AUDIENCE
app.config['JWT_ISSUER_WHITELIST'] = [settings.JWT_ISSUER]
app.config['JWT_CLOCK_SKEW_SECONDS'] = 60
app.config['JWT_ENABLE_CLOUDFLARE_HEADERS'] = True

jwt_middleware = init_jwt_middleware(app)

# Initialize services
face_service = FaceRecognitionService()
supabase_service = SupabaseService()
file_handler = FileHandler()

# Global state for loaded camps
loaded_camps: Dict[int, Dict] = {}

logger.info("="*60)
logger.info("🚀 Face Recognition API Service Starting...")
logger.info(f"📦 Model: {settings.DEEPFACE_MODEL}")
logger.info(f"🎯 Confidence Threshold (from .env): {settings.CONFIDENCE_THRESHOLD}")
logger.info(f"📁 Face DB Root: {settings.DATABASE_FOLDER}")
logger.info(f"🔒 JWT Auth: Enabled (Issuer: {app.config['JWT_ISSUER']})")
logger.info(f"📚 Swagger UI: /apidoc")
logger.info("="*60)


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system statistics"""
    try:
        # Get cache statistics from embedding cache
        cached_campers = list(face_service.embedding_cache.cache.keys())
        
        response = HealthResponse(
            status="healthy",
            service="Face Recognition API",
            model=settings.DEEPFACE_MODEL,
            cache_stats=CacheStats(
                total_cached=len(cached_campers),
                campers=cached_campers,
                model=settings.DEEPFACE_MODEL,
                distance_metric=settings.DEEPFACE_DISTANCE_METRIC,
                fps_limit=settings.RECOGNITION_FPS_LIMIT,
                threshold=settings.CONFIDENCE_THRESHOLD
            ),
            timestamp=get_now().isoformat()
        )
        
        response_dict = response.model_dump()
        response_dict['loaded_camps'] = len(loaded_camps)
        
        return jsonify(response_dict), 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": get_now().isoformat()
        }), 500


@app.route('/apidoc', methods=['GET'])
@app.route('/docs', methods=['GET'])
def api_documentation():
    """Manual API documentation endpoint (Swagger UI)"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Face Recognition API - Documentation</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui.css">
        <style>
            body { margin: 0; padding: 0; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui-bundle.js"></script>
        <script src="https://unpkg.com/swagger-ui-dist@5.10.5/swagger-ui-standalone-preset.js"></script>
        <script>
            window.onload = function() {
                const ui = SwaggerUIBundle({
                    url: "/openapi.json",
                    dom_id: '#swagger-ui',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIStandalonePreset
                    ],
                    layout: "StandaloneLayout"
                });
                window.ui = ui;
            };
        </script>
    </body>
    </html>
    """
    return html, 200, {'Content-Type': 'text/html'}


@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    """OpenAPI 3.0 specification"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Face Recognition API",
            "version": "2.0.0",
            "description": "Production-ready face recognition API with JWT authentication and latency-optimized processing"
        },
        "servers": [
            {"url": f"http://localhost:{settings.FLASK_PORT}", "description": "Local server"},
            {"url": f"http://127.0.0.1:{settings.FLASK_PORT}", "description": "Local server (IP)"}
        ],
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token from SummerCampBackend"
                }
            }
        },
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "tags": ["Health"],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {"application/json": {"example": {"status": "healthy", "service": "Face Recognition API"}}}
                        }
                    }
                }
            },
            "/api/face-db/load/{camp_id}": {
                "post": {
                    "summary": "Load camp face database",
                    "tags": ["Face Database"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [{"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "Camp loaded successfully"}}
                }
            },
            "/api/face-db/unload/{camp_id}": {
                "delete": {
                    "summary": "Unload camp face database",
                    "tags": ["Face Database"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [{"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "Camp unloaded successfully"}}
                }
            },
            "/api/face-db/stats": {
                "get": {
                    "summary": "Get face database statistics",
                    "tags": ["Face Database"],
                    "security": [{"bearerAuth": []}],
                    "responses": {"200": {"description": "Statistics retrieved successfully"}}
                }
            },
            "/api/recognition/recognize-group/{camp_id}/{group_id}": {
                "post": {
                    "summary": "Recognize faces for group activity",
                    "tags": ["Recognition"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "group_id", "in": "path", "required": True, "schema": {"type": "integer"}}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"photo": {"type": "string", "format": "binary"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Faces recognized successfully"}}
                }
            },
            "/api/recognition/recognize-activity/{camp_id}/{activity_schedule_id}": {
                "post": {
                    "summary": "Recognize faces for optional activity",
                    "tags": ["Recognition"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "activity_schedule_id", "in": "path", "required": True, "schema": {"type": "integer"}}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"photo": {"type": "string", "format": "binary"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Faces recognized successfully"}}
                }
            },
            "/api/face/detect": {
                "post": {
                    "summary": "Detect faces in image",
                    "tags": ["Face Detection"],
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"image": {"type": "string", "format": "binary"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Faces detected successfully"}}
                }
            },
            "/api/cache/clear": {
                "post": {
                    "summary": "Clear embedding cache",
                    "tags": ["Cache"],
                    "security": [{"bearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"camperId": {"type": "string"}}
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "Cache cleared successfully"}}
                }
            }
        }
    }
    return jsonify(spec), 200


# ============================================================================
# PROTECTED ENDPOINTS (JWT Required)
# ============================================================================

@app.route('/api/face-db/load/<int:camp_id>', methods=['POST'])
@require_auth
def load_camp_face_database(camp_id):
    """
    Load face database for a specific camp from Supabase
    Downloads face images and generates embeddings for all camper groups
    """
    try:
        logger.info(f"🔄 Loading face database for camp {camp_id} (requested by {g.user.get('sub')})")
        
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        
        if camp_id in loaded_camps:
            logger.info(f"✅ Camp {camp_id} already loaded with {loaded_camps[camp_id]['face_count']} faces")
            return jsonify({
                "success": True,
                "message": f"Camp {camp_id} already loaded",
                "camp_id": camp_id,
                "face_count": loaded_camps[camp_id]['face_count'],
                "groups": loaded_camps[camp_id]['groups']
            }), 200
        
        # Download avatar images from Supabase
        logger.info(f"📥 Downloading face images from Supabase for camp {camp_id}...")
        downloaded_files = supabase_service.download_camp_faces(camp_id, str(camp_folder))
        
        if not downloaded_files:
            return jsonify({
                "success": False,
                "message": f"No face images found for camp {camp_id}",
                "camp_id": camp_id
            }), 404
        
        logger.info(f"✅ Downloaded {len(downloaded_files)} face images")
        
        # Organize by groups and generate embeddings
        groups_data = {}
        total_faces = 0
        
        for file_path in downloaded_files:
            path = Path(file_path)
            parts = path.parts
            
            group_folder_name = None
            for part in parts:
                if part.startswith('camper_group_'):
                    group_folder_name = part
                    break
            
            if group_folder_name:
                group_id = int(group_folder_name.replace('camper_group_', ''))
                if group_id not in groups_data:
                    groups_data[group_id] = []
                groups_data[group_id].append(file_path)
        
        # Load embeddings for each group
        for group_id, face_files in groups_data.items():
            group_folder = camp_folder / f"camper_group_{group_id}"
            logger.info(f"⚡ Generating embeddings for group {group_id} ({len(face_files)} faces)...")
            face_service.embedding_cache._load_embeddings_from_folder(group_folder)
            total_faces += len(face_files)
        
        loaded_camps[camp_id] = {
            'face_count': total_faces,
            'load_time': get_now().isoformat(),
            'groups': list(groups_data.keys())
        }
        
        logger.info(f"✅ Camp {camp_id} loaded successfully: {total_faces} faces across {len(groups_data)} groups")
        
        return jsonify({
            "success": True,
            "message": f"Loaded {total_faces} faces for camp {camp_id}",
            "camp_id": camp_id,
            "face_count": total_faces,
            "groups": list(groups_data.keys())
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error loading camp {camp_id}: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "camp_id": camp_id
        }), 500


@app.route('/api/face-db/unload/<int:camp_id>', methods=['DELETE'])
@require_auth
def unload_camp_face_database(camp_id):
    """
    Unload face database for a specific camp
    Removes from cache and deletes local files
    """
    try:
        logger.info(f"🗑️ Unloading camp {camp_id} (requested by {g.user.get('sub')})")
        
        if camp_id not in loaded_camps:
            return jsonify({
                "success": False,
                "message": f"Camp {camp_id} is not loaded",
                "camp_id": camp_id
            }), 404
        
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        
        # Clear cache for this camp
        if camp_folder.exists():
            for group_folder in camp_folder.glob("camper_group_*"):
                for face_file in group_folder.glob("avatar_*.jpg"):
                    camper_id_str = face_file.stem.split('_')[1]
                    face_service.embedding_cache.remove_camper(camper_id_str)
        
        # Delete local files
        if camp_folder.exists():
            shutil.rmtree(camp_folder)
            logger.info(f"🗑️ Deleted folder: {camp_folder}")
        
        del loaded_camps[camp_id]
        
        logger.info(f"✅ Camp {camp_id} unloaded successfully")
        
        return jsonify({
            "success": True,
            "message": f"Camp {camp_id} unloaded successfully",
            "camp_id": camp_id
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error unloading camp {camp_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "camp_id": camp_id
        }), 500


@app.route('/api/face-db/stats', methods=['GET'])
@require_auth
def get_camp_stats():
    """Get statistics about loaded camps"""
    try:
        cached_campers = list(face_service.embedding_cache.cache.keys())
        
        return jsonify({
            "loaded_camps": loaded_camps,
            "total_camps": len(loaded_camps),
            "cache_stats": {
                "total_cached": len(cached_campers),
                "campers": cached_campers,
                "model": settings.DEEPFACE_MODEL,
                "distance_metric": settings.DEEPFACE_DISTANCE_METRIC,
                "fps_limit": settings.RECOGNITION_FPS_LIMIT,
                "threshold": settings.CONFIDENCE_THRESHOLD
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/recognition/recognize-group/<int:camp_id>/<int:group_id>', methods=['POST'])
@require_auth
def recognize_faces_for_group(camp_id, group_id):
    """
    Recognize faces for a specific camper group (core activities)
    Optimized: loads only faces from that group's folder
    """
    temp_file = None
    try:
        logger.info(f"🎯 Group recognition: camp={camp_id}, group={group_id} (by {g.user.get('sub')})")
        
        if camp_id not in loaded_camps:
            return jsonify({
                "success": False,
                "message": f"Camp {camp_id} not loaded. Please load camp first.",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        
        if group_id not in loaded_camps[camp_id].get('groups', []):
            return jsonify({
                "success": False,
                "message": f"Group {group_id} not found in camp {camp_id}",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        
        if 'photo' not in request.files:
            return jsonify({
                "success": False,
                "message": "No photo file provided",
                "campId": camp_id,
                "groupId": group_id
            }), 400
        
        file = request.files['photo']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({
                "success": False,
                "message": error_msg,
                "campId": camp_id,
                "groupId": group_id
            }), 400
        
        temp_file, error = file_handler.save_uploaded_file(file, "recognition")
        if error:
            return jsonify({
                "success": False,
                "message": f"Error saving uploaded file: {error}",
                "campId": camp_id,
                "groupId": group_id
            }), 500
        
        # Load embeddings for this specific group only
        group_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}" / f"camper_group_{group_id}"
        
        if not group_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Group folder not found. Please preload camp {camp_id} first.",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        
        # Clear cache and load only this group's embeddings for accurate group-specific recognition
        face_service.embedding_cache.clear_cache()
        logger.info(f"⚡ Loading embeddings for group {group_id} only...")
        face_service.embedding_cache._load_embeddings_from_folder(group_folder)
        
        session_id = str(uuid.uuid4())
        
        # Recognize faces with optimized threshold from .env
        result = face_service.check_attendance(
            str(temp_file),
            session_id=session_id,
            save_results=False
        )
        
        recognized_campers = [
            {
                "camperId": int(r['camper_id']),
                "confidence": round(1.0 - r['distance'], 4),  # Convert distance to confidence
                "boundingBox": [
                    r['face_region']['x'],
                    r['face_region']['y'],
                    r['face_region']['width'],
                    r['face_region']['height']
                ] if 'face_region' in r and r['face_region'] else None
            }
            for r in result.get('recognized_campers', [])
        ]
        
        logger.info(f"✅ Recognized {len(recognized_campers)} camper(s) from group {group_id}")
        
        return jsonify({
            "success": True,
            "message": f"Recognized {len(recognized_campers)} camper(s) from group {group_id}",
            "campId": camp_id,
            "groupId": group_id,
            "sessionId": session_id,
            "recognizedCampers": recognized_campers,
            "totalFacesDetected": result.get('total_faces_detected', 0),
            "matchedFaces": len(recognized_campers),
            "processingTimeMs": int(result.get('processing_time', 0) * 1000),
            "timestamp": get_now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error in group recognition: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "campId": camp_id,
            "groupId": group_id
        }), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


@app.route('/api/recognition/recognize-activity/<int:camp_id>/<int:activity_schedule_id>', methods=['POST'])
@require_auth
def recognize_faces_for_activity(camp_id, activity_schedule_id):
    """
    Recognize faces for optional activity
    Optimized: loads only faces from that activity's folder
    """
    temp_file = None
    try:
        logger.info(f"🎯 Activity recognition: camp={camp_id}, activity={activity_schedule_id} (by {g.user.get('sub')})")
        
        if camp_id not in loaded_camps:
            return jsonify({
                "success": False,
                "message": f"Camp {camp_id} not loaded. Please load camp first.",
                "campId": camp_id,
                "activityScheduleId": activity_schedule_id
            }), 404
        
        if 'photo' not in request.files:
            return jsonify({
                "success": False,
                "message": "No photo file provided",
                "campId": camp_id,
                "activityScheduleId": activity_schedule_id
            }), 400
        
        file = request.files['photo']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify({
                "success": False,
                "message": error_msg,
                "campId": camp_id,
                "activityScheduleId": activity_schedule_id
            }), 400
        
        temp_file, error = file_handler.save_uploaded_file(file, "recognition")
        if error:
            return jsonify({
                "success": False,
                "message": f"Error saving uploaded file: {error}",
                "campId": camp_id,
                "activityScheduleId": activity_schedule_id
            }), 500
        
        activity_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}" / f"activity_{activity_schedule_id}"
        
        if not activity_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Activity folder not found. Please preload camp {camp_id} first.",
                "campId": camp_id,
                "activityScheduleId": activity_schedule_id
            }), 404
        
        # Clear cache and load only this activity's embeddings for accurate activity-specific recognition
        face_service.embedding_cache.clear_cache()
        logger.info(f"⚡ Loading embeddings for activity {activity_schedule_id} only...")
        face_service.embedding_cache._load_embeddings_from_folder(activity_folder)
        
        session_id = str(uuid.uuid4())
        
        result = face_service.check_attendance(
            str(temp_file),
            session_id=session_id,
            save_results=False
        )
        
        recognized_campers = [
            {
                "camperId": int(r['camper_id']),
                "confidence": round(1.0 - r['distance'], 4),
                "boundingBox": [
                    r['face_region']['x'],
                    r['face_region']['y'],
                    r['face_region']['width'],
                    r['face_region']['height']
                ] if 'face_region' in r and r['face_region'] else None
            }
            for r in result.get('recognized_campers', [])
        ]
        
        logger.info(f"✅ Recognized {len(recognized_campers)} camper(s) for activity {activity_schedule_id}")
        
        return jsonify({
            "success": True,
            "message": f"Recognized {len(recognized_campers)} camper(s)",
            "campId": camp_id,
            "activityScheduleId": activity_schedule_id,
            "sessionId": session_id,
            "recognizedCampers": recognized_campers,
            "totalFacesDetected": result.get('total_faces', 0),
            "matchedFaces": len(recognized_campers),
            "processingTimeMs": int(result.get('processing_time', 0) * 1000),
            "timestamp": get_now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error in activity recognition: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "campId": camp_id,
            "activityScheduleId": activity_schedule_id
        }), 500
    
    finally:
        file_handler.cleanup_file(temp_file)


@app.route('/api/face/detect', methods=['POST'])
@require_auth
def detect_faces():
    """Detect faces in uploaded image"""
    temp_file = None
    try:
        if 'image' not in request.files:
            return jsonify(DetectFacesResponse(
                success=False,
                error="No image file provided",
                detected_faces=[],
                total_faces=0
            ).dict()), 400
        
        file = request.files['image']
        is_valid, error_msg = validate_image_file(file)
        if not is_valid:
            return jsonify(DetectFacesResponse(
                success=False,
                error=error_msg,
                detected_faces=[],
                total_faces=0
            ).dict()), 400
        
        temp_file, error = file_handler.save_uploaded_file(file, "detect")
        if error:
            return jsonify({
                "success": False,
                "message": f"Error saving uploaded file: {error}"
            }), 500
        
        faces = face_service.image_processor.extract_faces_from_frame(str(temp_file))
        
        detected_faces = [
            {
                "index": idx,
                "confidence": face.get('confidence', 0.0),
                "region": {
                    "x": face['facial_area']['x'],
                    "y": face['facial_area']['y'],
                    "width": face['facial_area']['w'],
                    "height": face['facial_area']['h']
                }
            }
            for idx, face in enumerate(faces)
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
@require_auth
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


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({
        "success": False,
        "error": "Unauthorized",
        "message": "Valid JWT token required from authorized backend server"
    }), 401


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("✅ All services initialized successfully")
    logger.info(f"🌐 Starting Flask server on {settings.FLASK_HOST}:{settings.FLASK_PORT}")
    logger.info(f"📚 Swagger UI available at: http://{settings.FLASK_HOST}:{settings.FLASK_PORT}/apidoc")
    logger.info("="*60)
    
    app.run(
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=settings.FLASK_DEBUG
    )
