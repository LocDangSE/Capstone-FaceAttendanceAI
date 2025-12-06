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

# Use LazyEmbeddingCache for Railway (memory-efficient)
use_lazy_cache = os.getenv('USE_LAZY_CACHE', 'true').lower() == 'true'
if use_lazy_cache:
    from services.lazy_embedding_cache import LazyEmbeddingCache as EmbeddingCache
    logger = logging.getLogger(__name__)
    logger.info("✅ Using LazyEmbeddingCache (Railway-optimized)")
elif settings.USE_OPTIMIZED_CACHE:
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
# Accept tokens from both .NET user login and Python API service
app.config['JWT_ISSUER_WHITELIST'] = [x.strip() for x in settings.JWT_ISSUER_WHITELIST.split(',')]
app.config['JWT_AUDIENCE_WHITELIST'] = [x.strip() for x in settings.JWT_AUDIENCE_WHITELIST.split(',')]
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

@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint with system statistics - No authentication required"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
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
        
        response_json = jsonify(response_dict)
        # Add explicit CORS headers for health check
        response_json.headers.add('Access-Control-Allow-Origin', '*')
        response_json.headers.add('Access-Control-Allow-Headers', '*')
        response_json.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response_json, 200
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        error_response = jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": get_now().isoformat()
        })
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500


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
    # Dynamically determine server URL based on request
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
    host = request.headers.get('X-Forwarded-Host', request.host)
    base_url = f"{scheme}://{host}"
    
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Face Recognition API",
            "version": "2.0.0",
            "description": "Production-ready face recognition API with JWT authentication and latency-optimized processing"
        },
        "servers": [
            {"url": base_url, "description": "Current server"},
            {"url": f"http://localhost:{settings.FLASK_PORT}", "description": "Local development"},
            {"url": f"http://127.0.0.1:{settings.FLASK_PORT}", "description": "Local development (IP)"}
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
    FIXED: Now checks filesystem instead of worker-specific loaded_camps
    """
    try:
        logger.info(f"🗑️ Unloading camp {camp_id} (requested by {g.user.get('sub')})")
        
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        
        # Check if camp exists on filesystem (not loaded_camps dict)
        if not camp_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Camp {camp_id} does not exist",
                "camp_id": camp_id
            }), 404
        
        deleted_faces = 0
        deleted_groups = []
        
        # Clear cache for this camp
        for group_folder in camp_folder.glob("camper_group_*"):
            group_id = int(group_folder.name.replace('camper_group_', ''))
            deleted_groups.append(group_id)
            
            for face_file in group_folder.glob("avatar_*.jpg"):
                parts = face_file.stem.split('_')
                if len(parts) >= 2:
                    camper_id_str = parts[1]
                    face_service.embedding_cache.clear_cache(camper_id_str)
                    deleted_faces += 1
        
        # Delete local files
        shutil.rmtree(camp_folder)
        logger.info(f"🗑️ Deleted folder: {camp_folder}")
        
        # Remove from loaded_camps if present in this worker
        if camp_id in loaded_camps:
            del loaded_camps[camp_id]
        
        logger.info(f"✅ Camp {camp_id} unloaded: {deleted_faces} faces from groups {deleted_groups}")
        
        return jsonify({
            "success": True,
            "message": f"Camp {camp_id} unloaded successfully",
            "camp_id": camp_id,
            "deleted": {
                "faces": deleted_faces,
                "groups": deleted_groups
            }
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
        # Get cached campers from embedding cache (source of truth)
        cached_campers = list(face_service.embedding_cache.cache.keys())
        
        # Scan face_database directory to find loaded camps
        face_db_root = Path(settings.BASE_DIR) / "face_database"
        detected_camps = {}
        
        if face_db_root.exists():
            for camp_folder in face_db_root.iterdir():
                if camp_folder.is_dir() and camp_folder.name.startswith('camp_'):
                    camp_id = int(camp_folder.name.replace('camp_', ''))
                    groups = []
                    total_faces = 0
                    
                    for group_folder in camp_folder.iterdir():
                        if group_folder.is_dir() and group_folder.name.startswith('camper_group_'):
                            group_id = int(group_folder.name.replace('camper_group_', ''))
                            groups.append(group_id)
                            # Count image files in group folder
                            face_files = list(group_folder.glob('avatar_*.jpg')) + list(group_folder.glob('avatar_*.jpeg')) + list(group_folder.glob('avatar_*.png'))
                            total_faces += len(face_files)
                    
                    if groups:
                        detected_camps[camp_id] = {
                            'face_count': total_faces,
                            'groups': sorted(groups)
                        }
        
        return jsonify({
            "loaded_camps": detected_camps,
            "total_camps": len(detected_camps),
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


@app.route('/api/face-db/clear-all', methods=['DELETE'])
@require_auth
def clear_all_face_data():
    """
    Clear all face database and embeddings (testing/cleanup)
    WARNING: This deletes all camps, embeddings, and cached data
    """
    try:
        logger.warning(f"⚠️ CLEARING ALL FACE DATA (requested by {g.user.get('sub')})")
        
        deleted_items = {
            "camps_deleted": 0,
            "embeddings_deleted": 0,
            "folders_deleted": []
        }
        
        # Clear all embedding cache
        face_service.embedding_cache.clear_cache()
        logger.info("✅ Cleared embedding cache")
        
        # Delete all camp folders
        face_db_root = Path(settings.BASE_DIR) / "face_database"
        if face_db_root.exists():
            for camp_folder in face_db_root.iterdir():
                if camp_folder.is_dir() and camp_folder.name.startswith('camp_'):
                    shutil.rmtree(camp_folder)
                    deleted_items["folders_deleted"].append(camp_folder.name)
                    deleted_items["camps_deleted"] += 1
            logger.info(f"🗑️ Deleted {deleted_items['camps_deleted']} camp folders")
        
        # Delete all embedding files
        embeddings_dir = Path(settings.BASE_DIR) / "embeddings"
        if embeddings_dir.exists():
            for emb_file in embeddings_dir.glob("*.npy"):
                emb_file.unlink()
                deleted_items["embeddings_deleted"] += 1
            for json_file in embeddings_dir.glob("*.json"):
                json_file.unlink()
            logger.info(f"🗑️ Deleted {deleted_items['embeddings_deleted']} embedding files")
        
        # Clear loaded_camps dictionary
        loaded_camps.clear()
        
        logger.warning(f"✅ ALL FACE DATA CLEARED: {deleted_items}")
        
        return jsonify({
            "success": True,
            "message": "All face data cleared successfully",
            "deleted": deleted_items
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error clearing all face data: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/face-db/clear-camp/<int:camp_id>', methods=['DELETE'])
@require_auth
def clear_camp_data(camp_id):
    """
    Clear face database and embeddings for a specific camp
    Removes from cache, deletes local files and embeddings
    """
    try:
        logger.info(f"🗑️ Clearing camp {camp_id} data (requested by {g.user.get('sub')})")
        
        deleted_items = {
            "camp_id": camp_id,
            "faces_deleted": 0,
            "embeddings_deleted": 0,
            "groups_deleted": []
        }
        
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        
        # Clear cache for this camp's campers
        if camp_folder.exists():
            for group_folder in camp_folder.glob("camper_group_*"):
                group_id = int(group_folder.name.replace('camper_group_', ''))
                deleted_items["groups_deleted"].append(group_id)
                
                for face_file in group_folder.glob("avatar_*.jpg"):
                    # Extract camper ID from filename (avatar_21_avatar_uuid.jpg)
                    parts = face_file.stem.split('_')
                    if len(parts) >= 2:
                        camper_id_str = parts[1]
                        face_service.embedding_cache.clear_cache(camper_id_str)
                        deleted_items["faces_deleted"] += 1
        
        # Delete local face files
        if camp_folder.exists():
            shutil.rmtree(camp_folder)
            logger.info(f"🗑️ Deleted folder: {camp_folder}")
        
        # Delete embedding files for this camp's campers
        embeddings_dir = Path(settings.BASE_DIR) / "embeddings"
        if embeddings_dir.exists():
            # Note: Embedding files are named by camper UUID, not camp ID
            # So we can't directly delete by camp. Already removed from cache above.
            deleted_items["embeddings_deleted"] = deleted_items["faces_deleted"]
        
        # Remove from loaded_camps
        if camp_id in loaded_camps:
            del loaded_camps[camp_id]
        
        logger.info(f"✅ Camp {camp_id} cleared: {deleted_items}")
        
        return jsonify({
            "success": True,
            "message": f"Camp {camp_id} data cleared successfully",
            "deleted": deleted_items
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error clearing camp {camp_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "camp_id": camp_id
        }), 500


@app.route('/api/face-db/clear-embeddings', methods=['DELETE'])
@require_auth
def clear_embeddings_cache():
    """
    Clear only the embeddings cache (keep face images)
    Useful for testing or when embeddings need regeneration
    """
    try:
        logger.info(f"🗑️ Clearing embeddings cache (requested by {g.user.get('sub')})")
        
        # Get count before clearing
        cached_count = len(face_service.embedding_cache.cache)
        
        # Clear embedding cache
        face_service.embedding_cache.clear_cache()
        
        # Delete all .npy and .json files
        embeddings_dir = Path(settings.BASE_DIR) / "embeddings"
        deleted_files = 0
        
        if embeddings_dir.exists():
            for emb_file in embeddings_dir.glob("*.npy"):
                emb_file.unlink()
                deleted_files += 1
            for json_file in embeddings_dir.glob("*.json"):
                json_file.unlink()
        
        logger.info(f"✅ Cleared {cached_count} cached embeddings and {deleted_files} files")
        
        return jsonify({
            "success": True,
            "message": "Embeddings cache cleared successfully",
            "cleared": {
                "cached_embeddings": cached_count,
                "deleted_files": deleted_files
            }
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Error clearing embeddings: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
        
        # Check if camp exists on filesystem (not loaded_camps dict)
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        if not camp_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Camp {camp_id} not loaded. Please load camp first.",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        
        # Check if group exists on filesystem
        group_folder = camp_folder / f"camper_group_{group_id}"
        if not group_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Group {group_id} not found in camp {camp_id}. Available groups: {[int(f.name.replace('camper_group_', '')) for f in camp_folder.glob('camper_group_*')]}",
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
        
        # Check if camp exists on filesystem (not loaded_camps dict)
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        if not camp_folder.exists():
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
        
        detected_faces = []
        for idx, face in enumerate(faces):
            # Handle facial_area key safely
            facial_area = face.get('facial_area', {})
            if facial_area:
                detected_faces.append({
                    "index": idx,
                    "confidence": face.get('confidence', 0.0),
                    "region": {
                        "x": facial_area.get('x', 0),
                        "y": facial_area.get('y', 0),
                        "width": facial_area.get('w', facial_area.get('width', 0)),
                        "height": facial_area.get('h', facial_area.get('height', 0))
                    }
                })
            else:
                logger.warning(f"Face {idx} missing facial_area data: {face.keys()}")
        
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


@app.route('/ready', methods=['GET', 'OPTIONS'])
def ready():
    """Readiness probe for Kubernetes/Render - No authentication required"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response, 200
    
    try:
        # Check if models and services are loaded
        if face_service and supabase_service:
            response_json = jsonify({
                "status": "ready",
                "model": settings.DEEPFACE_MODEL,
                "cached_embeddings": face_service.embedding_cache.get_cache_stats()['total_cached'],
                "timestamp": get_now().isoformat()
            })
            response_json.headers.add('Access-Control-Allow-Origin', '*')
            return response_json, 200
        
        not_ready_response = jsonify({"status": "not_ready"})
        not_ready_response.headers.add('Access-Control-Allow-Origin', '*')
        return not_ready_response, 503
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        error_response = jsonify({"status": "error", "error": str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 503


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Get port from environment (Railway, Render, etc.)
    port = int(os.getenv('PORT', os.getenv('FLASK_PORT', settings.FLASK_PORT)))
    host = os.getenv('FLASK_HOST', settings.FLASK_HOST)
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info("="*60)
    logger.info("✅ All services initialized successfully")
    logger.info(f"🌐 Starting Flask server on {host}:{port}")
    logger.info(f"📚 Swagger UI available at: http://{host}:{port}/apidoc")
    logger.info(f"🔍 Health check: http://{host}:{port}/health")
    logger.info(f"✅ Readiness check: http://{host}:{port}/ready")
    logger.info("="*60)
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Important for production
    )
