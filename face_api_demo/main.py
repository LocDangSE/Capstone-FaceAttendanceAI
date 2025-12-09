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
import numpy as np
import traceback

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


@app.route('/warmup', methods=['POST', 'GET', 'OPTIONS'])
def warmup():
    """
    Warmup endpoint to preload model and avoid cold start
    Called by health checks or startup scripts to keep instance warm
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        return response, 200
    
    try:
        import cv2
        # Create dummy 224x224 RGB image
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        temp_path = settings.TEMP_FOLDER / f"warmup_{uuid.uuid4()}.jpg"
        cv2.imwrite(str(temp_path), dummy_img)
        
        # Trigger model loading and face detection
        _ = face_service.embedding_cache.generate_embedding(str(temp_path))
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
        
        logger.info("✅ Warmup completed - model loaded and ready")
        
        response = jsonify({
            "status": "warmed_up",
            "model": settings.DEEPFACE_MODEL,
            "message": "Model preloaded successfully"
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        error_response = jsonify({
            "status": "warmup_failed",
            "error": str(e)
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
                    "description": "Returns comprehensive statistics about loaded camps, in-memory cache, and Redis-persistent embeddings storage.",
                    "tags": ["Face Database"],
                    "security": [{"bearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Statistics retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "loaded_camps": {
                                                "type": "object",
                                                "description": "Dictionary of loaded camps with face counts",
                                                "additionalProperties": {"type": "integer"}
                                            },
                                            "total_camps": {"type": "integer", "description": "Total number of loaded camps"},
                                            "cache_stats": {
                                                "type": "object",
                                                "description": "In-memory cache statistics",
                                                "properties": {
                                                    "total_cached": {"type": "integer", "description": "Number of embeddings in memory"},
                                                    "campers": {"type": "array", "items": {"type": "string"}, "description": "List of cached camper IDs"},
                                                    "model": {"type": "string", "description": "DeepFace model name"},
                                                    "distance_metric": {"type": "string", "description": "Distance metric used"},
                                                    "fps_limit": {"type": "integer", "description": "Recognition FPS limit"},
                                                    "threshold": {"type": "number", "description": "Confidence threshold"}
                                                }
                                            },
                                            "redis_stats": {
                                                "type": "object",
                                                "description": "Redis persistent storage statistics",
                                                "properties": {
                                                    "total_keys": {"type": "integer", "description": "Total number of Redis keys"},
                                                    "keys": {"type": "array", "items": {"type": "string"}, "description": "List of Redis keys"},
                                                    "camp_groups": {
                                                        "type": "object",
                                                        "description": "Camp and group breakdown",
                                                        "additionalProperties": {
                                                            "type": "object",
                                                            "additionalProperties": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "embeddings_count": {"type": "integer", "description": "Number of embeddings in this group"},
                                                                    "ttl_seconds": {"type": "integer", "description": "Time to live in seconds (-1 for no expiration)"},
                                                                    "expires_at": {"type": "integer", "description": "Unix timestamp when key expires"}
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "example": {
                                        "loaded_camps": {"17": 7, "18": 12},
                                        "total_camps": 2,
                                        "cache_stats": {
                                            "total_cached": 0,
                                            "campers": [],
                                            "model": "Facenet512",
                                            "distance_metric": "cosine",
                                            "fps_limit": 30,
                                            "threshold": 0.4
                                        },
                                        "redis_stats": {
                                            "total_keys": 2,
                                            "keys": ["face:embeddings:camp:17:group:14", "face:embeddings:camp:17:group:15"],
                                            "camp_groups": {
                                                "17": {
                                                    "14": {"embeddings_count": 3, "ttl_seconds": 86400, "expires_at": 1733779200},
                                                    "15": {"embeddings_count": 4, "ttl_seconds": 86400, "expires_at": 1733779200}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/recognition/recognize-group/{camp_id}/{group_id}": {
                "post": {
                    "summary": "Recognize faces for group activity (with webhook support)",
                    "description": "Recognizes faces and optionally triggers .NET webhook for real-time SignalR updates. Include activityScheduleId to enable webhook.",
                    "tags": ["Recognition"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "group_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "X-Request-ID", "in": "header", "required": False, "schema": {"type": "string"}, "description": "Optional request ID for idempotency"}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "photo": {"type": "string", "format": "binary", "description": "Image file to recognize"},
                                        "activityScheduleId": {"type": "integer", "description": "Activity schedule ID (required for webhook/SignalR)"}
                                    },
                                    "required": ["photo"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Faces recognized successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "requestId": {"type": "string"},
                                            "recognizedCampers": {"type": "array"},
                                            "webhookQueued": {"type": "boolean", "description": "True if webhook was triggered"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/recognition/recognize-activity/{camp_id}/{activity_schedule_id}": {
                "post": {
                    "summary": "Recognize faces for optional activity (with webhook support)",
                    "description": "Recognizes faces for optional activities. Automatically triggers .NET webhook for real-time SignalR updates using activity_schedule_id.",
                    "tags": ["Recognition"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "camp_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "activity_schedule_id", "in": "path", "required": True, "schema": {"type": "integer"}},
                        {"name": "X-Request-ID", "in": "header", "required": False, "schema": {"type": "string"}, "description": "Optional request ID for idempotency"}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "photo": {"type": "string", "format": "binary", "description": "Image file to recognize"}
                                    },
                                    "required": ["photo"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Faces recognized successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "requestId": {"type": "string"},
                                            "recognizedCampers": {"type": "array"},
                                            "webhookQueued": {"type": "boolean", "description": "True if webhook was triggered"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/recognition/mobile/recognize": {
                "post": {
                    "summary": "Mobile-optimized face recognition (with auto webhook)",
                    "description": "Mobile-optimized endpoint for real-time face recognition. All parameters in form-data. Automatically triggers .NET webhook for database updates and SignalR broadcasts. Returns detailed performance metrics.",
                    "tags": ["Recognition"],
                    "security": [{"bearerAuth": []}],
                    "parameters": [
                        {"name": "X-Request-ID", "in": "header", "required": False, "schema": {"type": "string"}, "description": "Optional request ID for idempotency"}
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image": {"type": "string", "format": "binary", "description": "Image file to recognize"},
                                        "activityScheduleId": {"type": "integer", "description": "Activity schedule ID (required)"},
                                        "groupId": {"type": "integer", "description": "Camper group ID (required)"},
                                        "campId": {"type": "integer", "description": "Camp ID (required)"}
                                    },
                                    "required": ["image", "activityScheduleId", "groupId", "campId"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Recognition successful with performance metrics",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "requestId": {"type": "string"},
                                            "recognizedCampers": {"type": "array"},
                                            "summary": {
                                                "type": "object",
                                                "properties": {
                                                    "totalDetected": {"type": "integer"},
                                                    "totalRecognized": {"type": "integer"},
                                                    "totalUnknown": {"type": "integer"}
                                                }
                                            },
                                            "performance": {
                                                "type": "object",
                                                "properties": {
                                                    "totalTime": {"type": "number"},
                                                    "recognitionTime": {"type": "number"},
                                                    "meetsRequirement": {"type": "boolean", "description": "True if <4 seconds"}
                                                }
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "properties": {
                                                    "activityScheduleId": {"type": "integer"},
                                                    "groupId": {"type": "integer"},
                                                    "campId": {"type": "integer"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
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
    Stores embeddings in Redis with TTL based on camp end date
    """
    request_id = str(uuid.uuid4())[:8]
    try:
        logger.info(f"[{request_id}] 📥 START: Loading camp {camp_id} (requested by {g.user.get('sub')})")
        
        # Get expire_at from request body (sent by .NET)
        expire_at = None
        if request.is_json:
            data = request.get_json()
            expire_at = data.get('expire_at')
        
        if expire_at:
            logger.info(f"[{request_id}] 📅 Redis TTL will expire at Unix timestamp: {expire_at}")
        else:
            logger.info(f"[{request_id}] ⏰ No expire_at provided, using default 1-hour TTL")
        
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        logger.info(f"[{request_id}] Checking camp folder: {camp_folder}")
        logger.info(f"[{request_id}] Folder exists: {camp_folder.exists()}")
        
        # ✅ FIXED: Check filesystem instead of worker-specific loaded_camps
        if camp_folder.exists():
            # Count existing faces in filesystem
            total_faces = 0
            groups = []
            for group_folder in camp_folder.iterdir():
                if group_folder.is_dir() and group_folder.name.startswith('camper_group_'):
                    group_id = int(group_folder.name.replace('camper_group_', ''))
                    groups.append(group_id)
                    face_files = list(group_folder.glob('avatar_*.jpg')) + list(group_folder.glob('avatar_*.png'))
                    total_faces += len(face_files)
            
            if total_faces > 0:
                logger.info(f"[{request_id}] ✅ Camp {camp_id} already loaded on filesystem ({total_faces} faces)")
                
                # Load embeddings into cache if not already cached
                cached_count = len(face_service.embedding_cache.cache)
                if cached_count == 0:
                    logger.info(f"[{request_id}] Loading embeddings from filesystem into cache...")
                    for group_id in groups:
                        group_folder = camp_folder / f"camper_group_{group_id}"
                        face_service.embedding_cache._load_embeddings_from_folder(group_folder)
                    logger.info(f"[{request_id}] ✅ Loaded {len(face_service.embedding_cache.cache)} embeddings into cache")
                
                return jsonify({
                    "success": True,
                    "message": f"Camp {camp_id} already loaded (filesystem check)",
                    "camp_id": camp_id,
                    "face_count": total_faces,
                    "groups": groups
                }), 200
        
        # Check Supabase configuration
        logger.info(f"[{request_id}] Checking Supabase configuration...")
        logger.info(f"[{request_id}] SUPABASE_ENABLED: {settings.SUPABASE_ENABLED}")
        
        if not settings.SUPABASE_ENABLED:
            logger.error(f"[{request_id}] ❌ Supabase is disabled in configuration")
            return jsonify({
                "success": False,
                "message": "Supabase storage is not enabled"
            }), 500
        
        # Download avatar images from Supabase
        logger.info(f"[{request_id}] 📥 Downloading face images from Supabase for camp {camp_id}...")
        try:
            downloaded_files = supabase_service.download_camp_faces(camp_id, str(camp_folder))
            logger.info(f"[{request_id}] ✅ Downloaded {len(downloaded_files)} face images")
        except Exception as supabase_error:
            logger.error(f"[{request_id}] ❌ Supabase download error: {str(supabase_error)}")
            logger.error(f"[{request_id}] {traceback.format_exc()}")
            return jsonify({
                "success": False,
                "message": f"Supabase error: {str(supabase_error)}",
                "camp_id": camp_id
            }), 500
        
        if not downloaded_files:
            logger.warning(f"[{request_id}] ⚠️ No face images found for camp {camp_id}")
            return jsonify({
                "success": False,
                "message": f"No face images found for camp {camp_id}",
                "camp_id": camp_id
            }), 404
        
        # Organize by groups and generate embeddings
        logger.info(f"[{request_id}] Organizing files by groups...")
        groups_data = {}
        total_faces = 0
        
        for idx, file_path in enumerate(downloaded_files, 1):
            if idx % 10 == 0:
                logger.info(f"[{request_id}] Progress: {idx}/{len(downloaded_files)} files processed")
            
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
        
        logger.info(f"[{request_id}] Found {len(groups_data)} groups")
        
        # Load embeddings for each group and store in Redis immediately
        successfully_loaded = 0
        for group_id, face_files in groups_data.items():
            try:
                group_folder = camp_folder / f"camper_group_{group_id}"
                logger.info(f"[{request_id}] ⚡ Generating embeddings for group {group_id} ({len(face_files)} faces)...")
                
                # Clear cache before loading each group to avoid mixing embeddings
                face_service.embedding_cache.clear_cache()
                face_service.embedding_cache._load_embeddings_from_folder(group_folder)
                
                # Store this group's embeddings in Redis immediately
                if expire_at and face_service.embedding_cache.cache:
                    face_service.embedding_cache.set_embeddings_redis(camp_id, group_id, face_service.embedding_cache.cache, expire_at)
                    logger.info(f"[{request_id}] ✅ Stored {len(face_service.embedding_cache.cache)} embeddings for group {group_id} in Redis")
                
                total_faces += len(face_files)
                successfully_loaded += 1
                logger.info(f"[{request_id}] ✅ Group {group_id} embeddings loaded and stored successfully")
            except Exception as group_error:
                logger.error(f"[{request_id}] ❌ Failed to load embeddings for group {group_id}: {group_error}")
                import traceback
                logger.error(f"[{request_id}] Group {group_id} traceback: {traceback.format_exc()}")
                # Continue processing other groups even if one fails
        
        if successfully_loaded == 0:
            logger.error(f"[{request_id}] ❌ Failed to load embeddings for all groups")
            return jsonify({
                "success": False,
                "message": f"Failed to load embeddings for camp {camp_id}",
                "camp_id": camp_id
            }), 500
        
        # ✅ FIXED: No need to update loaded_camps - filesystem is source of truth
        logger.info(f"[{request_id}] ✅ COMPLETE: Loaded {total_faces} faces for camp {camp_id} across {successfully_loaded}/{len(groups_data)} groups")
        
        return jsonify({
            "success": True,
            "message": f"Loaded {total_faces} faces for camp {camp_id}",
            "camp_id": camp_id,
            "face_count": total_faces,
            "groups": list(groups_data.keys())
        }), 200
    
    except Exception as e:
        logger.error(f"[{request_id}] ❌ EXCEPTION in load_camp_face_database: {str(e)}")
        logger.error(f"[{request_id}] {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "camp_id": camp_id
        }), 500


@app.route('/api/maintenance/cleanup-temp', methods=['POST'])
@require_auth
def cleanup_temp_folder():
    """
    Clean up temporary files in temp folder
    Query params:
    - older_than_hours: Only delete files older than X hours (optional, default: delete all)
    """
    try:
        # Get optional age filter from query params
        older_than_hours = request.args.get('older_than_hours', type=int)
        
        if older_than_hours:
            logger.info(f"🧹 Cleaning up temp files older than {older_than_hours}h (requested by {g.user.get('sub')})")
        else:
            logger.info(f"🧹 Cleaning up all temp files (requested by {g.user.get('sub')})")
        
        deleted_count, failed_count, error = file_handler.cleanup_temp_folder(older_than_hours)
        
        if error:
            return jsonify({
                "success": False,
                "message": f"Error cleaning temp folder: {error}"
            }), 500
        
        message = f"Cleaned up temp folder: {deleted_count} items deleted"
        if older_than_hours:
            message += f" (older than {older_than_hours} hours)"
        if failed_count > 0:
            message += f", {failed_count} failed"
        
        return jsonify({
            "success": True,
            "message": message,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "filter": f"{older_than_hours}h" if older_than_hours else "all"
        }), 200
    
    except Exception as e:
        logger.error(f"Error in cleanup_temp_folder endpoint: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
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
        
        # ✅ FIXED: No need to update loaded_camps - filesystem is source of truth
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
    """Get statistics about loaded camps (FIXED: Filesystem-based, worker-independent)"""
    try:
        # Scan face_database directory to find loaded camps (source of truth for multi-worker setup)
        face_db_root = settings.DATABASE_FOLDER
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
                        # Return face_count directly (not nested) for .NET compatibility
                        detected_camps[camp_id] = total_faces
        
        # Only report Redis-based embedding stats
        redis_stats = face_service.embedding_cache.get_redis_stats()
        return jsonify({
            "loaded_camps": detected_camps,
            "total_camps": len(detected_camps),
            "redis_stats": redis_stats
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
        
        # ✅ FIXED: No need to clear loaded_camps - filesystem is source of truth
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
        
        # ✅ FIXED: No need to update loaded_camps - filesystem is source of truth
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
    NOW WITH WEBHOOK SUPPORT: Updates .NET backend + SignalR broadcast
    """
    from services.webhook_service import start_webhook_thread
    
    temp_file = None
    request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"[{request_id}] 🎯 Group recognition: camp={camp_id}, group={group_id} (by {g.user.get('sub')})")
        
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
        
        # Verify embeddings are registered for this group (lazy loading)
        group_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}" / f"camper_group_{group_id}"
        
        if not group_folder.exists():
            return jsonify({
                "success": False,
                "message": f"Group folder not found. Please preload camp {camp_id} first.",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        
        # Only use Redis embeddings, do not fall back to RAM or filesystem
        embeddings = face_service.embedding_cache.get_embeddings_redis(camp_id, group_id)
        if not embeddings:
            logger.warning(f"⚠️ No embeddings found in Redis for camp {camp_id}, group {group_id}")
            return jsonify({
                "success": False,
                "message": f"No embeddings found in Redis for camp {camp_id}, group {group_id}. Please preload the camp/group.",
                "campId": camp_id,
                "groupId": group_id
            }), 404
        else:
            logger.info(f"✅ Using {len(embeddings)} embeddings from Redis for recognition")
        
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
        
        logger.info(f"[{request_id}] ✅ Recognized {len(recognized_campers)} camper(s) from group {group_id}")
        
        # Get activityScheduleId from request (required for webhook)
        activity_schedule_id = request.form.get('activityScheduleId') or request.args.get('activityScheduleId')
        
        # Prepare response
        response_data = {
            "success": True,
            "message": f"Recognized {len(recognized_campers)} camper(s) from group {group_id}",
            "requestId": request_id,
            "campId": camp_id,
            "groupId": group_id,
            "sessionId": session_id,
            "recognizedCampers": recognized_campers,
            "totalFacesDetected": result.get('total_faces_detected', 0),
            "matchedFaces": len(recognized_campers),
            "processingTimeMs": int(result.get('processing_time', 0) * 1000),
            "timestamp": get_now().isoformat()
        }
        
        # ========== TRIGGER WEBHOOK TO .NET (if activityScheduleId provided) ==========
        if activity_schedule_id:
            logger.info(f"[{request_id}] 🔄 Starting webhook to .NET for activity {activity_schedule_id}")
            
            # Extract user info from JWT token
            user_id = g.user.get('id') or g.user.get('sub') or g.user.get('userId') or 'system'
            username = g.user.get('name') or g.user.get('email') or g.user.get('username') or 'system'
            
            # Start background webhook thread
            start_webhook_thread(
                request_id=request_id,
                activity_schedule_id=int(activity_schedule_id),
                group_id=group_id,
                camp_id=camp_id,
                results=result.get('recognized_campers', []),
                user_id=user_id,
                username=username
            )
            
            response_data["webhookQueued"] = True
            response_data["activityScheduleId"] = int(activity_schedule_id)
        else:
            logger.warning(f"[{request_id}] ⚠️ No activityScheduleId provided, skipping webhook")
            response_data["webhookQueued"] = False
        
        return jsonify(response_data), 200
    
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
    NOW WITH WEBHOOK SUPPORT: Updates .NET backend + SignalR broadcast
    """
    from services.webhook_service import start_webhook_thread
    
    temp_file = None
    request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
    
    try:
        logger.info(f"[{request_id}] 🎯 Activity recognition: camp={camp_id}, activity={activity_schedule_id} (by {g.user.get('sub')})")
        
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
        
        logger.info(f"[{request_id}] ✅ Recognized {len(recognized_campers)} camper(s) for activity {activity_schedule_id}")
        
        # Prepare response
        response_data = {
            "success": True,
            "message": f"Recognized {len(recognized_campers)} camper(s)",
            "requestId": request_id,
            "campId": camp_id,
            "activityScheduleId": activity_schedule_id,
            "sessionId": session_id,
            "recognizedCampers": recognized_campers,
            "totalFacesDetected": result.get('total_faces', 0),
            "matchedFaces": len(recognized_campers),
            "processingTimeMs": int(result.get('processing_time', 0) * 1000),
            "timestamp": get_now().isoformat()
        }
        
        # ========== TRIGGER WEBHOOK TO .NET (always for activity endpoint) ==========
        logger.info(f"[{request_id}] 🔄 Starting webhook to .NET for activity {activity_schedule_id}")
        
        # Extract user info
        user_id = g.user.get('id') or g.user.get('sub') or g.user.get('userId') or 'system'
        username = getattr(g, 'username', g.user.get('name', g.user.get('email', 'unknown')))
        
        # Start background webhook thread
        start_webhook_thread(
            request_id=request_id,
            activity_schedule_id=activity_schedule_id,
            group_id=None,  # Optional activities may not have group_id
            camp_id=camp_id,
            results=result.get('recognized_campers', []),
            user_id=user_id,
            username=username
        )
        
        response_data["webhookQueued"] = True
        
        return jsonify(response_data), 200
    
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
# Mobile Direct Recognition API (NEW FLOW)
# ============================================================================

@app.route('/api/recognition/mobile/recognize', methods=['POST'])
@require_auth
def mobile_recognize_faces():
    """
    🎯 Mobile-optimized endpoint: Direct recognition with async .NET webhook
    
    Flow:
    1. Mobile → Python (this endpoint)
    2. Python recognizes faces (2-4s)
    3. Python returns to mobile IMMEDIATELY ✅
    4. Python → .NET webhook (background, async)
    5. .NET updates attendance + broadcasts SignalR
    
    Requirements:
    - activityScheduleId: int (required)
    - groupId: int (required)
    - campId: int (required)
    - image: file (required)
    
    Headers:
    - Authorization: Bearer {userJwt}
    - X-Request-ID: {uuid} (optional, auto-generated if missing)
    """
    from services.webhook_service import start_webhook_thread
    
    request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
    start_time = datetime.now()
    image_path = None
    
    try:
        logger.info(f"[{request_id}] 📱 Mobile recognition request started")
        
        # ========== PHASE 1: VALIDATION (Target: <50ms) ==========
        validation_start = datetime.now()
        
        # Get parameters
        activity_schedule_id = request.form.get('activityScheduleId')
        group_id = request.form.get('groupId')
        camp_id = request.form.get('campId')
        
        if not all([activity_schedule_id, group_id, camp_id]):
            return jsonify({
                "success": False,
                "error": "Missing required parameters: activityScheduleId, groupId, campId"
            }), 400
        
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400
        
        # Extract user info from JWT (set by @require_auth middleware)
        user_id = g.user.get('id') or g.user.get('sub') or g.user.get('userId') or 'system'
        username = getattr(g, 'username', g.user.get('name', g.user.get('email', 'unknown')))
        
        validation_time = (datetime.now() - validation_start).total_seconds()
        logger.info(f"[{request_id}] ✅ Validation completed in {validation_time*1000:.0f}ms")
        
        # ========== PHASE 2: AUTHORIZATION CHECK (Target: <10ms) ==========
        auth_start = datetime.now()
        
        # Quick filesystem check (NO database query - optimized for speed)
        camp_folder = settings.DATABASE_FOLDER / f"camp_{camp_id}"
        group_folder = camp_folder / f"camper_group_{group_id}"
        
        if not camp_folder.exists():
            return jsonify({
                "success": False,
                "error": f"Camp {camp_id} not loaded. Please pre-load the camp first.",
                "requiresPreload": True,
                "campId": int(camp_id)
            }), 404
        
        if not group_folder.exists():
            available_groups = [
                int(g.name.replace('camper_group_', ''))
                for g in camp_folder.iterdir()
                if g.is_dir() and g.name.startswith('camper_group_')
            ]
            return jsonify({
                "success": False,
                "error": f"Group {group_id} not found in camp {camp_id}",
                "availableGroups": available_groups,
                "campId": int(camp_id),
                "groupId": int(group_id)
            }), 404
        
        auth_time = (datetime.now() - auth_start).total_seconds()
        logger.info(f"[{request_id}] ✅ Authorization check completed in {auth_time*1000:.0f}ms")
        
        # ========== PHASE 3: SAVE IMAGE (Target: <100ms) ==========
        save_start = datetime.now()
        
        # Save to temp folder
        image_path, error = file_handler.save_uploaded_file(image_file, f"mobile_{request_id}")
        if error:
            return jsonify({
                "success": False,
                "error": f"Failed to save image: {error}"
            }), 500
        
        save_time = (datetime.now() - save_start).total_seconds()
        logger.info(f"[{request_id}] ✅ Image saved in {save_time*1000:.0f}ms")
        
        # ========== PHASE 4: FACE RECOGNITION (Target: 2-3s) ==========
        recognition_start = datetime.now()
        
        logger.info(
            f"[{request_id}] 🎯 Starting face recognition "
            f"(camp={camp_id}, group={group_id}, activity={activity_schedule_id})"
        )
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Use existing recognition service (already optimized)
        result = face_service.check_attendance(
            image_path=str(image_path),
            session_id=session_id,
            save_results=False  # Skip disk I/O for speed
        )
        
        recognition_time = (datetime.now() - recognition_start).total_seconds()
        logger.info(f"[{request_id}] ✅ Recognition completed in {recognition_time:.2f}s")
        
        # ========== PHASE 5: PREPARE RESPONSE (Target: <50ms) ==========
        response_start = datetime.now()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Extract recognized campers from result
        recognized_faces = result.get('recognized_campers', [])
        
        # Prepare recognized campers list
        recognized_campers = []
        for r in recognized_faces:
            camper_id = r.get('camper_id', -1)
            # Convert to int safely (handle string or int)
            camper_id = int(camper_id) if camper_id not in [None, -1, '-1'] else -1
            if camper_id > 0:  # Only include recognized faces
                recognized_campers.append({
                    "camperId": int(camper_id),
                    "confidence": round(1.0 - r.get('distance', 0.0), 3),  # Convert distance to confidence
                    "distance": round(r.get('distance', 0.0), 3),
                    "boundingBox": r.get('face_region'),
                    "faceArea": r.get('face_region', {}).get('width', 0) * r.get('face_region', {}).get('height', 0) if r.get('face_region') else 0
                })
        
        total_faces = result.get('total_faces_detected', 0)
        
        response_data = {
            "success": True,
            "requestId": request_id,
            "recognizedCampers": recognized_campers,
            "summary": {
                "totalDetected": total_faces,
                "totalRecognized": len(recognized_campers),
                "totalUnknown": total_faces - len(recognized_campers)
            },
            "performance": {
                "totalTime": round(total_time, 3),
                "validationTime": round(validation_time, 3),
                "authorizationTime": round(auth_time, 3),
                "imageLoadTime": round(save_time, 3),
                "recognitionTime": round(recognition_time, 3),
                "meetsRequirement": total_time < 4.0  # Capstone requirement: <4s
            },
            "metadata": {
                "activityScheduleId": int(activity_schedule_id),
                "groupId": int(group_id),
                "campId": int(camp_id),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "processedBy": username
            }
        }
        
        response_time = (datetime.now() - response_start).total_seconds()
        logger.info(f"[{request_id}] ✅ Response prepared in {response_time*1000:.0f}ms")
        
        # ========== PHASE 6: RETURN TO MOBILE IMMEDIATELY ==========
        logger.info(
            f"[{request_id}] 📱 Returning to mobile "
            f"(total: {total_time:.2f}s, meets requirement: {total_time < 4.0})"
        )
        
        # ========== PHASE 7: ASYNC DATABASE UPDATE (Background) ==========
        # Start background thread to update .NET database
        start_webhook_thread(
            request_id=request_id,
            activity_schedule_id=int(activity_schedule_id),
            group_id=int(group_id),
            camp_id=int(camp_id),
            results=recognized_faces,  # Pass the recognized_campers list
            user_id=user_id,
            username=username
        )
        
        logger.info(f"[{request_id}] 🔄 Database update queued (async)")
        
        # Cleanup temp file (async to save time)
        if image_path and Path(image_path).exists():
            import threading
            threading.Thread(
                target=lambda: Path(image_path).unlink(),
                daemon=True
            ).start()
        
        return jsonify(response_data), 200
    
    except Exception as e:
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"[{request_id}] ❌ Error after {total_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup temp file on error
        if image_path and Path(image_path).exists():
            try:
                Path(image_path).unlink()
            except:
                pass
        
        return jsonify({
            "success": False,
            "error": str(e),
            "requestId": request_id,
            "totalTime": round(total_time, 3)
        }), 500


@app.route('/api/health/webhook', methods=['GET'])
@require_auth
def webhook_health_check():
    """
    Test .NET backend connectivity and service JWT generation
    Returns webhook health status for debugging
    """
    try:
        from services.jwt_service import generate_service_jwt
        import requests
        
        # Test JWT generation
        try:
            service_token = generate_service_jwt()
            jwt_valid = len(service_token) > 0
        except Exception as e:
            jwt_valid = False
            jwt_error = str(e)
        
        # Test .NET connectivity
        dotnet_reachable = False
        dotnet_response_time = None
        dotnet_error = None
        
        try:
            start = datetime.now()
            response = requests.get(
                f"{settings.DOTNET_API_URL}/health",
                timeout=5
            )
            dotnet_response_time = (datetime.now() - start).total_seconds()
            dotnet_reachable = response.status_code == 200
        except Exception as e:
            dotnet_error = str(e)
        
        return jsonify({
            "success": True,
            "webhook": {
                "jwtValid": jwt_valid,
                "jwtError": jwt_error if not jwt_valid else None,
                "dotnetReachable": dotnet_reachable,
                "dotnetUrl": settings.DOTNET_API_URL,
                "dotnetResponseTime": round(dotnet_response_time, 3) if dotnet_response_time else None,
                "dotnetError": dotnet_error,
                "webhookTimeout": settings.DOTNET_WEBHOOK_TIMEOUT,
                "webhookRetryCount": settings.DOTNET_WEBHOOK_RETRY_COUNT
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 200
    
    except Exception as e:
        logger.error(f"Webhook health check error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
