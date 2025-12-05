"""
Centralized Configuration Management
Type-safe settings using Pydantic BaseSettings
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Explicitly load .env file from the project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # ==================== Flask Settings ====================
    FLASK_HOST: str = Field(default="0.0.0.0", description="Flask server host")
    FLASK_PORT: int = Field(default=5000, description="Flask server port")
    FLASK_DEBUG: bool = Field(default=True, description="Flask debug mode")
    
    # ==================== Supabase Settings ====================
    SUPABASE_URL: str = Field(default="", description="Supabase project URL")
    SUPABASE_KEY: str = Field(default="", description="Supabase anon/public key")
    SUPABASE_BUCKET: str = Field(default="attendance-sessions", description="Supabase storage bucket for attendance sessions")
    SUPABASE_ENABLED: bool = Field(default=False, description="Enable Supabase cloud storage")
    
    # ==================== JWT Authentication Settings ====================
    JWT_SECRET_KEY: str = Field(default="dev-secret-CHANGE-IN-PRODUCTION", description="JWT secret key (must match .NET backend)")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_ISSUER: str = Field(default="SummerCampBackend", description="JWT token issuer")
    JWT_AUDIENCE: str = Field(default="face-recognition-api", description="JWT token audience")
    JWT_ISSUER_WHITELIST: str = Field(default="SummerCampBackend,https://localhost:7075,https://inheritable-narcisa-uncuriously.ngrok-free.dev,https://summercampapi-339197681269.asia-southeast1.run.app", description="Comma-separated list of allowed issuers")
    JWT_AUDIENCE_WHITELIST: str = Field(default="face-recognition-api,https://localhost:7075,https://inheritable-narcisa-uncuriously.ngrok-free.dev,https://summercampapi-339197681269.asia-southeast1.run.app", description="Comma-separated list of allowed audiences")
    
    # ==================== DeepFace Settings ====================
    DEEPFACE_MODEL: str = Field(default="Facenet512", description="Face recognition model")
    DEEPFACE_DETECTOR: str = Field(default="opencv", description="Face detector backend")
    DEEPFACE_DISTANCE_METRIC: str = Field(default="cosine", description="Distance metric for comparison")
    CONFIDENCE_THRESHOLD: float = Field(default=0.6, description="Recognition confidence threshold (0.0-1.0)")
    
    # ==================== Image Processing Settings ====================
    IMAGE_MAX_SIZE: int = Field(default=5 * 1024 * 1024, description="Max file size in bytes (5MB)")
    IMAGE_RESIZE_WIDTH: int = Field(default=256, description="Target image width for processing")
    IMAGE_RESIZE_HEIGHT: int = Field(default=256, description="Target image height for processing")
    IMAGE_QUALITY: int = Field(default=85, description="JPEG compression quality (1-100)")
    ALLOWED_EXTENSIONS: set = Field(default={'png', 'jpg', 'jpeg'}, description="Allowed file extensions")
    
    # ==================== Performance Settings ====================
    RECOGNITION_FPS_LIMIT: float = Field(default=1.0, description="Max FPS for recognition (rate limiting)")
    BATCH_SIZE: int = Field(default=10, description="Batch size for bulk operations")
    CACHE_PRELOAD: bool = Field(default=True, description="Preload embeddings on startup")
    
    # Optimization flags
    USE_OPTIMIZED_CACHE: bool = Field(default=True, description="Use FAISS-accelerated embedding cache")
    USE_OPTIMIZED_PROCESSOR: bool = Field(default=True, description="Use optimized image processor")
    USE_LAZY_CACHE: bool = Field(default=True, description="Use lazy-loading cache (Railway-optimized)")
    
    # Railway-specific settings
    LAZY_CACHE_MAX_MEMORY: int = Field(default=50, description="Max embeddings in memory for lazy cache")
    PRELOAD_BATCH_SIZE: int = Field(default=5, description="Batch size for preloading embeddings")
    ASYNC_PRELOAD: bool = Field(default=True, description="Use async preloading to avoid blocking")
    
    # ==================== Directory Paths ====================
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    UPLOAD_FOLDER: Optional[Path] = Field(default=None)
    TEMP_FOLDER: Optional[Path] = Field(default=None)
    DATABASE_FOLDER: Optional[Path] = Field(default=None)
    SESSIONS_FOLDER: Optional[Path] = Field(default=None)
    EMBEDDINGS_FOLDER: Optional[Path] = Field(default=None)
    LOGS_FOLDER: Optional[Path] = Field(default=None)
    
    # ==================== Validators ====================
    @field_validator('CONFIDENCE_THRESHOLD')
    @classmethod
    def validate_threshold(cls, v):
        """Ensure threshold is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('CONFIDENCE_THRESHOLD must be between 0.0 and 1.0')
        return v
    
    @field_validator('IMAGE_QUALITY')
    @classmethod
    def validate_quality(cls, v):
        """Ensure image quality is between 1 and 100"""
        if not 1 <= v <= 100:
            raise ValueError('IMAGE_QUALITY must be between 1 and 100')
        return v
    
    @field_validator('DEEPFACE_MODEL')
    @classmethod
    def validate_model(cls, v):
        """Validate DeepFace model name"""
        valid_models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace']
        if v not in valid_models:
            raise ValueError(f'DEEPFACE_MODEL must be one of: {", ".join(valid_models)}')
        return v
    
    @field_validator('DEEPFACE_DETECTOR')
    @classmethod
    def validate_detector(cls, v):
        """Validate detector backend"""
        valid_detectors = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        if v not in valid_detectors:
            raise ValueError(f'DEEPFACE_DETECTOR must be one of: {", ".join(valid_detectors)}')
        return v
    
    @field_validator('DEEPFACE_DISTANCE_METRIC')
    @classmethod
    def validate_distance(cls, v):
        """Validate distance metric"""
        valid_metrics = ['cosine', 'euclidean', 'euclidean_l2']
        if v not in valid_metrics:
            raise ValueError(f'DEEPFACE_DISTANCE_METRIC must be one of: {", ".join(valid_metrics)}')
        return v
    
    def __init__(self, **kwargs):
        """Initialize settings and create directory paths"""
        super().__init__(**kwargs)
        
        # Set default paths relative to BASE_DIR
        if self.UPLOAD_FOLDER is None:
            self.UPLOAD_FOLDER = self.BASE_DIR / "uploads"
        if self.TEMP_FOLDER is None:
            self.TEMP_FOLDER = self.BASE_DIR / "temp"
        if self.DATABASE_FOLDER is None:
            self.DATABASE_FOLDER = self.BASE_DIR / "face_database"
        if self.SESSIONS_FOLDER is None:
            self.SESSIONS_FOLDER = self.BASE_DIR / "sessions"
        if self.EMBEDDINGS_FOLDER is None:
            self.EMBEDDINGS_FOLDER = self.BASE_DIR / "embeddings"
        if self.LOGS_FOLDER is None:
            self.LOGS_FOLDER = self.BASE_DIR / "logs"
        
        # Create all directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all required directories if they don't exist"""
        directories = [
            self.UPLOAD_FOLDER,
            self.TEMP_FOLDER,
            self.DATABASE_FOLDER,
            self.SESSIONS_FOLDER,
            self.EMBEDDINGS_FOLDER,
            self.LOGS_FOLDER
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        """Pydantic config"""
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Helper functions
def get_session_path(session_id: str) -> Path:
    """Get path for a specific session"""
    session_path = settings.SESSIONS_FOLDER / f"session_{session_id}"
    session_path.mkdir(parents=True, exist_ok=True)
    return session_path


def get_embedding_path(camper_id: str) -> Path:
    """Get path for a camper's embedding file"""
    return settings.EMBEDDINGS_FOLDER / f"{camper_id}.npy"


def get_activity_schedule_database_path(activity_schedule_id: str) -> Path:
    """Get path for an activity schedule's face database"""
    schedule_path = settings.DATABASE_FOLDER / activity_schedule_id
    schedule_path.mkdir(parents=True, exist_ok=True)
    return schedule_path
