"""
Pydantic Data Models
Type-safe request/response schemas for API
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class FaceRegion(BaseModel):
    """Face region coordinates"""
    x: int = Field(..., description="X coordinate of top-left corner")
    y: int = Field(..., description="Y coordinate of top-left corner")
    width: int = Field(..., description="Width of face region")
    height: int = Field(..., description="Height of face region")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x": 100,
                "y": 150,
                "width": 200,
                "height": 250
            }
        }


class RecognizedCamper(BaseModel):
    """Recognized camper information"""
    camper_id: str = Field(..., description="Camper identifier (UUID)")
    distance: float = Field(..., ge=0.0, description="Embedding distance (for internal use)")
    face_region: FaceRegion = Field(..., description="Face location in image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "camper_id": "550e8400-e29b-41d4-a716-446655440000",
                "distance": 0.15,
                "face_region": {
                    "x": 100,
                    "y": 150,
                    "width": 200,
                    "height": 250
                }
            }
        }


class RegisterCamperRequest(BaseModel):
    """Request for camper registration"""
    camper_id: str = Field(..., min_length=1, max_length=100, description="Unique camper identifier (UUID)")
    preprocess: bool = Field(default=True, description="Whether to preprocess image")
    
    @field_validator('camper_id')
    @classmethod
    def validate_camper_id(cls, v):
        """Validate camper ID format"""
        if not v.strip():
            raise ValueError('camper_id cannot be empty')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "camper_id": "550e8400-e29b-41d4-a716-446655440000",
                "preprocess": True
            }
        }


class RegisterCamperResponse(BaseModel):
    """Response for camper registration"""
    success: bool = Field(..., description="Registration success status")
    message: Optional[str] = Field(None, description="Success or error message")
    error: Optional[str] = Field(None, description="Error message if failed")
    camper_id: str = Field(..., description="Camper identifier (UUID)")
    embedding_shape: Optional[tuple] = Field(None, description="Shape of embedding vector")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Camper registered successfully",
                "camper_id": "550e8400-e29b-41d4-a716-446655440000",
                "embedding_shape": [512],
                "processing_time": 1.234
            }
        }


class CheckAttendanceRequest(BaseModel):
    """Request for attendance check via face recognition"""
    activity_schedule_id: str = Field(..., description="Activity schedule identifier (UUID)")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    preprocess: bool = Field(default=True, description="Whether to preprocess image")
    save_results: bool = Field(default=True, description="Whether to save results to session")
    
    class Config:
        json_schema_extra = {
            "example": {
                "activity_schedule_id": "660aed1b-05e7-4259-ae87-57b3c356d398",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "preprocess": True,
                "save_results": True
            }
        }


class CheckAttendanceResponse(BaseModel):
    """Response for attendance check via face recognition"""
    success: bool = Field(..., description="Recognition success status")
    message: str = Field(..., description="Result message")
    error: Optional[str] = Field(None, description="Error message if failed")
    session_id: str = Field(..., description="Session identifier")
    recognized_campers: List[RecognizedCamper] = Field(default_factory=list, description="List of recognized campers")
    total_faces_detected: int = Field(..., description="Total number of faces detected")
    total_recognized: Optional[int] = Field(None, description="Number of campers recognized")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: Optional[str] = Field(None, description="Recognition timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Recognized 2 camper(s)",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "recognized_campers": [
                    {
                        "camper_id": "550e8400-e29b-41d4-a716-446655440000",
                        "distance": 0.15,
                        "face_region": {
                            "x": 100,
                            "y": 150,
                            "width": 200,
                            "height": 250
                        }
                    }
                ],
                "total_faces_detected": 2,
                "total_recognized": 2,
                "processing_time": 0.856,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class DetectedFace(BaseModel):
    """Detected face information"""
    face_id: str = Field(..., description="Unique face identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    region: FaceRegion = Field(..., description="Face location")
    
    class Config:
        json_schema_extra = {
            "example": {
                "face_id": "face-12345",
                "confidence": 0.95,
                "region": {
                    "x": 100,
                    "y": 150,
                    "width": 200,
                    "height": 250
                }
            }
        }


class DetectFacesResponse(BaseModel):
    """Response for face detection"""
    success: bool = Field(..., description="Detection success status")
    message: Optional[str] = Field(None, description="Result message")
    error: Optional[str] = Field(None, description="Error message if failed")
    detected_faces: List[DetectedFace] = Field(default_factory=list, description="List of detected faces")
    total_faces: int = Field(..., description="Total number of faces detected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Detected 2 face(s)",
                "detected_faces": [
                    {
                        "face_id": "face-12345",
                        "confidence": 0.95,
                        "region": {
                            "x": 100,
                            "y": 150,
                            "width": 200,
                            "height": 250
                        }
                    }
                ],
                "total_faces": 2
            }
        }


class CacheStats(BaseModel):
    """Cache statistics"""
    total_cached: int = Field(..., description="Total number of cached embeddings")
    campers: List[str] = Field(default_factory=list, description="List of cached camper IDs")
    model: str = Field(..., description="Face recognition model")
    distance_metric: str = Field(..., description="Distance metric used")
    fps_limit: float = Field(..., description="FPS rate limit")
    threshold: float = Field(..., description="Recognition confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_cached": 50,
                "campers": ["550e8400-e29b-41d4-a716-446655440000", "660aed1b-05e7-4259-ae87-57b3c356d398"],
                "model": "Facenet512",
                "distance_metric": "cosine",
                "fps_limit": 1.0,
                "threshold": 0.6
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    model: str = Field(..., description="Face recognition model")
    cache_stats: Optional[CacheStats] = Field(None, description="Cache statistics")
    timestamp: str = Field(..., description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "Face Recognition API",
                "model": "Facenet512",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }
