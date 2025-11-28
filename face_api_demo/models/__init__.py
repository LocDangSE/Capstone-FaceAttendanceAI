"""Data models package"""

from .schemas import (
    RegisterCamperRequest,
    RegisterCamperResponse,
    CheckAttendanceRequest,
    CheckAttendanceResponse,
    RecognizedCamper,
    FaceRegion
)

__all__ = [
    'RegisterCamperRequest',
    'RegisterCamperResponse',
    'CheckAttendanceRequest',
    'CheckAttendanceResponse',
    'RecognizedCamper',
    'FaceRegion'
]
