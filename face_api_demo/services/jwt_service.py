"""
Service-to-Service JWT Token Generator
Generates JWT tokens for Python → .NET webhook authentication
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


def generate_service_jwt() -> str:
    """
    Generate JWT token for service-to-service authentication (Python → .NET)
    
    Returns:
        str: Encoded JWT token valid for 5 minutes
    """
    try:
        # Use dedicated service secret if configured, otherwise fall back to user JWT secret
        secret = settings.SERVICE_JWT_SECRET or settings.JWT_SECRET_KEY
        
        if not secret or secret == "dev-secret-CHANGE-IN-PRODUCTION":
            logger.warning("Using default JWT secret for service token - configure SERVICE_JWT_SECRET in production!")
        
        # Create JWT payload
        payload: Dict[str, Any] = {
            'iss': settings.SERVICE_JWT_ISSUER,  # "PythonAiService"
            'aud': settings.SERVICE_JWT_AUDIENCE,  # "SummerCampBackend"
            'sub': 'face-recognition-service',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(minutes=5),  # Short-lived token
            'service': True,  # Flag to identify this as a service token
            'scope': 'attendance:update'  # Permission scope
        }
        
        # Encode token
        token = jwt.encode(payload, secret, algorithm=settings.JWT_ALGORITHM)
        
        logger.debug(f"Generated service JWT: iss={payload['iss']}, exp={payload['exp']}")
        
        return token
    
    except Exception as e:
        logger.error(f"Failed to generate service JWT: {e}")
        raise


def validate_service_jwt(token: str) -> Dict[str, Any]:
    """
    Validate a service JWT token (for testing purposes)
    
    Args:
        token: JWT token string
        
    Returns:
        dict: Decoded token payload
        
    Raises:
        jwt.InvalidTokenError: If token is invalid
    """
    secret = settings.SERVICE_JWT_SECRET or settings.JWT_SECRET_KEY
    
    decoded = jwt.decode(
        token,
        secret,
        algorithms=[settings.JWT_ALGORITHM],
        issuer=settings.SERVICE_JWT_ISSUER,
        audience=settings.SERVICE_JWT_AUDIENCE
    )
    
    return decoded
