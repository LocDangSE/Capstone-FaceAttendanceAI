"""
Production-ready JWT Authentication Middleware for Flask.

This middleware validates JWT tokens issued by the .NET backend server.
It ensures cryptographic verification, claim validation, and strict security controls.

Security Features:
- Bearer token validation
- JWT signature verification (HMAC-SHA256 or RS256)
- Mandatory claim validation (iss, aud, exp, iat, nbf)
- Issuer whitelist enforcement
- Cloudflare-compatible headers
- Comprehensive error handling
"""

import jwt
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, List
from flask import request, jsonify, g

logger = logging.getLogger(__name__)


class JWTAuthMiddleware:
    """
    JWT Authentication Middleware for Flask.
    
    Validates Bearer tokens issued by trusted .NET backend server(s).
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        required_issuer: str = "SummerCampBackend",
        required_audience: str = "face-recognition-api",
        public_key: Optional[str] = None,
        issuer_whitelist: Optional[List[str]] = None,
        clock_skew_seconds: int = 60,
        enable_cloudflare_headers: bool = True
    ):
        """
        Initialize JWT authentication middleware.
        
        Args:
            secret_key: Shared secret for HMAC algorithms or private key verification
            algorithm: JWT algorithm (HS256, HS512, RS256, RS512, ES256, etc.)
            required_issuer: Expected issuer claim (must match backend identity)
            required_audience: Expected audience claim (this service's identifier)
            public_key: RSA/ECDSA public key for asymmetric algorithms (optional)
            issuer_whitelist: List of allowed issuers (defaults to [required_issuer])
            clock_skew_seconds: Allowed time drift for exp/nbf validation (default 60s)
            enable_cloudflare_headers: Parse Cloudflare security headers (CF-Access-*)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.required_issuer = required_issuer
        self.required_audience = required_audience
        self.public_key = public_key
        self.issuer_whitelist = issuer_whitelist or [required_issuer]
        self.clock_skew_seconds = clock_skew_seconds
        self.enable_cloudflare_headers = enable_cloudflare_headers
        
        # Validate configuration
        if algorithm.startswith(('RS', 'ES', 'PS')) and not public_key:
            raise ValueError(f"Algorithm {algorithm} requires public_key parameter")
        
        if algorithm.startswith('HS') and not secret_key:
            raise ValueError(f"Algorithm {algorithm} requires secret_key parameter")
        
        logger.info(
            f"JWT Auth Middleware initialized: "
            f"algorithm={algorithm}, issuer={required_issuer}, audience={required_audience}"
        )
    
    def _extract_token_from_header(self) -> Optional[str]:
        """
        Extract Bearer token from Authorization header.
        
        Returns:
            JWT token string or None if not found/invalid format
        """
        auth_header = request.headers.get('Authorization', '').strip()
        
        if not auth_header:
            logger.debug("No Authorization header found")
            return None
        
        # Check Bearer scheme
        if not auth_header.lower().startswith('bearer '):
            logger.warning(f"Invalid authorization scheme: {auth_header[:20]}")
            return None
        
        # Extract token
        token = auth_header[7:].strip()  # Remove "Bearer " prefix
        
        if not token:
            logger.warning("Empty token after Bearer prefix")
            return None
        
        return token
    
    def _get_verification_key(self) -> str:
        """
        Get the appropriate key for JWT verification based on algorithm.
        
        Returns:
            Verification key (public key for asymmetric, secret for symmetric)
        """
        if self.algorithm.startswith(('RS', 'ES', 'PS')):
            if not self.public_key:
                raise ValueError(f"Public key required for {self.algorithm}")
            return self.public_key
        else:
            return self.secret_key
    
    def _validate_claims(self, payload: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate JWT claims according to security requirements.
        
        Args:
            payload: Decoded JWT payload
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Validate issuer (iss)
        iss = payload.get('iss')
        if not iss:
            return False, "Missing required claim: iss (issuer)"
        
        if iss not in self.issuer_whitelist:
            logger.warning(f"Unauthorized issuer: {iss} (not in whitelist)")
            return False, f"Unauthorized issuer: {iss}"
        
        # 2. Validate audience (aud)
        aud = payload.get('aud')
        if not aud:
            return False, "Missing required claim: aud (audience)"
        
        # Handle both string and list audience claims
        audience_list = [aud] if isinstance(aud, str) else aud
        if self.required_audience not in audience_list:
            logger.warning(
                f"Invalid audience: {aud} (expected: {self.required_audience})"
            )
            return False, f"Invalid audience: {aud}"
        
        # 3. Validate expiration (exp) - already checked by PyJWT, but log it
        exp = payload.get('exp')
        if exp:
            exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
            logger.debug(f"Token expires at: {exp_datetime.isoformat()}")
        
        # 4. Validate issued-at (iat) - ensure not from future
        iat = payload.get('iat')
        if iat:
            iat_datetime = datetime.fromtimestamp(iat, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            
            # Allow clock skew
            if iat_datetime > now + datetime.timedelta(seconds=self.clock_skew_seconds):
                logger.warning(f"Token issued in future: {iat_datetime.isoformat()}")
                return False, "Token issued in future (check system clock)"
        
        # 5. Validate not-before (nbf) if present
        nbf = payload.get('nbf')
        if nbf:
            nbf_datetime = datetime.fromtimestamp(nbf, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            
            # Token not yet valid
            if nbf_datetime > now + datetime.timedelta(seconds=self.clock_skew_seconds):
                logger.warning(f"Token not yet valid: {nbf_datetime.isoformat()}")
                return False, f"Token not yet valid (nbf: {nbf_datetime.isoformat()})"
        
        return True, None
    
    def _parse_cloudflare_headers(self) -> Dict[str, Any]:
        """
        Parse Cloudflare Access headers for additional security context.
        
        Returns:
            Dictionary of Cloudflare security metadata
        """
        if not self.enable_cloudflare_headers:
            return {}
        
        cf_headers = {}
        
        # Cloudflare Access JWT (if using Cloudflare Access)
        cf_access_jwt = request.headers.get('Cf-Access-Jwt-Assertion')
        if cf_access_jwt:
            cf_headers['cf_access_jwt'] = cf_access_jwt
        
        # Cloudflare Ray ID (for request tracing)
        cf_ray = request.headers.get('Cf-Ray')
        if cf_ray:
            cf_headers['cf_ray'] = cf_ray
        
        # Cloudflare Worker/Page request indicator
        cf_worker = request.headers.get('Cf-Worker')
        if cf_worker:
            cf_headers['cf_worker'] = cf_worker
        
        # Client IP (from Cloudflare)
        cf_connecting_ip = request.headers.get('Cf-Connecting-Ip')
        if cf_connecting_ip:
            cf_headers['cf_connecting_ip'] = cf_connecting_ip
        
        # Country code
        cf_ipcountry = request.headers.get('Cf-Ipcountry')
        if cf_ipcountry:
            cf_headers['cf_ipcountry'] = cf_ipcountry
        
        return cf_headers
    
    def verify_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Verify JWT token cryptographically and validate claims.
        
        Args:
            token: JWT token string
        
        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        try:
            # Decode and verify JWT signature + expiration
            verification_key = self._get_verification_key()
            
            payload = jwt.decode(
                token,
                verification_key,
                algorithms=[self.algorithm],
                audience=self.required_audience,
                options={
                    'require_exp': True,  # Require expiration
                    'require_iat': True,  # Require issued-at
                    'verify_signature': True,  # Strict signature verification
                    'verify_exp': True,  # Verify expiration
                    'verify_aud': True,  # Verify audience
                },
                leeway=self.clock_skew_seconds  # Clock skew tolerance
            )
            
            # Additional claim validation
            is_valid, error_msg = self._validate_claims(payload)
            if not is_valid:
                return False, None, error_msg
            
            # Parse Cloudflare headers if enabled
            cf_context = self._parse_cloudflare_headers()
            if cf_context:
                payload['_cloudflare'] = cf_context
                logger.debug(f"Cloudflare context: {cf_context}")
            
            logger.info(
                f"Token verified successfully: "
                f"sub={payload.get('sub')}, iss={payload.get('iss')}"
            )
            
            return True, payload, None
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return False, None, "Token has expired"
        
        except jwt.InvalidAudienceError as e:
            logger.warning(f"Invalid audience: {e}")
            return False, None, "Invalid token audience"
        
        except jwt.InvalidIssuerError as e:
            logger.warning(f"Invalid issuer: {e}")
            return False, None, "Invalid token issuer"
        
        except jwt.InvalidSignatureError:
            logger.error("Invalid token signature (possible forgery attempt)")
            return False, None, "Invalid token signature"
        
        except jwt.DecodeError as e:
            logger.error(f"Token decode error: {e}")
            return False, None, "Malformed token"
        
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return False, None, f"Invalid token: {str(e)}"
        
        except Exception as e:
            logger.error(f"Unexpected error during token verification: {e}", exc_info=True)
            return False, None, "Internal authentication error"
    
    def authenticate_request(self) -> tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Authenticate the current Flask request.
        
        Returns:
            Tuple of (is_authenticated, user_payload, error_response)
        """
        # Extract token
        token = self._extract_token_from_header()
        if not token:
            error_response = {
                'success': False,
                'error': 'Unauthorized',
                'message': 'Missing or invalid Authorization header. Expected: Bearer <token>',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return False, None, error_response
        
        # Verify token
        is_valid, payload, error_msg = self.verify_token(token)
        
        if not is_valid:
            error_response = {
                'success': False,
                'error': 'Unauthorized',
                'message': error_msg or 'Invalid token',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return False, None, error_response
        
        return True, payload, None


# Global middleware instance (initialized in app factory)
_jwt_middleware: Optional[JWTAuthMiddleware] = None


def init_jwt_middleware(
    app,
    secret_key: Optional[str] = None,
    algorithm: str = "HS256",
    **kwargs
) -> JWTAuthMiddleware:
    """
    Initialize JWT middleware for Flask app.
    
    Args:
        app: Flask application instance
        secret_key: JWT secret key (defaults to app.config['JWT_SECRET_KEY'])
        algorithm: JWT algorithm
        **kwargs: Additional JWTAuthMiddleware parameters
    
    Returns:
        Initialized JWTAuthMiddleware instance
    """
    global _jwt_middleware
    
    # Get secret from config if not provided
    if secret_key is None:
        secret_key = app.config.get('JWT_SECRET_KEY')
        if not secret_key:
            raise ValueError(
                "JWT_SECRET_KEY must be set in app.config or passed as parameter"
            )
    
    # Get other config values
    config_defaults = {
        'required_issuer': app.config.get('JWT_ISSUER', 'SummerCampBackend'),
        'required_audience': app.config.get('JWT_AUDIENCE', 'face-recognition-api'),
        'public_key': app.config.get('JWT_PUBLIC_KEY'),
        'issuer_whitelist': app.config.get('JWT_ISSUER_WHITELIST'),
        'clock_skew_seconds': app.config.get('JWT_CLOCK_SKEW_SECONDS', 60),
        'enable_cloudflare_headers': app.config.get('JWT_ENABLE_CLOUDFLARE_HEADERS', True),
    }
    
    # Merge with provided kwargs (kwargs take precedence)
    config_defaults.update(kwargs)
    
    _jwt_middleware = JWTAuthMiddleware(
        secret_key=secret_key,
        algorithm=algorithm,
        **config_defaults
    )
    
    logger.info("JWT middleware initialized for Flask app")
    return _jwt_middleware


def require_auth(f):
    """
    Decorator to protect Flask routes with JWT authentication.
    
    Usage:
        @app.route('/protected')
        @require_auth
        def protected_route():
            user_info = g.user  # Access authenticated user payload
            return jsonify({'message': 'Success'})
    
    Returns:
        401 response if authentication fails, otherwise calls wrapped function
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if _jwt_middleware is None:
            logger.error("JWT middleware not initialized. Call init_jwt_middleware() first.")
            return jsonify({
                'success': False,
                'error': 'Internal Server Error',
                'message': 'Authentication system not configured',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 500
        
        # Authenticate request
        is_authenticated, payload, error_response = _jwt_middleware.authenticate_request()
        
        if not is_authenticated:
            return jsonify(error_response), 401
        
        # Store user payload in Flask's g object for access in route handlers
        g.user = payload
        g.user_id = payload.get('sub')
        g.issuer = payload.get('iss')
        
        # Call the original function
        return f(*args, **kwargs)
    
    return decorated_function


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Get the authenticated user payload for the current request.
    
    Returns:
        User payload dictionary or None if not authenticated
    """
    return getattr(g, 'user', None)
