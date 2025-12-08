"""
Webhook Service for .NET Backend Integration
Handles asynchronous attendance updates via webhook callbacks
"""

import json
import logging
import time
import traceback
import requests
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from config.settings import settings
from services.jwt_service import generate_service_jwt
from services.dead_letter_queue import log_failed_webhook

logger = logging.getLogger(__name__)


def update_attendance_async(
    request_id: str,
    activity_schedule_id: int,
    group_id: int,
    camp_id: int,
    results: List[Dict[str, Any]],
    user_id: str,
    username: str
) -> None:
    """
    Background task: Update .NET attendance logs asynchronously
    This function runs in a separate thread so mobile gets results faster
    
    Args:
        request_id: Unique correlation ID for this recognition request
        activity_schedule_id: Activity schedule being tracked
        group_id: Camper group ID
        camp_id: Camp ID
        results: List of recognition results from face_service
        user_id: User ID from JWT token
        username: Username from JWT token
    """
    webhook_start = time.time()
    
    try:
        logger.info(f"[{request_id}] 🔄 [Async] Starting attendance update webhook...")
        
        # Generate service-to-service JWT token
        service_token = generate_service_jwt()
        
        # Filter only recognized faces (camper_id > 0)
        # Handle both string and int camper_ids
        recognized_faces = []
        for r in results:
            camper_id = r.get('camper_id', -1)
            try:
                camper_id_int = int(camper_id) if camper_id not in [None, -1, '-1'] else -1
                if camper_id_int > 0:
                    recognized_faces.append(r)
            except (ValueError, TypeError):
                logger.warning(f"[{request_id}] Invalid camper_id: {camper_id}")
                continue
        
        if not recognized_faces:
            logger.warning(f"[{request_id}] ⚠️ [Async] No recognized faces to update, skipping webhook")
            return
        
        # Prepare webhook payload
        payload = {
            "requestId": request_id,
            "activityScheduleId": activity_schedule_id,
            "groupId": group_id,
            "campId": camp_id,
            "recognizedFaces": [
                {
                    "embedding": r.get('embedding', []).tolist() if hasattr(r.get('embedding', []), 'tolist') else (r.get('embedding', []) if r.get('embedding') else []),
                    "confidence": float(1.0 - r.get('distance', 0.0)),  # Convert distance to confidence
                    "boundingBox": {
                        "x": r['face_region']['x'],
                        "y": r['face_region']['y'],
                        "width": r['face_region']['width'],
                        "height": r['face_region']['height']
                    } if r.get('face_region') else None,
                    "faceArea": r['face_region']['width'] * r['face_region']['height'] if r.get('face_region') else 0,
                    "camperId": int(r.get('camper_id', -1))  # Ensure integer
                }
                for r in recognized_faces
            ],
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "processedBy": username,
                "userId": user_id,
                "source": "mobile-direct",
                "pythonVersion": "1.0.0"
            }
        }
        
        # Retry logic with exponential backoff
        max_retries = settings.DOTNET_WEBHOOK_RETRY_COUNT
        timeout = settings.DOTNET_WEBHOOK_TIMEOUT
        webhook_url = f"{settings.DOTNET_API_URL}/api/attendance/update-from-recognition"
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"[{request_id}] [Async] Attempt {attempt}/{max_retries}: POST {webhook_url}"
                )
                
                # Skip SSL verification for localhost (development only)
                verify_ssl = not webhook_url.startswith(('https://localhost', 'https://127.0.0.1'))
                
                response = requests.post(
                    webhook_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {service_token}",
                        "Content-Type": "application/json",
                        "X-Request-ID": request_id,
                        "X-Service-Name": "PythonFaceRecognitionAPI"
                    },
                    timeout=timeout,
                    verify=verify_ssl  # Skip SSL verification for localhost
                )
                
                webhook_time = time.time() - webhook_start
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(
                        f"[{request_id}] ✅ [Async] Webhook succeeded in {webhook_time:.2f}s "
                        f"(attempt {attempt}, updated: {result.get('updatedCount', 0)}, "
                        f"created: {result.get('createdCount', 0)})"
                    )
                    return  # Success - exit function
                
                else:
                    logger.warning(
                        f"[{request_id}] ⚠️ [Async] Attempt {attempt}/{max_retries} failed: "
                        f"HTTP {response.status_code} - {response.text[:200]}"
                    )
                    
                    if attempt < max_retries:
                        # Exponential backoff: 1s, 2s, 4s
                        backoff_delay = 2 ** (attempt - 1)
                        logger.debug(f"[{request_id}] Waiting {backoff_delay}s before retry...")
                        time.sleep(backoff_delay)
            
            except requests.Timeout:
                logger.warning(
                    f"[{request_id}] ⚠️ [Async] Attempt {attempt}/{max_retries} timed out after {timeout}s"
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
            
            except requests.ConnectionError as e:
                logger.warning(
                    f"[{request_id}] ⚠️ [Async] Attempt {attempt}/{max_retries} connection error: {e}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
            
            except Exception as e:
                logger.error(
                    f"[{request_id}] ❌ [Async] Attempt {attempt}/{max_retries} unexpected error: {e}"
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))
        
        # All retries failed - log to dead-letter queue
        webhook_time = time.time() - webhook_start
        error_msg = f"Failed after {max_retries} attempts in {webhook_time:.2f}s"
        logger.error(f"[{request_id}] ❌ [Async] {error_msg}")
        log_failed_webhook(request_id, payload, error_msg)
    
    except Exception as e:
        webhook_time = time.time() - webhook_start
        logger.error(f"[{request_id}] ❌ [Async] Critical webhook error after {webhook_time:.2f}s: {e}")
        logger.error(traceback.format_exc())
        
        # Log to DLQ if payload was prepared
        if 'payload' in locals():
            log_failed_webhook(request_id, payload, str(e))


def send_webhook_sync(request_id: str, payload: Dict[str, Any]) -> bool:
    """
    Synchronous webhook send (used for DLQ replay)
    
    Args:
        request_id: Request correlation ID
        payload: Webhook payload
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        service_token = generate_service_jwt()
        webhook_url = f"{settings.DOTNET_API_URL}/api/attendance/update-from-recognition"
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {service_token}",
                "Content-Type": "application/json",
                "X-Request-ID": request_id
            },
            timeout=settings.DOTNET_WEBHOOK_TIMEOUT
        )
        
        return response.status_code == 200
    
    except Exception as e:
        logger.error(f"[{request_id}] Sync webhook error: {e}")
        return False


def start_webhook_thread(
    request_id: str,
    activity_schedule_id: int,
    group_id: int,
    camp_id: int,
    results: List[Dict[str, Any]],
    user_id: str,
    username: str
) -> threading.Thread:
    """
    Start webhook update in background thread (non-blocking)
    
    Returns:
        threading.Thread: The started thread (for testing/debugging)
    """
    thread = threading.Thread(
        target=update_attendance_async,
        args=(
            request_id,
            activity_schedule_id,
            group_id,
            camp_id,
            results,
            user_id,
            username
        ),
        daemon=True,
        name=f"WebhookThread-{request_id}"
    )
    
    thread.start()
    logger.debug(f"[{request_id}] Started webhook thread: {thread.name}")
    
    return thread
