"""
Dead-Letter Queue Service
Logs failed webhook attempts for manual review and replay
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from config.settings import settings

logger = logging.getLogger(__name__)


def log_failed_webhook(request_id: str, payload: Dict[str, Any], error_message: str = "") -> None:
    """
    Log a failed webhook attempt to the dead-letter queue
    
    Args:
        request_id: Unique request identifier
        payload: Original webhook payload that failed
        error_message: Optional error description
    """
    try:
        # Create daily DLQ file
        dlq_file = settings.LOGS_FOLDER / f"dlq_{date.today().isoformat()}.jsonl"
        
        # Prepare DLQ entry
        dlq_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'request_id': request_id,
            'error': error_message,
            'payload': payload,
            'retry_count': settings.DOTNET_WEBHOOK_RETRY_COUNT,
            'status': 'failed'
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(dlq_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dlq_entry) + '\n')
        
        logger.error(
            f"[{request_id}] ❌ Logged to dead-letter queue: {dlq_file.name}"
        )
    
    except Exception as e:
        logger.critical(f"[{request_id}] Failed to write to DLQ: {e}")


def get_failed_webhooks(days_back: int = 1) -> List[Dict[str, Any]]:
    """
    Retrieve failed webhook entries from DLQ for manual replay
    
    Args:
        days_back: Number of days to look back (default: 1 day)
        
    Returns:
        List of DLQ entries
    """
    entries = []
    
    try:
        # Scan DLQ files from past N days
        for i in range(days_back):
            target_date = date.today() - timedelta(days=i)
            dlq_file = settings.LOGS_FOLDER / f"dlq_{target_date.isoformat()}.jsonl"
            
            if dlq_file.exists():
                with open(dlq_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            entries.append(entry)
        
        logger.info(f"Retrieved {len(entries)} failed webhooks from past {days_back} day(s)")
        return entries
    
    except Exception as e:
        logger.error(f"Failed to read DLQ entries: {e}")
        return []


def check_dlq_size() -> Dict[str, Any]:
    """
    Check total size of DLQ files and raise alert if too large
    
    Returns:
        dict: Statistics about DLQ size
    """
    try:
        dlq_files = list(settings.LOGS_FOLDER.glob("dlq_*.jsonl"))
        total_size = sum(f.stat().st_size for f in dlq_files)
        total_entries = 0
        
        for dlq_file in dlq_files:
            with open(dlq_file, 'r') as f:
                total_entries += sum(1 for _ in f)
        
        size_mb = total_size / (1024 * 1024)
        
        stats = {
            'total_size_mb': round(size_mb, 2),
            'total_files': len(dlq_files),
            'total_entries': total_entries,
            'alert': size_mb > 100  # Alert if >100MB
        }
        
        if stats['alert']:
            logger.critical(
                f"⚠️ DLQ size exceeded 100MB: {size_mb:.2f}MB across {len(dlq_files)} files"
            )
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to check DLQ size: {e}")
        return {'error': str(e)}


def cleanup_old_dlq_files(keep_days: int = 7) -> int:
    """
    Delete DLQ files older than N days to save disk space
    
    Args:
        keep_days: Number of days to retain (default: 7 days)
        
    Returns:
        int: Number of files deleted
    """
    deleted_count = 0
    
    try:
        cutoff_date = date.today() - timedelta(days=keep_days)
        
        for dlq_file in settings.LOGS_FOLDER.glob("dlq_*.jsonl"):
            try:
                # Extract date from filename: dlq_2024-12-08.jsonl
                file_date_str = dlq_file.stem.replace("dlq_", "")
                file_date = date.fromisoformat(file_date_str)
                
                if file_date < cutoff_date:
                    dlq_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old DLQ file: {dlq_file.name}")
            
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to process DLQ file {dlq_file.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup: Deleted {deleted_count} DLQ files older than {keep_days} days")
        
        return deleted_count
    
    except Exception as e:
        logger.error(f"Failed to cleanup DLQ files: {e}")
        return 0


def replay_webhook(entry: Dict[str, Any]) -> bool:
    """
    Manually replay a failed webhook from DLQ
    
    Args:
        entry: DLQ entry containing original payload
        
    Returns:
        bool: True if replay succeeded, False otherwise
    """
    try:
        from services.webhook_service import send_webhook_sync
        
        request_id = entry['request_id']
        payload = entry['payload']
        
        logger.info(f"[{request_id}] Replaying failed webhook from DLQ...")
        
        success = send_webhook_sync(request_id, payload)
        
        if success:
            logger.info(f"[{request_id}] ✅ Webhook replay succeeded")
        else:
            logger.error(f"[{request_id}] ❌ Webhook replay failed")
        
        return success
    
    except Exception as e:
        logger.error(f"Webhook replay error: {e}")
        return False
