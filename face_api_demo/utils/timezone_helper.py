"""
Timezone Utilities for UTC+7 (Indochina Time / Bangkok Time)
"""

from datetime import datetime, timezone, timedelta
import pytz

# Define UTC+7 timezone
ICT = pytz.timezone('Asia/Bangkok')  # UTC+7
UTC_OFFSET = timedelta(hours=7)


def get_now() -> datetime:
    """
    Get current date/time in UTC+7
    
    Returns:
        datetime: Current time in UTC+7 timezone
    """
    return datetime.now(ICT)


def to_local_time(utc_dt: datetime) -> datetime:
    """
    Convert UTC datetime to UTC+7
    
    Args:
        utc_dt: UTC datetime object
        
    Returns:
        datetime: Datetime in UTC+7 timezone
    """
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt)
    return utc_dt.astimezone(ICT)


def to_utc(local_dt: datetime) -> datetime:
    """
    Convert UTC+7 datetime to UTC
    
    Args:
        local_dt: Datetime in UTC+7 timezone
        
    Returns:
        datetime: UTC datetime object
    """
    if local_dt.tzinfo is None:
        local_dt = ICT.localize(local_dt)
    return local_dt.astimezone(pytz.utc)


def get_utc_now_for_storage() -> datetime:
    """
    Get current UTC time for database storage
    Returns UTC time but represents current ICT moment
    
    Returns:
        datetime: Current time in UTC (for database storage)
    """
    return datetime.now(pytz.utc)


def format_datetime(dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime in UTC+7 timezone
    
    Args:
        dt: Datetime object
        fmt: Format string
        
    Returns:
        str: Formatted datetime string
    """
    local_dt = to_local_time(dt) if dt.tzinfo != ICT else dt
    return local_dt.strftime(fmt)


class TimezoneInfo:
    """Timezone information constants"""
    NAME = 'Asia/Bangkok'
    DISPLAY_NAME = 'Indochina Time (ICT)'
    ABBREVIATION = 'ICT'
    OFFSET_HOURS = 7
    TIMEZONE = ICT
