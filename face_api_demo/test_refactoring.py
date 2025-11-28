"""
Test Script for Refactored Face Recognition API
Run basic tests to verify the refactoring
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("="*60)
    print("Testing Module Imports...")
    print("="*60)
    
    try:
        from config import settings, setup_logging
        print("✅ Config imports successful")
        print(f"   - Model: {settings.DEEPFACE_MODEL}")
        print(f"   - Threshold: {settings.CONFIDENCE_THRESHOLD}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from services import ImageProcessor, EmbeddingCache, FaceRecognitionService, SupabaseService
        print("✅ Services imports successful")
    except Exception as e:
        print(f"❌ Services import failed: {e}")
        return False
    
    try:
        from models.schemas import RegisterStudentResponse, RecognizeFacesResponse
        print("✅ Models imports successful")
    except Exception as e:
        print(f"❌ Models import failed: {e}")
        return False
    
    try:
        from utils import FileHandler, validate_image_file
        print("✅ Utils imports successful")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True


def test_service_initialization():
    """Test if services can be initialized"""
    print("\n" + "="*60)
    print("Testing Service Initialization...")
    print("="*60)
    
    try:
        from services import ImageProcessor
        img_processor = ImageProcessor()
        print(f"✅ ImageProcessor initialized")
        print(f"   - Resize: {img_processor.resize_width}x{img_processor.resize_height}")
        print(f"   - Quality: {img_processor.quality}")
    except Exception as e:
        print(f"❌ ImageProcessor init failed: {e}")
        return False
    
    try:
        from services import EmbeddingCache
        cache = EmbeddingCache(preload=False)
        print(f"✅ EmbeddingCache initialized")
        print(f"   - Model: {cache.model_name}")
        print(f"   - Distance: {cache.distance_metric}")
    except Exception as e:
        print(f"❌ EmbeddingCache init failed: {e}")
        return False
    
    try:
        from services import FaceRecognitionService
        face_service = FaceRecognitionService()
        print(f"✅ FaceRecognitionService initialized")
        print(f"   - FPS Limit: {face_service.fps_limit}")
    except Exception as e:
        print(f"❌ FaceRecognitionService init failed: {e}")
        return False
    
    try:
        from services import SupabaseService
        supabase_service = SupabaseService()
        print(f"✅ SupabaseService initialized")
        print(f"   - Enabled: {supabase_service.is_enabled()}")
        print(f"   - Bucket: {supabase_service.bucket}")
    except Exception as e:
        print(f"❌ SupabaseService init failed: {e}")
        return False
    
    return True


def test_directory_structure():
    """Test if all required directories exist"""
    print("\n" + "="*60)
    print("Testing Directory Structure...")
    print("="*60)
    
    from config import settings
    
    required_dirs = [
        settings.UPLOAD_FOLDER,
        settings.TEMP_FOLDER,
        settings.DATABASE_FOLDER,
        settings.SESSIONS_FOLDER,
        settings.EMBEDDINGS_FOLDER,
        settings.LOGS_FOLDER
    ]
    
    all_exist = True
    for directory in required_dirs:
        if directory.exists():
            print(f"✅ {directory.name}: {directory}")
        else:
            print(f"❌ {directory.name}: Missing!")
            all_exist = False
    
    return all_exist


def test_cache_stats():
    """Test cache statistics retrieval"""
    print("\n" + "="*60)
    print("Testing Cache Statistics...")
    print("="*60)
    
    try:
        from services import FaceRecognitionService
        face_service = FaceRecognitionService()
        stats = face_service.get_cache_statistics()
        
        print("✅ Cache stats retrieved:")
        print(f"   - Total Cached: {stats['total_cached']}")
        print(f"   - Model: {stats['model']}")
        print(f"   - Distance Metric: {stats['distance_metric']}")
        print(f"   - FPS Limit: {stats['fps_limit']}")
        print(f"   - Threshold: {stats['threshold']}")
        
        return True
    except Exception as e:
        print(f"❌ Cache stats test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 REFACTORING VALIDATION TESTS")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Service Initialization": test_service_initialization(),
        "Directory Structure": test_directory_structure(),
        "Cache Statistics": test_cache_stats()
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Refactoring successful!")
    else:
        print("⚠️  SOME TESTS FAILED! Please review errors above.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
