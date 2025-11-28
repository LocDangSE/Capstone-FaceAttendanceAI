#!/usr/bin/env python3
"""
Quick Deployment Script for Optimized Face Recognition Service
Validates environment and applies optimizations
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False
    print("✅ Python version OK")
    return True

def install_requirements():
    print_header("Installing Requirements")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", "--upgrade"
        ])
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_faiss():
    print_header("Checking FAISS Installation")
    try:
        import faiss
        print(f"✅ FAISS available (version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'})")
        return True
    except ImportError:
        print("⚠️  FAISS not available")
        print("   Installing faiss-cpu...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            print("✅ FAISS installed")
            return True
        except:
            print("❌ Failed to install FAISS")
            print("   System will fall back to brute-force search")
            return False

def check_opencv():
    print_header("Checking OpenCV Installation")
    try:
        import cv2
        print(f"✅ OpenCV available (version: {cv2.__version__})")
        return True
    except ImportError:
        print("❌ OpenCV not available")
        return False

def validate_env_file():
    print_header("Validating .env Configuration")
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_vars = [
        'CONFIDENCE_THRESHOLD',
        'USE_OPTIMIZED_CACHE',
        'USE_OPTIMIZED_PROCESSOR',
        'DEEPFACE_MODEL',
        'SUPABASE_URL'
    ]
    
    missing = []
    for var in required_vars:
        if var not in content:
            missing.append(var)
    
    if missing:
        print(f"❌ Missing variables: {', '.join(missing)}")
        return False
    
    print("✅ .env file valid")
    
    # Display key settings
    print("\nKey Configuration:")
    for line in content.split('\n'):
        if any(var in line for var in required_vars) and not line.startswith('#'):
            print(f"  {line}")
    
    return True

def test_imports():
    print_header("Testing Imports")
    
    modules = [
        ('flask', 'Flask'),
        ('deepface', 'DeepFace'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV')
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            all_ok = False
    
    return all_ok

def validate_services():
    print_header("Validating Services")
    
    required_files = [
        'services/embedding_cache.py',
        'services/embedding_cache_optimized.py',
        'services/image_processor.py',
        'services/image_processor_optimized.py',
        'services/face_recognition_service.py',
        'services/supabase_service.py',
        'config/settings.py',
        'app_refactored.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_configuration():
    print_header("Testing Configuration Load")
    try:
        from config.settings import settings
        
        print(f"✅ Configuration loaded")
        print(f"\nOptimization Status:")
        print(f"  FAISS Cache: {'✅ Enabled' if settings.USE_OPTIMIZED_CACHE else '❌ Disabled'}")
        print(f"  Optimized Processor: {'✅ Enabled' if settings.USE_OPTIMIZED_PROCESSOR else '❌ Disabled'}")
        print(f"  Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
        print(f"  Model: {settings.DEEPFACE_MODEL}")
        print(f"  Detector: {settings.DEEPFACE_DETECTOR}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def main():
    print("\n" + "🚀" * 30)
    print(" Face Recognition Service - Deployment Validator")
    print("🚀" * 30)
    
    checks = [
        ("Python Version", check_python_version),
        ("Requirements", install_requirements),
        ("FAISS", check_faiss),
        ("OpenCV", check_opencv),
        (".env File", validate_env_file),
        ("Imports", test_imports),
        ("Service Files", validate_services),
        ("Configuration", test_configuration)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("Deployment Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System ready for deployment.")
        print("\nNext steps:")
        print("  1. Start Python service: python app_refactored.py")
        print("  2. Start .NET service: dotnet run")
        print("  3. Test recognition: curl http://localhost:5000/health")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
