"""
Quick API Test Script
Tests the refactored Flask server endpoints
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check successful!")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Service: {data.get('service')}")
            print(f"   - Model: {data.get('model')}")
            
            if 'cache_stats' in data:
                stats = data['cache_stats']
                print(f"   - Cached Embeddings: {stats.get('total_cached', 0)}")
                print(f"   - FPS Limit: {stats.get('fps_limit', 'N/A')}")
                print(f"   - Threshold: {stats.get('threshold', 'N/A')}")
            
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running on port 5000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_cache_stats_endpoint():
    """Test cache stats endpoint"""
    print("\n" + "="*60)
    print("Testing /api/cache/stats endpoint...")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/api/cache/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Cache stats endpoint successful!")
            
            if 'cache_stats' in data:
                stats = data['cache_stats']
                print(f"   - Total Cached: {stats.get('total_cached', 0)}")
                print(f"   - Model: {stats.get('model')}")
                print(f"   - Distance Metric: {stats.get('distance_metric')}")
            
            return True
        else:
            print(f"❌ Cache stats failed with status: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n🧪 API ENDPOINT TESTS")
    print("="*60)
    print("Testing refactored Flask server...")
    print("="*60)
    
    results = {
        "Health Check": test_health_endpoint(),
        "Cache Stats": test_cache_stats_endpoint()
    }
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL API TESTS PASSED!")
        print("The refactored server is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("Make sure the server is running: python app.py")
    print("="*60)


if __name__ == "__main__":
    main()
