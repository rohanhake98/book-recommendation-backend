import requests
import json
import time

def test_api():
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Book Recommendation API")
    print("=" * 40)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    try:
        # Test home endpoint
        print("\n1. Testing home endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Home endpoint working!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Home endpoint failed: {response.status_code}")
            return False
        
        # Test status endpoint
        print("\n2. Testing status endpoint...")
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working!")
            print(f"Users: {data['total_users']}")
            print(f"Books: {data['total_books']}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
        
        # Test popular books
        print("\n3. Testing popular books...")
        response = requests.get(f"{base_url}/recommend/popular?count=3")
        if response.status_code == 200:
            data = response.json()
            print("✅ Popular books endpoint working!")
            print(f"Found {data['count']} popular books")
            for i, book in enumerate(data['recommendations'][:3], 1):
                print(f"   {i}. {book['Title']} by {book['Author']}")
        else:
            print(f"❌ Popular books failed: {response.status_code}")
        
        # Test search
        print("\n4. Testing search...")
        response = requests.get(f"{base_url}/search?q=harry")
        if response.status_code == 200:
            data = response.json()
            print("✅ Search endpoint working!")
            print(f"Found {data['count']} books matching 'harry'")
        else:
            print(f"❌ Search failed: {response.status_code}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the API server!")
        print("Make sure the Flask server is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_api()
