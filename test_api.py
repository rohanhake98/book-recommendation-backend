import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, description):
    """Test a single endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"   URL: {BASE_URL}{endpoint}")
    
    try:
        response = requests.get(f"{BASE_URL}{endpoint}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success!")
            
            # Print relevant information based on endpoint
            if 'recommendations' in data:
                print(f"   📚 Found {data.get('count', 0)} recommendations")
                if data['recommendations']:
                    first_rec = data['recommendations'][0]
                    print(f"   📖 First: {first_rec['title']} by {first_rec['author']}")
            elif 'results' in data:
                print(f"   🔍 Found {data.get('count', 0)} search results")
            elif 'ratings' in data:
                print(f"   ⭐ Found {data.get('total_ratings', 0)} ratings")
            else:
                print(f"   📄 Response keys: {list(data.keys())}")
        else:
            print(f"   ❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    print("🚀 TESTING BOOK RECOMMENDATION API")
    print("=" * 50)
    
    # Test basic endpoints
    test_endpoint("/", "Home endpoint")
    test_endpoint("/status", "Status endpoint")
    
    # Get a random user for testing
    try:
        response = requests.get(f"{BASE_URL}/random-user")
        if response.status_code == 200:
            random_user = response.json()['user_id']
            print(f"\n🎲 Using random user ID: {random_user}")
        else:
            random_user = 276725  # Fallback user
            print(f"\n👤 Using fallback user ID: {random_user}")
    except:
        random_user = 276725
        print(f"\n👤 Using fallback user ID: {random_user}")
    
    # Test recommendation endpoints
    test_endpoint(f"/recommend/user/{random_user}", "User recommendations")
    test_endpoint(f"/recommend/svd/{random_user}", "SVD recommendations")
    test_endpoint("/recommend/popular", "Popular books")
    
    # Test with a sample ISBN (you might need to adjust this)
    sample_isbn = "0195153448"  # This should exist in your dataset
    test_endpoint(f"/recommend/similar/{sample_isbn}", "Similar books")
    test_endpoint(f"/book/{sample_isbn}", "Book details")
    
    # Test other endpoints
    test_endpoint(f"/user/{random_user}/ratings", "User ratings")
    test_endpoint("/search?q=harry", "Search functionality")
    
    print(f"\n✅ API testing completed!")
    print(f"💡 Try opening http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
