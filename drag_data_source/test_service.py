"""Test script for the data source service.
"""
import requests
import json

def test_health_check(base_url="http://localhost:8001"):
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_query(base_url="http://localhost:8001", query="What is artificial intelligence?", k=5):
    """Test query endpoint."""
    print(f"Testing query: '{query}'...")
    payload = {"query": query, "k": k}
    response = requests.post(f"{base_url}/query", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data['results'])} results:")
        for i, result in enumerate(data['results'], 1):
            print(f"\n{i}. {result['id']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:200]}...")
    else:
        print(f"Error: {response.text}")
    print()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8001"
    
    if len(sys.argv) > 2:
        query = sys.argv[2]
    else:
        query = "What is machine learning?"
    
    try:
        test_health_check(base_url)
        test_query(base_url, query)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {base_url}")
        print("Make sure the service is running.")
        sys.exit(1)
