"""Test script for all three data source services.
"""
import requests
import json
import sys
import time

def test_all_services():
    """Test all three data source services."""
    services = [
        {"name": "data-source-0", "port": 8001},
        {"name": "data-source-20", "port": 8002},
        {"name": "data-source-100", "port": 8003}
    ]
    
    query = "What is machine learning?" if len(sys.argv) < 2 else sys.argv[1]
    
    print("Testing all data source services...")
    print("=" * 60)
    
    for service in services:
        base_url = f"http://localhost:{service['port']}"
        print(f"\n{service['name']} (port {service['port']}):")
        print("-" * 60)
        
        try:
            # Health check
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ Health: {health_data['status']} - Dataset: {health_data['dataset']}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
                continue
            
            # Query test (basic, without selected_sources)
            payload = {"query": query, "k": 3}
            response = requests.post(f"{base_url}/query", json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Query successful: Found {len(data['results'])} results")
                if data['results']:
                    top_result = data['results'][0]
                    print(f"  Top result: ID={top_result['id']}, Score={top_result['score']:.4f}")
                    print(f"  Text preview: {top_result['text'][:100]}...")
                if 'signature' in data:
                    print(f"  Signature: {data['signature'][:20]}...")
            else:
                print(f"✗ Query failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"  Error: {response.text[:200]}")
            
            # Test with selected_sources (optional - may fail if contract not configured)
            print(f"  Testing with selected_sources...")
            dataset_name = health_data.get('dataset', 'sources_0')
            payload_with_sources = {
                "query": query,
                "k": 3,
                "selected_sources": {
                    dataset_name: [10000, 10000]  # Use list format for JSON
                }
            }
            try:
                response = requests.post(f"{base_url}/query", json=payload_with_sources, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'signature' in data:
                        print(f"✓ Query with selected_sources successful: Signature present")
                        print(f"  Signature: {data['signature'][:50]}...")
                    else:
                        print(f"⚠ Query succeeded but no signature returned")
                elif response.status_code == 400:
                    error_data = response.json()
                    print(f"⚠ selected_sources validation failed: {error_data.get('error', 'Unknown')[:80]}")
                elif response.status_code == 500:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Unknown')
                    if 'contract' in error_msg.lower() or 'not initialized' in error_msg.lower():
                        print(f"ℹ Contract client not available (expected if contract not configured)")
                    else:
                        print(f"✗ Query with selected_sources failed: {error_msg[:80]}")
                else:
                    print(f"✗ Query with selected_sources failed: {response.status_code}")
            except Exception as e:
                print(f"⚠ Could not test selected_sources: {str(e)[:80]}")
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Could not connect to {base_url}")
        except requests.exceptions.Timeout:
            print(f"✗ Request timeout for {base_url}")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")


if __name__ == "__main__":
    test_all_services()
