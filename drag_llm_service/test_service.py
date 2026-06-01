"""Test script for the LLM service.
"""
import requests
import json
import time

def test_health_check(base_url="http://localhost:9000"):
    """Test basic health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        print()
        return False


def test_health_check_data_sources(base_url="http://localhost:9000"):
    """Test data sources health check endpoint."""
    print("Testing health check for data sources...")
    try:
        response = requests.get(f"{base_url}/health/data_sources", timeout=10)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Overall status: {data.get('status', 'N/A')}")
        print(f"Data sources status:")
        for source_name, source_status in data.get('data_sources', {}).items():
            print(f"  - {source_name}: {source_status.get('status', 'N/A')} "
                  f"({source_status.get('message', 'N/A')})")
        print()
        return response.status_code in [200, 503]  # 503 is acceptable if degraded
    except Exception as e:
        print(f"Data sources health check failed: {str(e)}")
        print()
        return False


def test_query(base_url="http://localhost:9000", query="Who won the first Nobel Prize in Physics"):
    """Test query endpoint that returns just the response."""
    print(f"Testing query: '{query}'...")
    payload = {"query": query}
    try:
        response = requests.post(f"{base_url}/query", json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', 'N/A')}")
            print()
            return True
        else:
            print(f"Error: {response.text}")
            print()
            return False
    except Exception as e:
        print(f"Query failed: {str(e)}")
        print()
        return False


def test_query_analyze(base_url="http://localhost:9000", query="Who won the first Nobel Prize in Physics", ground_truth=[" Wilhelm Conrad Röntgen"]):
    """Test query_analyze endpoint that returns response with analysis."""
    print(f"Testing query_analyze: '{query}'...")
    payload = {
        "query": query,
        "ground_truth": ground_truth
    }
    try:
        response = requests.post(f"{base_url}/query_analyze", json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('response', 'N/A')}")
            print(f"Correctness: {data.get('correctness', 'N/A')}")
            print(f"Sampled sources: {data.get('sampled_sources', [])}")
            importance_scores = data.get('importance_score', [])
            if importance_scores:
                print(f"Importance scores: {importance_scores}")
            else:
                print("Importance scores: None")
            updated_scores = data.get('updated_scores')
            if updated_scores:
                print(f"Updated scores: {updated_scores}")
            print()
            return True
        else:
            print(f"Error: {response.text}")
            print()
            return False
    except Exception as e:
        print(f"Query analyze failed: {str(e)}")
        print()
        return False


def test_score_events(base_url="http://localhost:9000", source_id=None, source_address=None, from_block=None, to_block=None):
    """Test score_events endpoint that returns score update events from blockchain."""
    print(f"Testing score_events endpoint...")
    try:
        # Build query parameters
        params = {}
        if source_id:
            params['source_id'] = source_id
        if source_address:
            params['source_address'] = source_address
        if from_block:
            params['from_block'] = from_block
        if to_block:
            params['to_block'] = to_block
        
        response = requests.get(f"{base_url}/score_events", params=params, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"Retrieved {len(events)} score update event(s)")
            
            if events:
                print("Recent events:")
                for i, event in enumerate(events[-5:], 1):  # Show last 5 events
                    print(f"  Event {i}:")
                    print(f"    Source Name: {event.get('sourceName', 'N/A')}")
                    print(f"    Source ID: {event.get('sourceID', 'N/A')}")
                    print(f"    Source Address: {event.get('sourceAddress', 'N/A')}")
                    print(f"    Reliability Score: {event.get('reliabilityScore', 'N/A')}")
                    print(f"    Usefulness Score: {event.get('usefulnessScore', 'N/A')}")
                    print(f"    Block Number: {event.get('blockNumber', 'N/A')}")
                    print(f"    Transaction Hash: {event.get('transactionHash', 'N/A')}")
                    if event.get('info'):
                        info_preview = event.get('info', '')[:100]
                        print(f"    Info: {info_preview}...")
                    print()
            else:
                print("No events found")
                print()
            
            return True
        else:
            print(f"Error: {response.text}")
            print()
            return False
    except Exception as e:
        print(f"Score events test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_query_analyze_sse(base_url="http://localhost:9000", query="Who won the first Nobel Prize in Physics", ground_truth=[" Wilhelm Conrad Röntgen"], update_scores=False):
    """Test query_analyze endpoint with SSE streaming."""
    print(f"Testing query_analyze with SSE: '{query}'...")
    if update_scores:
        print("  (with score updates enabled)")
    payload = {
        "query": query,
        "ground_truth": ground_truth,
        "update_scores": update_scores
    }
    try:
        # Make request with stream parameter
        url = f"{base_url}/query_analyze?stream=true"
        response = requests.post(url, json=payload, stream=True, timeout=180)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            # Verify response is SSE
            content_type = response.headers.get('content-type', '')
            if 'text/event-stream' not in content_type:
                print(f"Warning: Expected text/event-stream, got {content_type}")
                print(f"Response headers: {dict(response.headers)}")
            
            print("Receiving SSE events...")
            print("-" * 60)
            
            # Parse SSE events manually using requests
            final_result = None
            score_update_events = []  # Track score update events
            current_event_type = None
            current_data = []
            
            # Parse SSE stream line by line
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    # Empty line indicates end of event - process accumulated data
                    if current_event_type and current_data:
                        data_str = '\n'.join(current_data)
                        try:
                            data = json.loads(data_str)
                            
                            # Print event based on type
                            if current_event_type == 'sampling_sources':
                                print(f"[sampling_sources] Sources: {data.get('sources', [])}")
                            elif current_event_type == 'querying_sources':
                                source = data.get('source', 'unknown')
                                status = data.get('status', 'unknown')
                                if status == 'success':
                                    print(f"[querying_sources] {source}: {status} ({data.get('results_count', 0)} results)")
                                else:
                                    print(f"[querying_sources] {source}: {status}")
                                    if 'error' in data:
                                        print(f"  Error: {data['error']}")
                            elif current_event_type == 'reranking':
                                print(f"[reranking] {data.get('status', 'unknown')} - Selected: {data.get('selected_count', 0)}")
                            elif current_event_type == 'mc_shap':
                                print(f"[mc_shap] {data.get('status', 'unknown')}")
                            elif current_event_type == 'mc_shap_baseline':
                                baseline = data.get('baseline', '')
                                print(f"[mc_shap_baseline] Baseline calculated: {baseline[:100]}...")
                            elif current_event_type == 'mc_shap_progress':
                                count = data.get('count', 0)
                                total = data.get('total', 0)
                                progress = data.get('progress', 0.0)
                                # Update progress inline (overwrite same line)
                                print(f"\r[mc_shap_progress] Processing: {count}/{total} ({progress*100:.1f}%)", end='', flush=True)
                                # Print newline when complete
                                if count >= total:
                                    print()  # Newline when progress reaches 100%
                            elif current_event_type == 'computing_importance':
                                print(f"[computing_importance] {data.get('status', 'unknown')} - Sources: {data.get('sources_count', 0)}")
                            elif current_event_type == 'updating_scores':
                                score_update_events.append(data)
                                status = data.get('status', 'unknown')
                                print(f"[updating_scores] {status}")
                                if 'tx_hash' in data:
                                    print(f"  ✓ Transaction hash: {data['tx_hash']}")
                                if 'error' in data:
                                    print(f"  ✗ Error: {data['error']}")
                                if 'reason' in data:
                                    print(f"  Reason: {data['reason']}")
                            elif current_event_type == 'final_result':
                                final_result = data
                                print(f"[final_result] Received final result")
                                print(f"  Response: {data.get('response', 'N/A')[:100]}...")
                                print(f"  Correctness: {data.get('correctness', 'N/A')}")
                                print(f"  Sampled sources: {data.get('sampled_sources', [])}")
                                if 'updated_scores' in data:
                                    print(f"  Updated scores: {data['updated_scores']}")
                            elif current_event_type == 'error':
                                print(f"[error] {data.get('error', 'Unknown error')}")
                                
                        except json.JSONDecodeError:
                            print(f"[parse_error] Failed to parse data: {data_str[:100] if data_str else 'empty'}")
                    
                    # Reset for next event
                    current_event_type = None
                    current_data = []
                    continue
                
                # Parse SSE line
                if line.startswith('event:'):
                    current_event_type = line[6:].strip()
                elif line.startswith('data:'):
                    data_line = line[5:].strip()
                    if data_line:
                        current_data.append(data_line)
                # Ignore other line types (id:, retry:, etc.)
            
            print("-" * 60)
            if final_result:
                success_msg = "SSE test completed successfully!"
                if update_scores:
                    # Verify score update was attempted/completed
                    if score_update_events:
                        print(f"  Score update events received: {len(score_update_events)}")
                        for i, event_data in enumerate(score_update_events, 1):
                            print(f"    Event {i}: {event_data.get('status', 'unknown')}")
                            if event_data.get('status') == 'completed':
                                print(f"      ✓ Score update transaction completed")
                            elif event_data.get('status') == 'error':
                                print(f"      ✗ Score update failed: {event_data.get('error', 'unknown error')}")
                            elif event_data.get('status') == 'skipped':
                                print(f"      ⊘ Score update skipped: {event_data.get('reason', 'unknown reason')}")
                    
                    if 'updated_scores' in final_result:
                        success_msg += " (Score updates included in final result)"
                        print(f"  Updated scores for sources: {list(final_result.get('updated_scores', {}).keys())}")
                        print(final_result)
                    else:
                        print("  Warning: update_scores was true but no updated_scores in result")
                print(success_msg)
                print()
                return True
            else:
                print("SSE test completed but no final result received")
                print()
                return False
        else:
            print(f"Error: {response.text}")
            print()
            return False
    except Exception as e:
        print(f"SSE test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:9000"
    
    print("Testing LLM Service")
    print("=" * 60)
    print()
    
    # Test basic health check
    if not test_health_check(base_url):
        print("Service is not running. Exiting.")
        sys.exit(1)
    
    # Test data sources health check
    # test_health_check_data_sources(base_url)
    
    # Test simple query
    # test_query(base_url)
    
    # Test query with analysis (non-streaming)
    # test_query_analyze(base_url)
    
    # Test query with analysis (SSE streaming)
    # test_query_analyze_sse(base_url)
    
    # Test query with analysis (SSE streaming with score updates)
    # print()
    test_query_analyze_sse(base_url, update_scores=True)
    
    # Test score events endpoint to query score update events
    print()
    test_score_events(base_url)
    
    print("Testing complete!")
