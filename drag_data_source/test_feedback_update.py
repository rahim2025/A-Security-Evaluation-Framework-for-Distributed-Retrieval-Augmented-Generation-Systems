"""Test script for feedback_and_update_score_records functionality.
Queries all 3 data source services with selected_sources, collects signatures,
and calls feedback_and_update_score_records to update scores.
"""
import requests
import json
import sys
import os
from typing import Dict, List

# Add drag_python_client to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'drag_python_client')))

from drag_python_client import DragScoresClient, sign_message_personal

# Hardhat default accounts
HARDHAT_ACCOUNTS = {
    1: {
        "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
        "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    },
    2: {
        "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
    },
    3: {
        "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
    },
    4: {
        "address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
        "private_key": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
    }
}

# Default data sources
DEFAULT_SOURCES = ["sources_0", "sources_20", "sources_100"]
DEFAULT_PORTS = [8001, 8002, 8003]


def get_current_scores(client: DragScoresClient, source_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """Get current scores from the contract."""
    returned_ids, reliability_scores, usefulness_scores = client.get_scores_batch(source_ids)
    
    scores = {}
    for i, source_id in enumerate(returned_ids):
        if i < len(reliability_scores) and i < len(usefulness_scores):
            scores[source_id] = {
                "reliability": reliability_scores[i],
                "usefulness": usefulness_scores[i]
            }
    
    return scores


def query_data_sources(query: str, selected_sources: Dict[str, List[int]]) -> Dict[str, str]:
    """Query all 3 data source services and collect signatures."""
    signatures = {}
    
    print("Querying data source services...")
    print("-" * 60)
    
    for i, source_name in enumerate(DEFAULT_SOURCES):
        port = DEFAULT_PORTS[i]
        base_url = f"http://localhost:{port}"
        
        print(f"\n{source_name} (port {port}):")
        
        payload = {
            "query": query,
            "k": 3,
            "selected_sources": selected_sources
        }
        
        try:
            response = requests.post(f"{base_url}/query", json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'signature' in data:
                    signatures[source_name] = data['signature']
                    print(f"✓ Query successful: Signature received")
                    print(f"  Signature: {data['signature'][:50]}...")
                    print(f"  Results: {len(data.get('results', []))} documents")
                else:
                    print(f"⚠ Query succeeded but no signature returned")
                    return None
            else:
                error_data = response.json() if response.content else {}
                print(f"✗ Query failed: {response.status_code}")
                print(f"  Error: {error_data.get('error', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"✗ Error querying {source_name}: {str(e)}")
            return None
    
    return signatures


def test_feedback_and_update():
    """Test feedback_and_update_score_records functionality."""
    print("=" * 60)
    print("Testing feedback_and_update_score_records")
    print("=" * 60)
    
    # Configuration
    provider_url = "http://localhost:8545"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    query_text = "What is machine learning?" if len(sys.argv) < 2 else sys.argv[1]
    
    # Initialize contract client (using account #4 for the caller)
    caller_account = HARDHAT_ACCOUNTS[4]
    print(f"\nInitializing contract client...")
    print(f"Caller account: {caller_account['address']}")
    
    try:
        client = DragScoresClient(
            project_root=project_root,
            provider_url=provider_url,
            contract_address=None  # Will auto-detect from Hardhat Ignition
        )
        print(f"✓ Contract connected: {client.hello()}")
    except Exception as e:
        print(f"✗ Failed to initialize contract client: {str(e)}")
        print("Make sure Hardhat node is running on localhost:8545")
        return
    
    # Get current scores
    print(f"\nFetching current scores...")
    current_scores = get_current_scores(client, DEFAULT_SOURCES)
    
    if not current_scores:
        print("✗ Could not fetch current scores. Make sure scores exist in the contract.")
        return
    
    print("Current scores:")
    for source_id, scores in current_scores.items():
        print(f"  {source_id}: reliability={scores['reliability']}, usefulness={scores['usefulness']}")
    
    # Prepare selected_sources dict with current scores
    selected_sources = {
        source_id: [scores['usefulness'], scores['reliability']]
        for source_id, scores in current_scores.items()
    }
    
    print(f"\nPrepared selected_sources:")
    for source_id, scores in selected_sources.items():
        print(f"  {source_id}: [{scores[0]}, {scores[1]}]")
    
    # Query all data sources and collect signatures
    signatures = query_data_sources(query_text, selected_sources)
    
    if not signatures or len(signatures) != 3:
        print("\n✗ Failed to collect all signatures")
        return
    
    print(f"\n✓ Collected {len(signatures)} signatures")
    
    # Construct the message (same format as data sources use)
    message_dict = {
        "query": query_text,
        "selected_sources": selected_sources
    }
    message_json = json.dumps(message_dict, sort_keys=True)
    
    print(f"\nMessage to sign:")
    print(f"  {message_json[:100]}...")
    
    # Sign the message with the caller account (#4)
    print(f"\nSigning message with caller account...")
    try:
        signature_bytes = sign_message_personal(message_json, caller_account['private_key'])
        caller_signature = "0x" + signature_bytes.hex()
        print(f"✓ Caller signature: {caller_signature[:50]}...")
    except Exception as e:
        print(f"✗ Failed to sign message: {str(e)}")
        return
    
    # Calculate updated scores (add 10 to each) - must be done before preparing signatures
    update_source_ids = []
    update_reliability_scores = []
    update_usefulness_scores = []
    
    for source_id in DEFAULT_SOURCES:
        if source_id in current_scores:
            update_source_ids.append(source_id)
            update_reliability_scores.append(current_scores[source_id]['reliability'] + 10)
            update_usefulness_scores.append(current_scores[source_id]['usefulness'] + 10)
    
    # Prepare signatures list (convert hex strings to bytes)
    # The contract expects one signature per source being updated
    # Each signature should be from the data source that corresponds to the update_source_id
    all_signatures = []
    for source_id in update_source_ids:
        if source_id in signatures:
            sig_hex = signatures[source_id]
            if sig_hex.startswith('0x'):
                sig_hex = sig_hex[2:]
            all_signatures.append(bytes.fromhex(sig_hex))
        else:
            print(f"⚠ Warning: No signature found for {source_id}, using empty signature")
            all_signatures.append(b'')
    
    print(f"\nPrepared {len(all_signatures)} signatures (one per source being updated)")
    
    print(f"\nUpdating scores (+10 to each):")
    for i, source_id in enumerate(update_source_ids):
        print(f"  {source_id}: reliability {current_scores[source_id]['reliability']} -> {update_reliability_scores[i]}")
        print(f"             usefulness {current_scores[source_id]['usefulness']} -> {update_usefulness_scores[i]}")
    
    # Sign the message with the caller account (this signature is used for verification)
    # The caller signature should be included in the signatures array
    # Based on contract requirements, we need signatures matching the update_source_ids
    # Let's use the data source signatures for each corresponding source
    print(f"\nCalling feedback_and_update_score_records...")
    print(f"  Message: {message_json[:80]}...")
    print(f"  Signatures count: {len(all_signatures)}")
    print(f"  Update sources count: {len(update_source_ids)}")
    
    try:
        tx_hash = client.feedback_and_update_score_records(
            caller_private_key=caller_account['private_key'],
            message=message_json,
            signatures=all_signatures,  # One signature per source being updated
            update_source_ids=update_source_ids,
            update_reliability_scores=update_reliability_scores,
            update_usefulness_scores=update_usefulness_scores,
            info=f"Test update: added 10 to all scores for query: {query_text[:50]}"
        )
        
        print(f"✓ Transaction successful!")
        print(f"  Transaction hash: {tx_hash}")
        
        # Verify updated scores
        print(f"\nVerifying updated scores...")
        updated_scores = get_current_scores(client, DEFAULT_SOURCES)
        print("Updated scores:")
        for source_id, scores in updated_scores.items():
            print(f"  {source_id}: reliability={scores['reliability']}, usefulness={scores['usefulness']}")
            
            # Verify the update
            if source_id in current_scores:
                expected_reliability = current_scores[source_id]['reliability'] + 10
                expected_usefulness = current_scores[source_id]['usefulness'] + 10
                if scores['reliability'] == expected_reliability and scores['usefulness'] == expected_usefulness:
                    print(f"    ✓ Update verified correctly")
                else:
                    print(f"    ✗ Update mismatch! Expected: rel={expected_reliability}, use={expected_usefulness}")
        
    except Exception as e:
        print(f"✗ Transaction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_feedback_and_update()

