import os
import yaml
from pathlib import Path
from typing import Tuple

from eth_account import Account
from web3 import Web3

from drag_python_client import DragScoresClient, sign_message_personal


def get_project_root() -> str:
    # Resolve to the monorepo root that contains drag_contract
    return str(Path(__file__).resolve().parents[3])


def get_hardhat_private_keys(provider_url: str) -> Tuple[str, list]:
    """
    Returns an owner private key and a list of 10 source private keys from the local Hardhat node.
    Uses the standard dev keys commonly documented for Hardhat.
    Can be overridden via env: OWNER_PK and SOURCES_PKS (comma-separated).
    """
    web3 = Web3(Web3.HTTPProvider(provider_url))
    if not web3.is_connected():
        raise RuntimeError("Connect to Hardhat first: npm run node")

    owner_pk = os.getenv(
        "OWNER_PK",
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    )

    env_sources = os.getenv("SOURCES_PKS")
    if env_sources:
        sources = [s.strip() for s in env_sources.split(",") if s.strip()]
    else:
        # First 11 Hardhat default keys (owner + 10 sources). We already took index 0 for owner.
        defaults = [
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
            "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
            "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
            "0x47e179ec9f2efb88fc8b5ae980f29e6020f3606578f8569f6313a9c8d3d8f6f7",
            "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d0f6e8ce6b3c8f7f7e",
            "0x0dbbe8e4f7b9a43653b83175727b0d1a6d6d5c3a3c0b7b5b3e6f8d7c6e5f4a3b",
            "0x8626f6940e2eb28930efb4cef49b2d1f2c9c1199b3dcdc47e1d7b9f2c5a4d8e7",
            "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d",
            "0x6c2ee5c6b7a7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0918273645544332211",
            "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc991e0c9a1f73c7f5f0d4d7a3",
        ]
        # Use next 10 as sources
        sources = defaults[1:11]
    if len(sources) < 10:
        raise RuntimeError("Need 10 source private keys; provide SOURCES_PKS env if defaults not available.")
    return owner_pk, sources[:10]


def main() -> None:
    provider_url = "http://127.0.0.1:8545"
    project_root = get_project_root()

    # Resolve accounts
    owner_pk, sources_pks = get_hardhat_private_keys(provider_url)
    owner = Account.from_key(owner_pk)
    sources_accts = [Account.from_key(pk) for pk in sources_pks]

    # Connect client
    client = DragScoresClient(
        project_root=project_root,
        provider_url=provider_url,
    )

    print("hello():", client.hello())

    # Create 10 source records (owner only)
    source_ids = [f"source-{i+1}" for i in range(10)]
    init_rel = [10 + i for i in range(10)]
    init_use = [20 + i for i in range(10)]
    for i in range(10):
        tx = client.create_score_record_by_owner(
            owner_private_key=owner_pk,
            source_id=source_ids[i],
            source_address=sources_accts[i].address,
            reliability_score=init_rel[i],
            usefulness_score=init_use[i],
            reserved=f"init-{i}",
            ignore_exists=True,
        )
        print(f"create[{i}] tx:", tx)

    # Test single getters
    for i in range(10):
        r = client.get_reliability_score(source_ids[i])
        u = client.get_usefulness_score(source_ids[i])
        assert r == init_rel[i] and u == init_use[i]

    # Test batch getter
    returned_ids, r_batch, u_batch = client.get_scores_batch(source_ids)
    assert returned_ids == source_ids
    assert r_batch == init_rel and u_batch == init_use

    # Feedback updates: first update 3 sources
    message_3 = "feedback:batch=3;selected=[source-1,source-2,source-3]"
    sigs_3 = [sign_message_personal(message_3, sources_pks[i]) for i in range(3)]
    upd_rel_3 = [100, 101, 102]
    upd_use_3 = [200, 201, 202]
    tx3 = client.feedback_and_update_score_records(
        caller_private_key=owner_pk,
        message=message_3,
        signatures=sigs_3,
        update_source_ids=source_ids[:3],
        update_reliability_scores=upd_rel_3,
        update_usefulness_scores=upd_use_3,
        info="batch-3",
    )
    print("feedback(3) tx:", tx3)

    # Verify first 3 updated, others unchanged
    returned_ids, r_batch, u_batch = client.get_scores_batch(source_ids)
    assert returned_ids == source_ids
    assert r_batch[:3] == upd_rel_3 and u_batch[:3] == upd_use_3
    assert r_batch[3:] == init_rel[3:] and u_batch[3:] == init_use[3:]

    # Feedback updates: update next 7 sources
    message_7 = "feedback:batch=7;selected=[source-4..source-10]"
    sigs_7 = [sign_message_personal(message_7, pk) for pk in sources_pks[3:10]]
    upd_rel_7 = [300 + i for i in range(7)]
    upd_use_7 = [400 + i for i in range(7)]
    tx4 = client.feedback_and_update_score_records(
        caller_private_key=owner_pk,
        message=message_7,
        signatures=sigs_7,
        update_source_ids=source_ids[3:10],
        update_reliability_scores=upd_rel_7,
        update_usefulness_scores=upd_use_7,
        info="batch-7",
    )
    print("feedback(7) tx:", tx4)

    # Verify all 10 reflect the two batches
    returned_ids, r_batch, u_batch = client.get_scores_batch(source_ids)
    assert returned_ids == source_ids
    assert r_batch[:3] == upd_rel_3 and u_batch[:3] == upd_use_3
    assert r_batch[3:10] == upd_rel_7 and u_batch[3:10] == upd_use_7

    # Test update by owner (timestamp/reserved only) on source-1
    import time
    now = int(time.time())
    tx5 = client.update_score_record_by_owner(owner_private_key=owner_pk, source_id=source_ids[0], timestamp=now, reserved="owner-touch")
    print("owner update tx:", tx5)

    # Feedback updates: update all 10
    message_10 = "feedback:batch=10;selected=[all]"
    sigs_10 = [sign_message_personal(message_10, pk) for pk in sources_pks]
    upd_rel_10 = [500 + i for i in range(10)]
    upd_use_10 = [600 + i for i in range(10)]
    tx6 = client.feedback_and_update_score_records(
        caller_private_key=owner_pk,
        message=message_10,
        signatures=sigs_10,
        update_source_ids=source_ids,
        update_reliability_scores=upd_rel_10,
        update_usefulness_scores=upd_use_10,
        info="batch-10",
    )
    print("feedback(10) tx:", tx6)

    # Verify final state
    returned_ids, r_final, u_final = client.get_scores_batch(source_ids)
    assert returned_ids == source_ids
    print("final scores:", r_final, u_final)
    assert r_final == upd_rel_10 and u_final == upd_use_10


def test_create_default_sources_from_configs() -> None:
    """
    Creates 3 default sources using source names and private keys from drag_data_source configs.
    Both reliability_score and usefulness_score are initialized to 10000.
    """
    provider_url = "http://127.0.0.1:8545"
    project_root = get_project_root()
    
    # Path to drag_data_source configs directory
    configs_dir = Path(project_root) / "drag_data_source" / "configs"
    
    # Load the 3 config files
    config_files = [
        "config_sources_0.yaml",
        "config_sources_20.yaml",
        "config_sources_100.yaml",
    ]
    
    sources_data = []
    for config_file in config_files:
        config_path = configs_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        source_name = config['data']['dataset_name']
        private_key = config['blockchain']['private_key']
        source_address = Account.from_key(private_key).address
        
        sources_data.append({
            'source_id': source_name,
            'private_key': private_key,
            'source_address': source_address,
        })
    
    # Connect client
    client = DragScoresClient(
        project_root=project_root,
        provider_url=provider_url,
    )
    
    print("hello():", client.hello())
    
    # Create 3 source records with both scores initialized to 10000
    reliability_score = 10000
    usefulness_score = 10000
    
    for i, source in enumerate(sources_data):
        tx = client.create_score_record_by_owner(
            owner_private_key=source['private_key'],
            source_id=source['source_id'],
            source_address=source['source_address'],
            reliability_score=reliability_score,
            usefulness_score=usefulness_score,
            reserved=f"default-init-{i}",
            ignore_exists=True,
        )
        print(f"create[{i}] source_id={source['source_id']}, tx={tx}")
    
    # Verify all sources were created with correct scores
    source_ids = [s['source_id'] for s in sources_data]
    returned_ids, r_batch, u_batch = client.get_scores_batch(source_ids)
    
    assert returned_ids == source_ids
    assert all(r == reliability_score for r in r_batch), f"Expected all reliability scores to be {reliability_score}, got {r_batch}"
    assert all(u == usefulness_score for u in u_batch), f"Expected all usefulness scores to be {usefulness_score}, got {u_batch}"
    
    print(f"Successfully created and verified {len(sources_data)} sources with scores: reliability={reliability_score}, usefulness={usefulness_score}")
    print(f"Source IDs: {source_ids}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_default_sources":
        test_create_default_sources_from_configs()
    else:
        main()


