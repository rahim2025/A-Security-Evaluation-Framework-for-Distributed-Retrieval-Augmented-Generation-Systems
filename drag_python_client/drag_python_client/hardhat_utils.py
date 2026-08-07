import json
import os
from typing import Optional


def find_local_ignition_deployed_address(project_root: str) -> Optional[str]:
    """
    Try to read Ignition's deployed address for DragScores on the localhost chain (31337).
    Returns the address as a hex string if found, else None.
    """
    deployed_file = os.path.join(
        project_root,
        "drag_contract",
        "ignition",
        "deployments",
        "chain-31337",
        "deployed_addresses.json",
    )
    try:
        with open(deployed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ignition key equals module name + ":DragScores#DragScores"
        for key, value in data.items():
            if key.endswith("#DragScores") and isinstance(value, str) and value.startswith("0x"):
                return value
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


def load_drag_scores_abi(project_root: str) -> dict:
    """
    Load the DragScores ABI from Hardhat artifacts.
    """
    abi_file = os.path.join(
        project_root,
        "drag_contract",
        "artifacts",
        "contracts",
        "drag_scores.sol",
        "DragScores.json",
    )
    with open(abi_file, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    return artifact["abi"]


