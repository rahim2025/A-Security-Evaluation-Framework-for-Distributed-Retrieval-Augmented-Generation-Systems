import os
from typing import List, Optional, Tuple, Dict, Any

from eth_account import Account
from web3 import Web3
from web3.exceptions import Web3RPCError

from .hardhat_utils import find_local_ignition_deployed_address, load_drag_scores_abi


class ScoreRecordAlreadyExistsError(Exception):
    pass


class DragScoresClient:
    def __init__(
        self,
        project_root: str,
        provider_url: str = "http://127.0.0.1:8545",
        contract_address: Optional[str] = None,
    ) -> None:
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        if not self.web3.is_connected():
            raise RuntimeError(f"Web3 not connected to {provider_url}")

        self.abi = load_drag_scores_abi(project_root)

        env_addr = os.getenv("DRAG_SCORES_ADDRESS")
        if contract_address is None:
            contract_address = env_addr or find_local_ignition_deployed_address(project_root)
        if not contract_address:
            raise RuntimeError(
                "Could not resolve DragScores address. Set DRAG_SCORES_ADDRESS or deploy with Hardhat Ignition."
            )

        self.contract = self.web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=self.abi)

    # ----- view methods -----
    def hello(self) -> str:
        return self.contract.functions.hello().call()

    def get_reliability_score(self, source_id: str) -> int:
        return int(self.contract.functions.getReliabilityScore(source_id).call())

    def get_usefulness_score(self, source_id: str) -> int:
        return int(self.contract.functions.getUsefulnessScore(source_id).call())

    def get_scores_batch(self, source_ids: List[str]) -> Tuple[List[str], List[int], List[int]]:
        returned_source_ids, reliability, usefulness = self.contract.functions.getScoreRecordsBatch(source_ids).call()
        return list(returned_source_ids), list(map(int, reliability)), list(map(int, usefulness))

    # ----- state-changing methods -----
    def _build_and_send(self, func, sender_private_key: str, gas: Optional[int] = None) -> str:
        acct = Account.from_key(sender_private_key)
        tx = func.build_transaction(
            {
                "from": acct.address,
                "nonce": self.web3.eth.get_transaction_count(acct.address),
                "gas": gas or 2_000_000,
                "maxFeePerGas": self.web3.to_wei("2", "gwei"),
                "maxPriorityFeePerGas": self.web3.to_wei("1", "gwei"),
                "chainId": self.web3.eth.chain_id,
            }
        )
        signed = self.web3.eth.account.sign_transaction(tx, private_key=sender_private_key)
        try:
            tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        except Web3RPCError as e:
            err_text = str(e)
            if "Score record already exists" in err_text:
                print("Score record already exists: skipping creation or handle accordingly.")
                raise ScoreRecordAlreadyExistsError("Score record already exists") from e
            raise
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise RuntimeError("Transaction failed")
        return tx_hash.hex()

    def create_score_record_by_owner(
        self,
        owner_private_key: str,
        source_id: str,
        source_address: str,
        reliability_score: int,
        usefulness_score: int,
        reserved: str = "",
        ignore_exists: bool = False,
    ) -> Optional[str]:
        func = self.contract.functions.createScoreRecordByOwner(
            source_id,
            Web3.to_checksum_address(source_address),
            int(reliability_score),
            int(usefulness_score),
            reserved,
        )
        try:
            return self._build_and_send(func, owner_private_key)
        except ScoreRecordAlreadyExistsError:
            if ignore_exists:
                return None
            raise

    def update_score_record_by_owner(
        self,
        owner_private_key: str,
        source_id: str,
        timestamp: int,
        reserved: str,
    ) -> str:
        func = self.contract.functions.updateScoreRecordByOwner(
            source_id,
            int(timestamp),
            reserved,
        )
        return self._build_and_send(func, owner_private_key)

    def feedback_and_update_score_records(
        self,
        caller_private_key: str,
        message: str,
        signatures: List[bytes],
        update_source_ids: List[str],
        update_reliability_scores: List[int],
        update_usefulness_scores: List[int],
        info: str = "",
    ) -> str:
        func = self.contract.functions.feedbackAndUpdateScoreRecords(
            message,
            signatures,
            update_source_ids,
            list(map(int, update_reliability_scores)),
            list(map(int, update_usefulness_scores)),
            info,
        )
        return self._build_and_send(func, caller_private_key)

    # ----- event methods -----
    def get_score_record_updated_events(
        self,
        source_id: Optional[str] = None,
        source_address: Optional[str] = None,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for ScoreRecordUpdated events.
        
        Args:
            source_id: Filter by source ID (indexed parameter). If None, matches all.
            source_address: Filter by source address (indexed parameter). If None, matches all.
            from_block: Starting block number (inclusive). If None, starts from block 0 or earliest available.
            to_block: Ending block number (inclusive). If None, uses 'latest'.
            
        Returns:
            List of event dictionaries with the following keys:
            - sourceAddress: Address of the source
            - sourceID: ID of the source (indexed, may be bytes)
            - sourceName: Name of the source (unindexed, readable string)
            - reliabilityScore: Reliability score (int32)
            - usefulnessScore: Usefulness score (int32)
            - timestamp: Block timestamp (uint256)
            - info: Additional info string
            - blockNumber: Block number where event was emitted
            - transactionHash: Transaction hash
            - logIndex: Log index in the block
        """
        # Build argument filters for indexed parameters
        argument_filters = {}
        if source_address is not None:
            argument_filters["sourceAddress"] = Web3.to_checksum_address(source_address)
        if source_id is not None:
            argument_filters["sourceID"] = source_id
        
        # Set block range
        if from_block is None:
            from_block = max(0, self.web3.eth.block_number - 100)
        if to_block is None:
            to_block = self.web3.eth.block_number
        
        # Get events using web3.py's get_logs method
        logs = self.contract.events.ScoreRecordUpdated.get_logs(
            argument_filters=argument_filters if argument_filters else {},
            from_block=from_block,
            to_block=to_block
        )
        
        print(logs)
        # Parse events (get_logs returns decoded events)
        # Helper function to convert bytes/HexBytes to hex string
        def to_hex_string(value):
            """Convert bytes, HexBytes, or other types to hex string."""
            if value is None:
                return None
            if isinstance(value, bytes):
                return self.web3.to_hex(value)
            if hasattr(value, 'hex'):
                # HexBytes or similar - hex() returns string with 0x prefix
                return value.hex()
            if isinstance(value, str) and value.startswith('0x'):
                return value
            # For addresses and other string types, convert to string
            return str(value)
        
        # Helper function to convert sourceID/sourceName to readable string
        def to_source_name(value):
            """Convert sourceID or sourceName to readable string."""
            if value is None:
                return None
            if isinstance(value, bytes):
                # Try to decode as UTF-8, if that fails, return hex
                try:
                    decoded = value.decode('utf-8')
                    # Filter out null bytes and control characters
                    return ''.join(c for c in decoded if c.isprintable() or c == ' ')
                except (UnicodeDecodeError, AttributeError):
                    return self.web3.to_hex(value)
            if isinstance(value, str):
                return value
            return str(value)
        
        events = []
        for event in logs:
            # Extract sourceName (unindexed) - this is the readable source name
            source_name = event["args"].get("sourceName")
            if source_name is None:
                # Fallback: try to get from sourceID if sourceName is not available
                source_name = event["args"].get("sourceID")
            
            event_dict = {
                "sourceAddress": to_hex_string(event["args"]["sourceAddress"]),
                "sourceID": to_hex_string(event["args"]["sourceID"]),
                "sourceName": to_source_name(source_name),
                "reliabilityScore": int(event["args"]["reliabilityScore"]),
                "usefulnessScore": int(event["args"]["usefulnessScore"]),
                "timestamp": int(event["args"]["timestamp"]),
                "info": str(event["args"]["info"]),
                "blockNumber": int(event["blockNumber"]),
                "transactionHash": to_hex_string(event.get("transactionHash")),
                "logIndex": int(event["logIndex"]),
            }
            events.append(event_dict)
        
        return events


