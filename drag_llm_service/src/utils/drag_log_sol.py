"""
Draglog Solidity Contract Interaction Class

This module provides a Python interface to interact with the Draglog smart contract
deployed on Ethereum. It supports all contract operations including creating,
reading, updating records, and managing reliability scores.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

from datetime import datetime

try:
    import fsspec
except ImportError:
    pass

class DragLogSol:
    """
    Python class to interact with the Draglog smart contract.
    
    This class provides a high-level interface for all contract operations
    including reliability records, log records, and feedback records.
    """
    
    def __init__(
        self,
        contract_address: str,
        network_url: str,
        private_key: Optional[str] = None,
        account_address: Optional[str] = None,
        abi_path: Optional[str] = None
    ):
        """
        Initialize the DragLogSol class.
        
        Args:
            contract_address (str): The deployed contract address
            network_url (str): The network RPC URL (e.g., 'http://127.0.0.1:8545')
            private_key (str, optional): Private key for transactions. If None, read-only mode
            account_address (str, optional): Account address. If None, derived from private_key
            abi_path (str, optional): Path to contract ABI JSON file. If None, uses default path
        """
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.network_url = network_url
        self.private_key = private_key
        self.account_address = account_address
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(network_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to network at {network_url}")
        
        # Load contract ABI
        if abi_path is None:
            abi_path = "../artifacts/contracts/Draglog.sol/Draglog.json"
        
        try:
            # Try fsspec first if available, otherwise fall back to regular open
            if 'fsspec' in globals():
                with fsspec.open(abi_path, 'r') as f:
                    print(f"Loading contract ABI from {abi_path}")
                    contract_json = json.load(f)
                    self.contract_abi = contract_json['abi']
            else:
                with open(abi_path, 'r') as f:
                    print(f"Loading contract ABI from {abi_path}")
                    contract_json = json.load(f)
                    self.contract_abi = contract_json['abi']
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract ABI not found at {abi_path}")
        
        # Create contract instance
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        
        # Set up account if private key provided
        if private_key:
            self.account: LocalAccount = Account.from_key(private_key)
            if account_address:
                if self.account.address.lower() != account_address.lower():
                    raise ValueError("Private key doesn't match provided account address")
            self.account_address = self.account.address
        else:
            self.account = None
            if not account_address:
                raise ValueError("Either private_key or account_address must be provided")
            self.account_address = Web3.to_checksum_address(account_address)
        
        # Verify contract connection
        try:
            hello_message = self.contract.functions.hello().call()
            print(f"✅ Connected to Draglog contract: {hello_message}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to contract: {e}")
    
    def _build_transaction(self, function_call, gas_limit: int = 300000) -> Dict:
        """
        Build a transaction for contract interaction.
        
        Args:
            function_call: The contract function call
            gas_limit (int): Gas limit for the transaction
            
        Returns:
            Dict: Transaction dictionary
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        return function_call.build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': gas_limit,
            'gasPrice': self.w3.eth.gas_price
        })
    
    def _send_transaction(self, transaction: Dict) -> Dict:
        """
        Send a signed transaction and wait for receipt.
        
        Args:
            transaction (Dict): Transaction dictionary
            
        Returns:
            Dict: Transaction receipt
        """
        if not self.account:
            raise ValueError("Private key required for transactions")
        
        # Sign transaction
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        
        # Send transaction - handle different web3.py versions
        if hasattr(signed_txn, 'rawTransaction'):
            # For older web3.py versions (v5-)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        elif hasattr(signed_txn, 'raw_transaction'):
            # For newer web3.py versions (v6+)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        else:
            raise ValueError("Unable to find raw transaction data in signed transaction")
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, poll_latency=3)
        
        return receipt
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get network information.
        
        Returns:
            Dict: Network information
        """
        return {
            'connected': self.w3.is_connected(),
            'chain_id': self.w3.eth.chain_id,
            'block_number': self.w3.eth.block_number,
            'contract_address': self.contract_address,
            'account_address': self.account_address,
            'account_balance': self.w3.from_wei(
                self.w3.eth.get_balance(self.account_address), 'ether'
            )
        }
    
    def get_contract_stats(self) -> Dict[str, Any]:
        """
        Get contract statistics.
        
        Returns:
            Dict: Contract statistics
        """
        try:
            total_records = self.contract.functions.getRecordCount().call()
            reliability_records = self.contract.functions.getReliabilityRecords().call()   
            feedback_records = self.contract.functions.getFeedbackRecords().call()
            usefulness_records = self.contract.functions.getUsefulnessRecords().call()
            
            return {
                'total_records': total_records,
                'reliability_records': len(reliability_records),
                'feedback_records': len(feedback_records),
                'usefulness_records': len(usefulness_records),
                'has_feedback': self.contract.functions.hasRecordType("feedback").call(),
                'has_usefulness': self.contract.functions.hasRecordType("usefulness").call(),
                'has_reliability': self.contract.functions.hasRecordType("reliability").call(),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def does_record_exist(self, record_id: str) -> bool:
        """
        Check if a record exists.
        
        Args:
            record_id (str): The record ID to check
            
        Returns:
            bool: True if record exists, False otherwise
        """
        return self.contract.functions.doesRecordExist(record_id).call()
    
    def read_reliability_record(self, data_source_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a reliability record.
        
        Args:
            data_source_id (str): The data source ID
            
        Returns:
            Dict: Reliability record data or None if not found
        """
        try:
            record = self.contract.functions.readReliabilityRecord(data_source_id).call()
            return {
                'logID': record[0],
                'loggerID': record[1],
                'recordType': record[2],
                'input': record[3],
                'inputFrom': record[4],
                'output': record[5],
                'outputTo': record[6],
                'reliabilityScore': record[7],
                'timestamp': record[8],
                'reserved': record[9]
            }
        except Exception as e:
            print(f"Error reading reliability record: {e}")
            return None
    
    def read_log_record(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a log record.
        
        Args:
            log_id (str): The log ID
            
        Returns:
            Dict: Log record data or None if not found
        """
        try:
            record = self.contract.functions.readLogRecord(log_id).call()
            return {
                'logID': record[0],
                'loggerID': record[1],
                'recordType': record[2],
                'input': record[3],
                'inputFrom': record[4],
                'output': record[5],
                'outputTo': record[6],
                'reliabilityScore': record[7],
                'timestamp': record[8],
                'reserved': record[9]
            }
        except Exception as e:
            print(f"Error reading log record: {e}")
            return None
    
    def read_feedback_record(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a feedback record.
        
        Args:
            log_id (str): The log ID
            
        Returns:
            Dict: Feedback record data or None if not found
        """
        try:
            record = self.contract.functions.readFeedbackRecord(log_id).call()
            return {
                'logID': record[0],
                'loggerID': record[1],
                'recordType': record[2],
                'input': record[3],
                'inputFrom': record[4],
                'output': record[5],
                'outputTo': record[6],
                'reliabilityScore': record[7],
                'timestamp': record[8],
                'reserved': record[9]
            }
        except Exception as e:
            print(f"Error reading feedback record: {e}")
            return None
    
    def create_reliability_record(
        self,
        data_source_id: str,
        digest: str,
        reserved: str = "",
        score: int = 100,
        gas_limit: int = 300000
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new reliability record.
        
        Args:
            data_source_id (str): The data source ID
            digest (str): The digest/hash
            reserved (str): Reserved field
            score (int): The reliability score
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record already exists
            if self.does_record_exist(data_source_id):
                print(f"⚠️  Record '{data_source_id}' already exists")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.createReliabilityRecord(data_source_id, digest, reserved, score),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Reliability record '{data_source_id}' created successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error creating reliability record: {e}")
            return None
    
    def create_log_record(
        self,
        log_id: str,
        logger_id: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        timestamp: str,
        reserved: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new log record.
        
        Args:
            log_id (str): The log ID
            logger_id (str): The logger ID
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            timestamp (str): Timestamp
            reserved (str): Reserved field
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record already exists
            if self.does_record_exist(log_id):
                print(f"⚠️  Record '{log_id}' already exists")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.createLogRecord(
                    log_id, logger_id, input_data, input_from,
                    output, output_to, timestamp, reserved
                )
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Log record '{log_id}' created successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error creating log record: {e}")
            return None
    
    def create_feedback_record(
        self,
        log_id: str,
        logger_id: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        timestamp: str,
        reserved: str = "",
        gas_limit: int = 500000
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new feedback record.
        
        Args:
            log_id (str): The log ID
            logger_id (str): The logger ID
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            timestamp (str): Timestamp
            reserved (str): Reserved field
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record already exists
            if self.does_record_exist(log_id):
                print(f"⚠️  Record '{log_id}' already exists")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.createFeedbackRecord(
                    log_id, logger_id, input_data, input_from,
                    output, output_to, timestamp, reserved
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Feedback record '{log_id}' created successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error creating feedback record: {e}")
            return None
    
    def update_reliability_score(
        self,
        data_source_id: str,
        score_delta: int,
        is_delta: bool = True,
        info: str = "",
        gas_limit: int = 300000
    ) -> Optional[Dict[str, Any]]:
        """
        Update the reliability score of a data source.
        
        Args:
            data_source_id (str): The data source ID
            score_delta (int): Score change or new score
            is_delta (bool): True if score_delta is a change, False if absolute value
            info (str): Additional information
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record exists
            if not self.does_record_exist(data_source_id):
                print(f"❌ Record '{data_source_id}' does not exist")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.updateReliabilityScore(
                    data_source_id, score_delta, is_delta, info
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Reliability score updated for '{data_source_id}'")
            print(f"Score delta: {score_delta}")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error updating reliability score: {e}")
            return None

    def batch_update_reliability_scores_contract(
        self,
        updates_data: List[Tuple[str, int, bool, str]],
        gas_limit: int = 2000000
    ) -> Optional[Dict[str, Any]]:
        """
        Update multiple reliability scores in a single contract transaction.
        This is more gas-efficient than individual updates but requires all updates to succeed or fail together.
        
        Args:
            updates_data: List of tuples (data_source_id, score, is_delta, info)
            gas_limit (int): Gas limit for the batch transaction
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        if not updates_data:
            print("❌ No update data provided")
            return None
        
        if len(updates_data) > 50:
            print("❌ Batch size limited to 50 records to prevent gas issues")
            return None
        
        try:
            # Prepare arrays for the contract call
            data_source_ids = [update[0] for update in updates_data]
            scores = [update[1] for update in updates_data]
            is_deltas = [update[2] for update in updates_data]
            infos = [update[3] for update in updates_data]
            
            print(f"🚀 Batch updating {len(updates_data)} reliability scores in single transaction...")
            
            # Validate all records exist before sending transaction
            for data_source_id in data_source_ids:
                if not self.does_record_exist(data_source_id):
                    print(f"❌ Record '{data_source_id}' does not exist")
                    return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.batchUpdateReliabilityScores(
                    data_source_ids, scores, is_deltas, infos
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            if receipt['status'] == 1:
                print(f"✅ Batch update successful for {len(updates_data)} records")
                print(f"Transaction hash: {receipt['transactionHash'].hex()}")
                print(f"Gas used: {receipt['gasUsed']}")
                
                # Show individual updates
                for i, (data_source_id, score, is_delta, info) in enumerate(updates_data):
                    action = "delta" if is_delta else "absolute"
                    print(f"  {i+1}. {data_source_id}: {action} {score}")
                
                return receipt
            else:
                print(f"❌ Batch transaction failed")
                return None
            
        except Exception as e:
            print(f"❌ Error in batch update: {e}")
            return {'error': 'Error in batch update: ' + str(e)}
    
    def update_log_record(
        self,
        log_id: str,
        logger_id: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        timestamp: str,
        reserved: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Update a log record.
        
        Args:
            log_id (str): The log ID
            logger_id (str): The logger ID
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            timestamp (str): Timestamp
            reserved (str): Reserved field
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record exists
            if not self.does_record_exist(log_id):
                print(f"❌ Record '{log_id}' does not exist")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.updateLogRecord(
                    log_id, logger_id, input_data, input_from,
                    output, output_to, timestamp, reserved
                ),
                gas_limit=150000
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Log record '{log_id}' updated successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error updating log record: {e}")
            return None
    
    def get_all_reliability_records(self) -> List[Dict[str, Any]]:
        """
        Get all reliability records.
        
        Returns:
            List[Dict]: List of reliability records
        """
        try:
            records = self.contract.functions.getReliabilityRecords().call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'reliabilityScore': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting reliability records: {e}")
            return []
    
    def get_all_log_records(self) -> List[Dict[str, Any]]:
        """
        Get all log records.
        
        Returns:
            List[Dict]: List of log records
        """
        try:
            records = self.contract.functions.getLogRecords().call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'reliabilityScore': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting log records: {e}")
            return []
    
    def get_all_feedback_records(self) -> List[Dict[str, Any]]:
        """
        Get all feedback records.
        
        Returns:
            List[Dict]: List of feedback records
        """
        try:
            records = self.contract.functions.getFeedbackRecords().call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'reliabilityScore': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting feedback records: {e}")
            return []
    
    def get_records_by_type(self, record_type: str) -> List[Dict[str, Any]]:
        """
        Get records by type.
        
        Args:
            record_type (str): Record type ("reliability", "log", "feedback")
            
        Returns:
            List[Dict]: List of records of the specified type
        """
        try:
            records = self.contract.functions.getRecordsByType(record_type).call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'reliabilityScore': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting records by type: {e}")
            return []

    
    def batch_update_reliability_scores(
        self,
        updates_data: List[Tuple[str, int, bool, str]],
        gas_limit: int = 300000,
        timeout: int = 120,
        poll_latency: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Update multiple reliability scores in batch to reduce waiting time.
        
        Args:
            updates_data: List of tuples (data_source_id, score_delta, is_delta, info)
            gas_limit (int): Gas limit for each transaction
            timeout (int): Maximum time to wait for each transaction receipt
            poll_latency (float): Polling interval for transaction receipts
            
        Returns:
            List[Dict]: List of transaction results with success status and receipts
        """
        if not updates_data:
            print("❌ No update data provided")
            return []
        
        if not self.account:
            raise ValueError("Private key required for batch transactions")
        
        print(f"🚀 Starting batch update of {len(updates_data)} reliability scores...")
        
        # Phase 1: Validate all records exist and build transactions
        print("📋 Phase 1: Validating records and building transactions...")
        transactions = []
        tx_data = []
        
        base_nonce = self.w3.eth.get_transaction_count(self.account_address)
        
        for i, (data_source_id, score_delta, is_delta, info) in enumerate(updates_data):
            try:
                # Check if record exists
                if not self.does_record_exist(data_source_id):
                    print(f"⚠️  Record '{data_source_id}' does not exist - skipping")
                    tx_data.append({
                        'data_source_id': data_source_id,
                        'success': False,
                        'error': 'Record does not exist'
                    })
                    continue
                
                # Build transaction with incremental nonce
                transaction = {
                    'from': self.account_address,
                    'nonce': base_nonce + len(transactions),
                    'gas': gas_limit,
                    'gasPrice': self.w3.eth.gas_price
                }
                
                # Build the contract function call
                function_call = self.contract.functions.updateReliabilityScore(
                    data_source_id, score_delta, is_delta, info
                )
                transaction = function_call.build_transaction(transaction)
                
                transactions.append(transaction)
                tx_data.append({
                    'data_source_id': data_source_id,
                    'transaction': transaction,
                    'success': None
                })
                
            except Exception as e:
                print(f"❌ Error preparing transaction for '{data_source_id}': {e}")
                tx_data.append({
                    'data_source_id': data_source_id,
                    'success': False,
                    'error': str(e)
                })
        
        if not transactions:
            print("❌ No valid transactions to send")
            return tx_data
        
        # Phase 2: Sign and send all transactions
        print("🚀 Sending transactions...")
        
        for i, (transaction, data) in enumerate(zip(transactions, [d for d in tx_data if d.get('transaction')])):
            try:
                signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
                
                if hasattr(signed_txn, 'rawTransaction'):
                    tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                elif hasattr(signed_txn, 'raw_transaction'):
                    tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
                else:
                    raise ValueError("Unable to find raw transaction data")
                
                receipt = self.w3.eth.wait_for_transaction_receipt(
                    tx_hash,
                    timeout=timeout,
                    poll_latency=poll_latency
                )
                
                data['success'] = receipt['status'] == 1
                if not data['success']:
                    data['error'] = 'Transaction failed'
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"❌ Error processing transaction for '{data['data_source_id']}': {e}")
                data['success'] = False
                data['error'] = str(e)
        
        successful = sum(1 for data in tx_data if data['success'] is True)
        failed = sum(1 for data in tx_data if data['success'] is False)
        
        print(f"✅ Successful: {successful} | ❌ Failed: {failed} | 📋 Total: {len(tx_data)}")
        
        return tx_data
    
    def get_events(
        self,
        event_name: str,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contract events using the working method (from_block/to_block keyword arguments).
        
        Args:
            event_name (str): Event name (e.g., "ReliabilityRecordCreated")
            from_block (int, optional): Start block number
            to_block (int, optional): End block number
            
        Returns:
            List[Dict]: List of events
        """
        try:
            if from_block is None:
                from_block = max(0, self.w3.eth.block_number - 100)
            if to_block is None:
                to_block = self.w3.eth.block_number
            event_filter = getattr(self.contract.events, event_name)
            events = event_filter.get_logs(
                from_block=from_block,
                to_block=to_block
            )
            return events
        except Exception as e:
            print(f"Error getting events: {e}")
            return []
    
    def get_recent_events_simple(self, event_name: str, block_range: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events using a simpler approach that works better with Hardhat.
        
        Args:
            event_name (str): Event name (e.g., "ReliabilityRecordCreated")
            block_range (int): Number of recent blocks to check
            
        Returns:
            List[Dict]: List of events
        """
        try:
            current_block = self.w3.eth.block_number
            from_block = max(0, current_block - block_range)
            
            # Get the event filter
            event_filter = getattr(self.contract.events, event_name)
            
            # Try to get events from recent blocks
            try:
                # For Hardhat, try getting all events and filter by block number
                all_events = event_filter.get_logs()
                
                # Filter events by block number
                recent_events = [
                    event for event in all_events 
                    if from_block <= event['blockNumber'] <= current_block
                ]
                
                return recent_events
                
            except Exception as e:
                print(f"Simple event filtering failed: {e}")
                
                # Fallback: return empty list
                return []
                
        except Exception as e:
            print(f"Error in get_recent_events_simple: {e}")
            return []
    
    def print_contract_status(self):
        """
        Print comprehensive contract status.
        """
        print("=== Draglog Contract Status ===")
        
        # Network info
        network_info = self.get_network_info()
        print(f"Network: Connected={network_info['connected']}")
        print(f"Chain ID: {network_info['chain_id']}")
        print(f"Block Number: {network_info['block_number']}")
        print(f"Contract Address: {network_info['contract_address']}")
        print(f"Account Address: {network_info['account_address']}")
        print(f"Account Balance: {network_info['account_balance']} ETH")
        
        # Contract stats
        stats = self.get_contract_stats()
        print(f"\n=== Contract Statistics ===")
        print(f"Total Records: {stats.get('total_records', 'N/A')}")
        print(f"Reliability Records: {stats.get('reliability_records', 'N/A')}")
        print(f"Log Records: {stats.get('log_records', 'N/A')}")
        print(f"Feedback Records: {stats.get('feedback_records', 'N/A')}")
        print(f"Usefulness Records: {stats.get('usefulness_records', 'N/A')}")
        
        print(f"\n=== Record Type Availability ===")
        print(f"Reliability Records: {'✓' if stats.get('has_reliability') else '✗'}")
        print(f"Log Records: {'✓' if stats.get('has_log') else '✗'}")
        print(f"Feedback Records: {'✓' if stats.get('has_feedback') else '✗'}")
        print(f"Usefulness Records: {'✓' if stats.get('has_usefulness') else '✗'}")

    def read_usefulness_record(self, log_id: str) -> Optional[Dict[str, Any]]:
        """
        Read a usefulness record.
        
        Args:
            log_id (str): The log ID
            
        Returns:
            Dict: Usefulness record data or None if not found
        """
        try:
            record = self.contract.functions.readUsefulnessRecord(log_id).call()
            return {
                'logID': record[0],
                'loggerID': record[1],
                'recordType': record[2],
                'input': record[3],
                'inputFrom': record[4],
                'output': record[5],
                'outputTo': record[6],
                'score': record[7],
                'timestamp': record[8],
                'reserved': record[9]
            }
        except Exception as e:
            print(f"Error reading usefulness record: {e}")
            return None

    def read_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Read any record by ID.
        
        Args:
            record_id (str): The record ID
            
        Returns:
            Dict: Record data or None if not found
        """
        try:
            record = self.contract.functions.readRecord(record_id).call()
            return {
                'logID': record[0],
                'loggerID': record[1],
                'recordType': record[2],
                'input': record[3],
                'inputFrom': record[4],
                'output': record[5],
                'outputTo': record[6],
                'score': record[7],
                'timestamp': record[8],
                'reserved': record[9]
            }
        except Exception as e:
            print(f"Error reading record: {e}")
            return None

    def create_usefulness_record(
        self,
        log_id: str,
        logger_id: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        timestamp: str,
        reserved: str = "",
        gas_limit: int = 300000
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new usefulness record.
        
        Args:
            log_id (str): The log ID
            logger_id (str): The logger ID
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            timestamp (str): Timestamp
            reserved (str): Reserved field
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record already exists
            if self.does_record_exist(log_id):
                print(f"⚠️  Record '{log_id}' already exists")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.createUsefulnessRecord(
                    log_id, logger_id, input_data, input_from,
                    output, output_to, timestamp, reserved
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Usefulness record '{log_id}' created successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error creating usefulness record: {e}")
            return None

    def create_record(
        self,
        record_id: str,
        logger_id: str,
        record_type: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        score: int,
        timestamp: str,
        reserved: str = "",
        gas_limit: int = 500000
    ) -> Optional[Dict[str, Any]]:
        """
        Create a general record with specified type.
        
        Args:
            record_id (str): The record ID
            logger_id (str): The logger ID
            record_type (str): Record type ("reliability", "log", "feedback", "usefulness")
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            score (int): Score value
            timestamp (str): Timestamp
            reserved (str): Reserved field
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record already exists
            if self.does_record_exist(record_id):
                print(f"⚠️  Record '{record_id}' already exists")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.createRecord(
                    record_id, logger_id, record_type, input_data, input_from,
                    output, output_to, score, timestamp, reserved
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Record '{record_id}' of type '{record_type}' created successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error creating record: {e}")
            return None

    def update_record(
        self,
        record_id: str,
        logger_id: str,
        record_type: str,
        input_data: str,
        input_from: str,
        output: str,
        output_to: str,
        timestamp: str,
        reserved: str = "",
        info: str = "",
        gas_limit: int = 300000
    ) -> Optional[Dict[str, Any]]:
        """
        Update a general record.
        
        Args:
            record_id (str): The record ID
            logger_id (str): The logger ID
            record_type (str): Record type
            input_data (str): Input data
            input_from (str): Input source
            output (str): Output data
            output_to (str): Output destination
            timestamp (str): Timestamp
            reserved (str): Reserved field
            info (str): Update information
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            # Check if record exists
            if not self.does_record_exist(record_id):
                print(f"❌ Record '{record_id}' does not exist")
                return None
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.updateRecord(
                    record_id, logger_id, record_type, input_data, input_from,
                    output, output_to, timestamp, reserved, info
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Record '{record_id}' updated successfully")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error updating record: {e}")
            return None

    def update_record_score(
        self,
        record_id: str,
        score: int,
        is_delta: bool = True,
        info: str = "",
        gas_limit: int = 300000
    ) -> Optional[Dict[str, Any]]:
        """
        Update the score of any record.
        
        Args:
            record_id (str): The record ID
            score (int): Score change or new score
            is_delta (bool): True if score is a change, False if absolute value
            info (str): Additional information
            gas_limit (int): Gas limit for the transaction
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:

            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.updateRecordScore(record_id, score, is_delta, info),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            print(f"✅ Record score updated for '{record_id}'")
            print(f"Score {'delta' if is_delta else 'absolute'}: {score}")
            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            return receipt
            
        except Exception as e:
            print(f"❌ Error updating record score: {e}")
            return None

    def batch_update_record_scores_contract(
        self,
        updates_data: List[Tuple[str, int, bool, str]],
        gas_limit: int = 2000000
    ) -> Optional[Dict[str, Any]]:
        """
        Update multiple record scores in a single contract transaction.
        This is more gas-efficient than individual updates but requires all updates to succeed or fail together.
        
        Args:
            updates_data: List of tuples (record_id, score, is_delta, info)
            gas_limit (int): Gas limit for the batch transaction
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        if not updates_data:
            print("❌ No update data provided")
            return None
        
        if len(updates_data) > 50:
            print("❌ Batch size limited to 50 records to prevent gas issues")
            return None
        
        try:
            # Prepare arrays for the contract call
            record_ids = [update[0] for update in updates_data]
            scores = [update[1] for update in updates_data]
            is_deltas = [update[2] for update in updates_data]
            infos = [update[3] for update in updates_data]
            
            print(f"🚀 Batch updating {len(updates_data)} record scores in single transaction...")
            
            # Build transaction
            transaction = self._build_transaction(
                self.contract.functions.batchUpdateRecordScores(
                    record_ids, scores, is_deltas, infos
                ),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            if receipt['status'] == 1:
                print(f"✅ Batch update successful for {len(updates_data)} records")
                print(f"Transaction hash: {receipt['transactionHash'].hex()}")
                print(f"Gas used: {receipt['gasUsed']}")
                
                # Show individual updates
                for i, (record_id, score, is_delta, info) in enumerate(updates_data):
                    action = "delta" if is_delta else "absolute"
                    print(f"  {i+1}. {record_id}: {action} {score}")
                
                return receipt
            else:
                print(f"❌ Batch transaction failed")
                return None
            
        except Exception as e:
            print(f"❌ Error in batch update: {e}")
            return {'error': 'Error in batch update: ' + str(e)}

    def get_all_records(self) -> List[Dict[str, Any]]:
        """
        Get all records from the contract.
        
        Returns:
            List[Dict]: List of all records
        """
        try:
            records = self.contract.functions.getAllRecords().call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'score': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting all records: {e}")
            return []

    def get_all_record_ids(self) -> List[str]:
        """
        Get all record IDs from the contract.
        
        Returns:
            List[str]: List of all record IDs
        """
        try:
            return self.contract.functions.getAllRecordIDs().call()
        except Exception as e:
            print(f"Error getting all record IDs: {e}")
            return []

    def get_usefulness_records(self) -> List[Dict[str, Any]]:
        """
        Get all usefulness records.
        
        Returns:
            List[Dict]: List of usefulness records
        """
        try:
            records = self.contract.functions.getUsefulnessRecords().call()
            return [
                {
                    'logID': record[0],
                    'loggerID': record[1],
                    'recordType': record[2],
                    'input': record[3],
                    'inputFrom': record[4],
                    'output': record[5],
                    'outputTo': record[6],
                    'score': record[7],
                    'timestamp': record[8],
                    'reserved': record[9]
                }
                for record in records
            ]
        except Exception as e:
            print(f"Error getting usefulness records: {e}")
            return []

    def get_record_count(self) -> int:
        """
        Get the total number of records in the contract.
        
        Returns:
            int: Total record count
        """
        try:
            return self.contract.functions.getRecordCount().call()
        except Exception as e:
            print(f"Error getting record count: {e}")
            return 0

    def has_record_type(self, record_type: str) -> bool:
        """
        Check if the contract has any records of the specified type.
        
        Args:
            record_type (str): Record type to check
            
        Returns:
            bool: True if records of this type exist, False otherwise
        """
        try:
            return self.contract.functions.hasRecordType(record_type).call()
        except Exception as e:
            print(f"Error checking record type: {e}")
            return False



    # def batch_update_record_scores(
    #     self,
    #     updates_data: List[Tuple[str, int, bool, str]],
    #     gas_limit: int = 300000,
    #     timeout: int = 120,
    #     poll_latency: float = 0.5
    # ) -> List[Dict[str, Any]]:
    #     """
    #     Update multiple record scores in batch to reduce waiting time.
    #     
    #     Args:
    #         updates_data: List of tuples (record_id, score, is_delta, info)
    #         gas_limit (int): Gas limit for each transaction
    #         timeout (int): Maximum time to wait for each transaction receipt
    #         poll_latency (float): Polling interval for transaction receipts
    #         
    #     Returns:
    #         List[Dict]: List of transaction results with success status and receipts
    #     """
    #     if not updates_data:
    #         print("❌ No update data provided")
    #         return []
    #     
    #     if not self.account:
    #         raise ValueError("Private key required for batch transactions")
    #     
    #     print(f"🚀 Starting batch update of {len(updates_data)} record scores...")
    #     
    #     # Phase 1: Validate all records exist and build transactions
    #     print("📋 Phase 1: Validating records and building transactions...")
    #     transactions = []
    #     tx_data = []
    #     
    #     base_nonce = self.w3.eth.get_transaction_count(self.account_address)
    #     
    #     for i, (record_id, score, is_delta, info) in enumerate(updates_data):
    #         try:
    #             # Check if record exists
    #             if not self.does_record_exist(record_id):
    #                 print(f"⚠️  Record '{record_id}' does not exist - skipping")
    #                 tx_data.append({
    #                     'record_id': record_id,
    #                     'success': False,
    #                     'error': 'Record does not exist'
    #                 })
    #                 continue
    #             
    #             # Build transaction with incremental nonce
    #             transaction = {
    #                 'from': self.account_address,
    #                 'nonce': base_nonce + len(transactions),
    #                 'gas': gas_limit,
    #                 'gasPrice': self.w3.eth.gas_price
    #             }
    #             
    #             # Build the contract function call
    #             function_call = self.contract.functions.updateRecordScore(
    #                 record_id, score, is_delta, info
    #             )
    #             transaction = function_call.build_transaction(transaction)
    #             
    #             transactions.append(transaction)
    #             tx_data.append({
    #                 'record_id': record_id,
    #                 'transaction': transaction,
    #                 'success': None
    #             })
    #             
    #         except Exception as e:
    #             print(f"❌ Error preparing transaction for '{record_id}': {e}")
    #             tx_data.append({
    #                 'record_id': record_id,
    #                 'success': False,
    #                 'error': str(e)
    #             })
    #     
    #     if not transactions:
    #         print("❌ No valid transactions to send")
    #         return tx_data
    #     
    #     # Phase 2: Sign and send all transactions
    #     print("🚀 Sending transactions...")
    #     
    #     for i, (transaction, data) in enumerate(zip(transactions, [d for d in tx_data if d.get('transaction')])):
    #         try:
    #             signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
    #             
    #             if hasattr(signed_txn, 'rawTransaction'):
    #                 tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    #             elif hasattr(signed_txn, 'raw_transaction'):
    #                 tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    #             else:
    #                 raise ValueError("Unable to find raw transaction data")
    #             
    #             receipt = self.w3.eth.wait_for_transaction_receipt(
    #                 tx_hash,
    #                 timeout=timeout,
    #                 poll_latency=poll_latency
    #             )
    #             
    #             data['success'] = receipt['status'] == 1
    #             if not data['success']:
    #                 data['error'] = 'Transaction failed'
    #             
    #             time.sleep(0.05)
    #             
    #         except Exception as e:
    #             print(f"❌ Error processing transaction for '{data['record_id']}': {e}")
    #             data['success'] = False
    #             data['error'] = str(e)
    #     
    #     successful = sum(1 for data in tx_data if data['success'] is True)
    #     failed = sum(1 for data in tx_data if data['success'] is False)
    #     
    #     print(f"✅ Successful: {successful} | ❌ Failed: {failed} | 📋 Total: {len(tx_data)}")
    #     
    #     return tx_data
            

    def batch_read_records_contract(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Read multiple records in batch.
        """
        try:
            return self.contract.functions.batchReadRecords(record_ids).call()
        except Exception as e:
            print(f"Error reading records: {e}")
            return []

    def batch_create_score_records_contract(self, record_ids: List[str], logger_ids: List[str], record_types: List[str], scores: List[int], timestamp: str, gas_limit: int = 1000000) -> Optional[Dict[str, Any]]:
        """
        Create multiple records with scores in batch.
        
        Args:
            record_ids: List of record IDs
            logger_ids: List of logger IDs
            record_types: List of record types
            scores: List of scores
            timestamp: Timestamp string
            gas_limit: Gas limit for the transaction
            
        Returns:
            Dict: Transaction receipt or None if failed
        """
        try:
            transaction = self._build_transaction(
                self.contract.functions.batchCreateScoreRecords(record_ids, logger_ids, record_types, scores, timestamp),
                gas_limit=gas_limit
            )
            
            # Send transaction
            receipt = self._send_transaction(transaction)
            
            try:
                # check the event BatchCreatedcoreRecords
                events = self.get_events("BatchCreatedcoreRecords")
                for event in events:
                    if event['args']['timestamp'] == timestamp:
                        if event['args']['successCount'] == len(record_ids):
                            print(f"✅ {event['args']['successCount']} out of {len(record_ids)} records created successfully")
                            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
                            print(f"Gas used: {receipt['gasUsed']}")
                        elif event['args']['successCount'] == 0:
                            print(f"❌ No records created")
                            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
                            print(f"Gas used: {receipt['gasUsed']}")
                        else:
                            print(f"⚠️ {event['args']['successCount']} out of {len(record_ids)} records created successfully")
                            print(f"Transaction hash: {receipt['transactionHash'].hex()}")
                            print(f"Gas used: {receipt['gasUsed']}")
                        return receipt
            except Exception as e:
                print(f"❌ Error checking event BatchCreatedcoreRecords: {e}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating batch score records: {e}")
            return None
    
    def create_scores_for_sources(self, sources: List[str], score_types: List[str] = ["reliability", "usefulness"], score: int = 10000, gas_limit: int = 3000000) -> Dict[str, int]:
        """
        Create scores for multiple sources.
        """
        try:
            # source_ids = [f"{source}:{score_type}" for source in sources for score_type in score_types]
            timestamp = datetime.now().isoformat()
            record_ids = []
            logger_ids = []
            record_types = []
            for source in sources:
                for score_type in score_types:
                    record_ids.append(f"{source}:{score_type}")
                    logger_ids.append("source")
                    record_types.append(score_type)
            self.batch_create_score_records_contract(record_ids, logger_ids, record_types, [score] * len(record_ids), timestamp, gas_limit)
        except Exception as e:
            print(f"Error creating scores for sources: {e}")
            return {}

    def get_scores_for_sources(self, sources: List[str], score_types: List[str] = ["reliability", "usefulness"]) -> Dict[str, int]:
        """
        Compose on-chain records to a dictionary of scores for each source.
        Args:
            sources: List of source names
            score_types: List of score types to read, on-chain record id is in the format of "source_name:score_type"
        Returns:
            Dict[str, int]: Dictionary of scores for each source
        """
        try:
            source_ids = [f"{source}:{score_type}" for source in sources for score_type in score_types]
            records = self.batch_read_records_contract(source_ids)
            data_sources = {}
            for record in records:
                if record[0] == "":
                    continue
                source_name = record[0].split(":")[0]
                score_type = record[0].split(":")[1]
                if source_name not in data_sources:
                    data_sources[source_name] = {}
                if score_type != record[2]:
                    raise ValueError(f"Score type mismatch for source {source_name}: {record[0]} != {record[2]}")
                data_sources[source_name][score_type] = record[7]
            return data_sources
        except Exception as e:
            print(f"Error reading scores for sources: {e}")
            return {}

    def update_scores_for_sources(self, update_info, gas_limit: int = 3000000) -> Dict[str, int]:
        """
        Update scores for multiple sources.
        Args:
            update_info: List of dict, each dict contains (source_name, score_type, score, is_delta, info)
            gas_limit: Gas limit for the transaction
        Returns:
            Dict[str, int]: Dictionary of scores for each source
        """
        try:
            update_info_contract = []
            for update in update_info:
                record_id = f"{update['source_name']}:{update['score_type']}"
                score = update['score']
                is_delta = update['is_delta']
                info = update['info']
                update_info_contract.append((record_id, score, is_delta, info))
            self.batch_update_record_scores_contract(update_info_contract, gas_limit)
        except Exception as e:
            print(f"Error updating scores for sources: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Example configuration for local Hardhat network
    CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
    NETWORK_URL = "http://127.0.0.1:8545"
    PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    
    try:
        # Initialize the class
        draglog = DragLogSol(
            contract_address=CONTRACT_ADDRESS,
            network_url=NETWORK_URL,
            private_key=PRIVATE_KEY
        )
        
        # Print contract status
        draglog.print_contract_status()
        
        # Example: Create a new reliability record
        print("\n=== Creating Test Record ===")
        receipt = draglog.create_reliability_record(
            "python-test-source",
            "python-test-digest",
            "python-test-reserved"
        )
        
        if receipt:
            # Read the created record
            record = draglog.read_reliability_record("python-test-source")
            if record:
                print(f"Created record score: {record['reliabilityScore']}")
            
            # Update the score
            print("\n=== Updating Score ===")
            update_receipt = draglog.update_reliability_score(
                "python-test-source",
                -5,
                True,
                "python-test-update"
            )
            
            if update_receipt:
                updated_record = draglog.read_reliability_record("python-test-source")
                if updated_record:
                    print(f"Updated record score: {updated_record['reliabilityScore']}")

        # Example: Create a usefulness record
        print("\n=== Creating Usefulness Record ===")
        usefulness_receipt = draglog.create_usefulness_record(
            "python-usefulness-test",
            "python-logger",
            "test input data",
            "test input source",
            "test output data",
            "test output destination",
            str(int(time.time())),
            "python usefulness test"
        )
        
        if usefulness_receipt:
            # Read the created usefulness record
            usefulness_record = draglog.read_usefulness_record("python-usefulness-test")
            if usefulness_record:
                print(f"Created usefulness record: {usefulness_record['recordType']}")

        # Example: Create a general record
        print("\n=== Creating General Record ===")
        general_receipt = draglog.create_record(
            "python-general-test",
            "python-logger",
            "log",
            "general input data",
            "general input source",
            "general output data",
            "general output destination",
            85,
            str(int(time.time())),
            "python general test"
        )
        
        if general_receipt:
            # Read the created general record
            general_record = draglog.read_record("python-general-test")
            if general_record:
                print(f"Created general record: {general_record['recordType']} with score {general_record['score']}")
        
        # Example: Batch update multiple reliability scores
        print("\n=== Batch Update Test ===")
        
        # First create some test reliability records for batch updating
        test_sources = [
            ("batch-test-1", "digest-1", "reserved-1"),
            ("batch-test-2", "digest-2", "reserved-2"),
            ("batch-test-3", "digest-3", "reserved-3")
        ]
        
        # Create the test records individually
        print("Creating test records for batch update...")
        successful_creates = []
        for data_source_id, digest, reserved in test_sources:
            receipt = draglog.create_reliability_record(data_source_id, digest, reserved)
            if receipt:
                successful_creates.append(data_source_id)
        
        if successful_creates:
            # Prepare batch update data: (data_source_id, score_delta, is_delta, info)
            batch_updates = [
                ("batch-test-1", -10, True, "Batch decrease by 10"),
                ("batch-test-2", 15, True, "Batch increase by 15"),
                ("batch-test-3", 50, False, "Batch set to absolute 50")
            ]
            
            # Perform batch updates
            batch_results = draglog.batch_update_reliability_scores(
                batch_updates,
                timeout=60,  # Custom timeout
                poll_latency=0.3  # Custom polling interval
            )
            
            # Show results
            print(f"\nBatch update completed: {len([r for r in batch_results if r['success']])} successful")
            
            # Verify the updates
            for source_id in ["batch-test-1", "batch-test-2", "batch-test-3"]:
                record = draglog.read_reliability_record(source_id)
                if record:
                    print(f"  {source_id}: score = {record['reliabilityScore']}")

        # Example: New Contract-Based Batch Update (more efficient)
        print("\n=== Contract-Based Batch Update Test ===")
        
        # Create some more test records for the contract batch update
        contract_test_sources = [
            ("contract-batch-1", "digest-c1", "reserved-c1"),
            ("contract-batch-2", "digest-c2", "reserved-c2"),
            ("contract-batch-3", "digest-c3", "reserved-c3"),
            ("contract-batch-4", "digest-c4", "reserved-c4")
        ]
        
        # Create the test records individually
        print("Creating test records for contract batch update...")
        successful_contract_creates = []
        for data_source_id, digest, reserved in contract_test_sources:
            receipt = draglog.create_reliability_record(data_source_id, digest, reserved)
            if receipt:
                successful_contract_creates.append(data_source_id)
        
        if successful_contract_creates:
            # Prepare contract batch update data: (data_source_id, score, is_delta, info)
            contract_batch_updates = [
                ("contract-batch-1", -20, True, "Contract batch decrease by 20"),
                ("contract-batch-2", 25, True, "Contract batch increase by 25"),
                ("contract-batch-3", 75, False, "Contract batch set to absolute 75"),
                ("contract-batch-4", -5, True, "Contract batch decrease by 5")
            ]
            
            # Perform contract-based batch update (single transaction)
            print("Performing contract-based batch update (single transaction)...")
            contract_batch_receipt = draglog.batch_update_reliability_scores_contract(
                contract_batch_updates,
                gas_limit=2500000  # Higher gas limit for batch operation
            )
            
            if contract_batch_receipt:
                print(f"✅ Contract batch update successful!")
                print(f"Gas used: {contract_batch_receipt['gasUsed']}")
                
                # Verify the contract batch updates
                print("\nVerifying contract batch updates:")
                for source_id in ["contract-batch-1", "contract-batch-2", "contract-batch-3", "contract-batch-4"]:
                    record = draglog.read_reliability_record(source_id)
                    if record:
                        print(f"  {source_id}: score = {record['reliabilityScore']}")

        # Example: General record operations
        print("\n=== General Record Operations Test ===")
        
        # Create some test records for general operations
        general_test_records = [
            ("general-test-1", "python-logger", "log", "input1", "source1", "output1", "dest1", 90, str(int(time.time())), "test1"),
            ("general-test-2", "python-logger", "feedback", "input2", "source2", "output2", "dest2", 75, str(int(time.time())), "test2"),
            ("general-test-3", "python-logger", "usefulness", "input3", "source3", "output3", "dest3", 88, str(int(time.time())), "test3")
        ]
        
        # Create the test records individually
        print("Creating test records for general operations...")
        successful_general_creates = []
        for record_data in general_test_records:
            record_id, logger_id, record_type, input_data, input_from, output, output_to, score, timestamp, reserved = record_data
            receipt = draglog.create_record(record_id, logger_id, record_type, input_data, input_from, output, output_to, score, timestamp, reserved)
            if receipt:
                successful_general_creates.append(record_id)
        
        if successful_general_creates:
            # Test general record score updates
            print("\nTesting general record score updates...")
            general_score_updates = [
                ("general-test-1", -5, True, "General decrease by 5"),
                ("general-test-2", 10, True, "General increase by 10"),
                ("general-test-3", 95, False, "General set to absolute 95")
            ]
            
            # # Perform general batch score updates (individual transactions) - commented out
            # general_batch_results = draglog.batch_update_record_scores(
            #     general_score_updates,
            #     timeout=60
            # )
            # successful_individual = [r for r in general_batch_results if r['success']]
            # print(f"Individual batch updates: {len(successful_individual)}/{len(general_score_updates)} successful")
            
            # Perform general batch score updates (single contract transaction)
            general_batch_results = draglog.batch_update_record_scores_contract(
                general_score_updates,
                gas_limit=2500000
            )
            
            if general_batch_results:
                print(f"✅ General batch update successful!")
                
                # Verify the updates
                print("\nVerifying general record updates:")
                for record_id in ["general-test-1", "general-test-2", "general-test-3"]:
                    record = draglog.read_record(record_id)
                    if record:
                        print(f"  {record_id}: type={record['recordType']}, score={record['score']}")

        # Example: Contract statistics and utilities
        print("\n=== Contract Statistics and Utilities ===")
        
        # Get all records
        all_records = draglog.get_all_records()
        print(f"Total records in contract: {len(all_records)}")
        
        # Get all record IDs
        all_record_ids = draglog.get_all_record_ids()
        print(f"All record IDs: {all_record_ids[:5]}...")  # Show first 5
        
        # Get record count
        record_count = draglog.get_record_count()
        print(f"Record count: {record_count}")
        
        # Check record types
        record_types = ["reliability", "log", "feedback", "usefulness"]
        for record_type in record_types:
            has_type = draglog.has_record_type(record_type)
            print(f"Has {record_type} records: {'✓' if has_type else '✗'}")
        
    except Exception as e:
        print(f"Error: {e}") 