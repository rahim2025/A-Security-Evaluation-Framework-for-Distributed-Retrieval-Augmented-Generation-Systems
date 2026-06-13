import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from ctypes import c_float as float32
import json
import os
import hashlib


@dataclass
class LogRecord:
    logID: str  # logID or dataSourceID
    loggerID: str
    type: str  # log or reliability
    input: str
    inputFrom: str
    output: str
    outputTo: str
    reliabilityScore: float  # -1 for log
    timestamp: str
    reserved: str

@dataclass
class LogRecordInput:
    logID: str  # logID or dataSourceID
    loggerID: str
    input: str
    inputFrom: str
    output: str
    outputTo: str
    timestamp: str
    reserved: str
    type: str = "log"  
    reliabilityScore: float32 = -1

@dataclass
class LogRecordHistory:
    record: Optional[LogRecord]
    timestamp: str
    txID: str
    isDelete: bool

class DragLogClient:
    def __init__(self, base_url: str = "http://localhost:8080", local: bool = False, log_file: str = "logs/draglog.jsonl", reliability_history_path: str = "logs/reliability_history.jsonl"):
        """Initialize the DragLog client.
        
        Args:
            base_url: Base URL of the DragLog API server
            local: Whether to store records locally without making API requests
            log_file: Path to the log file for local mode
        """
        self.base_url = base_url.rstrip('/')
        self.local = local
        self.log_file = log_file
        self.reliability_history_path = reliability_history_path
        if self.local:
            # Initialize log file if it doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    json.dump([], f)
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the API server.
        
        Args:
            method: HTTP method (GET, POST, PUT)
            endpoint: API endpoint
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data as dictionary or empty dict if request fails
        """
        if self.local:
            # In local mode, return empty dict for GET requests
            if method == 'GET':
                return {'records': []}
            # For other methods, just write to log file and return empty dict
            if 'json' in kwargs:
                self._write_to_log_file(kwargs['json'], f'{method.lower()}_{endpoint.replace("/", "_")}')
            return {}
            
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Return empty dict if response is empty
            if not response.content:
                return {}
                
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Failed to connect to server at {self.base_url}: {str(e)}")
            return {}
    
    def init_ledger(self) -> None:
        """Initialize the ledger."""
        self._make_request('GET', '/init-ledger')
    
    def init_sources(self, sources: Dict) -> None:
        """Initialize the sources, clean the reliability history file and log file if it exists."""
        if os.path.exists(self.reliability_history_path):
            os.remove(self.reliability_history_path)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if self.local:
            self.reliability_scores = {source_id: sources[source_id]['reliability'] for source_id in sources}
            self.dump_reliability_records(last_feedback="")
        # create reliability records in batch
        records_batch = []
        for source_id in sources:
            records_batch.append(LogRecord(
                logID=source_id,
                loggerID="",
                type="reliability",
                input="",
                inputFrom="",
                output="",
                outputTo="",
                reliabilityScore=sources[source_id]['reliability'],
                timestamp=datetime.now().isoformat(),
                reserved=""
            ))
        self.create_reliability_records_batch(records_batch)
    
    def dump_reliability_records(self, last_feedback: str) -> None:
        """Dump the reliability records to a jsonl file."""
        with open(self.reliability_history_path, 'a') as f:
            # write the last feedback
            f.write(last_feedback + '\n')
            # write the reliability scores
            f.write(json.dumps(self.reliability_scores) + '\n')
    
    def _write_to_log_file(self, record: Dict[str, Any], operation: str) -> None:
        """Write a record to the log file in JSONL format.
        
        Args:
            record: The record to write
            operation: The operation performed (create, update, etc.)
        """
        if not self.local:
            return
            
        try:
            log_record = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'record': record
            }
            
            # Append the record as a single line
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_record) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def _read_log_records(self) -> List[Dict[str, Any]]:
        """Read all records from the log file.
        
        Returns:
            List of records from the log file
        """
        if not os.path.exists(self.log_file):
            return []
            
        records = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Error reading log file: {e}")
        return records

    def create_log_record(self, record: LogRecordInput) -> None:
        """Create a new log record.
        
        Args:
            record: LogRecordInput object containing the record details
        """
        record.type = "log"
        record_dict = record.__dict__
        self._make_request('POST', '/create-log-record', json=record_dict)
    
    def create_feedback_record(self, record: LogRecordInput) -> None:
        """Create a new feedback record.
        
        Args:
            record: LogRecordInput object containing the record details
        """
        record.type = "feedback"
        record_dict = record.__dict__
        self._make_request('POST', '/create-feedback-record', json=record_dict)
    
    def create_reliability_record(self, data_source_id: str, digest: str, reserved: str) -> None:
        """Create a new reliability record.
        
        Args:
            data_source_id: ID of the data source
            digest: Digest value
            reserved: Reserved value
        """
        record_dict = {"dataSourceID": data_source_id, "digest": digest, "reserved": reserved}
        self._make_request('POST', '/create-reliability-record', json=record_dict)

    def create_reliability_records_batch(self, records: List[LogRecord]) -> None:
        """Create a new reliability record in batch.
        
        Args:
            records: List of LogRecord objects
        """
        record_dict = {"recordsJSON": json.dumps([record.__dict__ for record in records])}
        self._make_request('POST', '/create-reliability-records-batch', json=record_dict)

    def create_reliability_record_async(self, data_source_id: str, digest: str, reserved: str) -> None:
        """Create a new reliability record asynchronously.
        
        Args:
            data_source_id: ID of the data source
            digest: Digest value
            reserved: Reserved value
        """
        record_dict = {"dataSourceID": data_source_id, "digest": digest, "reserved": reserved}
        self._make_request('POST', '/create-reliability-record-async', json=record_dict)
    
    def get_all_log_records(self) -> List[LogRecord]:
        """Get all log records.
        
        Returns:
            List of LogRecord objects
        """
        if self.local:
            records = self._read_log_records()
            return [LogRecord(**record['record']) for record in records 
                   if record['operation'].startswith('create_log_record')]
            
        response = self._make_request('GET', '/get-all-log-records')
        return [LogRecord(**record) for record in response['records']]
    
    def get_all_feedback_records(self) -> List[LogRecord]:
        """Get all feedback records.
        
        Returns:
            List of LogRecord objects
        """
        if self.local:
            records = self._read_log_records()
            return [LogRecord(**record['record']) for record in records 
                   if record['operation'].startswith('create_feedback_record')]
            
        response = self._make_request('GET', '/get-all-feedback-records')
        return [LogRecord(**record) for record in response['records']]
    
    
    def get_all_reliability_records(self) -> List[LogRecord]:
        """Get all reliability records.
        
        Returns:
            List of LogRecord objects
        """
        if self.local:
            records = self._read_log_records()
            return [LogRecord(**record['record']) for record in records 
                   if record['operation'].startswith('create_reliability_record')]
            
        response = self._make_request('GET', '/get-all-reliability-records')
        return [LogRecord(**record) for record in response['records']]
    
    def get_log_record(self, log_id: str) -> LogRecord:
        """Get a specific log record.
        
        Args:
            log_id: ID of the log record
            
        Returns:
            LogRecord object
        """
        response = self._make_request('GET', f'/get-log-record/{log_id}')
        return LogRecord(**response['records'][0])
    
    def get_reliability_record(self, data_source_id: str) -> LogRecord:
        """Get a specific reliability record.
        
        Args:
            data_source_id: ID of the data source
            
        Returns:
            LogRecord object
        """
        response = self._make_request('GET', f'/get-reliability-record/{data_source_id}')
        return LogRecord(**response['records'][0])
    
    def get_reliability_score(self, data_source_id: str) -> float:
        """Get the reliability score of a data source."""
        if self.local:
            return self.reliability_scores[data_source_id]
        else:
            return self.get_reliability_record(data_source_id).reliabilityScore

    def update_reliability_record(self, data_source_id: str, reliability_score: float, is_delta: bool, info: str) -> None:
        """Update a reliability record's score.
        
        Args:
            data_source_id: ID of the data source
            reliability_score: New reliability score
            is_delta: Whether the score is a delta or absolute value
            info: Info
        """
        if self.local:
            if is_delta:
                self.reliability_scores[data_source_id] += reliability_score
            else:
                self.reliability_scores[data_source_id] = reliability_score
        record_dict = {"reliabilityScore": reliability_score, "isDelta": is_delta, "info": info}
        self._make_request('PUT', f'/update-reliability-record/{data_source_id}', json=record_dict)
    
    def get_history_for_record(self, log_id: str) -> List[LogRecordHistory]:
        """Get the history of a record.
        
        Args:
            log_id: ID of the log record
            
        Returns:
            List of LogRecordHistory objects
        """
        response = self._make_request('GET', f'/get-history-for-record/{log_id}')
        return [LogRecordHistory(
            record=LogRecord(**history['record']) if history['record'] else None,
            timestamp=history['timestamp'],
            txID=history['txID'],
            isDelete=history['isDelete']
        ) for history in response['history']]
    
from tqdm import tqdm
    
def replay_logs(base_url: str, log_file: str) -> None:
    """Replay the logs to the server.
    
    Args:
        base_url: Base URL of the target server
        log_file: Path to the log file to replay
    """
    # Create a new client instance for the target server
    target_client = DragLogClient(base_url=base_url, local=False)
    
    print(f"Replaying logs from {log_file} to {base_url}")
    
    with open(log_file, 'r') as f:
        line_count = 0
        for line in tqdm(f):
            line_count += 1
            try:
                log_record = json.loads(line.strip())
                operation = log_record.get('operation', '')
                record = log_record.get('record', {})
                
                # print(f"Processing line {line_count}: {operation}")
                
                if operation.startswith('post__create-feedback-record'):
                    # Create feedback record
                    feedback_record = LogRecordInput(
                        logID=record.get('logID', ''),
                        loggerID=record.get('loggerID', ''),
                        input=record.get('input', ''),
                        inputFrom=record.get('inputFrom', ''),
                        output=record.get('output', ''),
                        outputTo=record.get('outputTo', ''),
                        timestamp=record.get('timestamp', ''),
                        reserved=record.get('reserved', ''),
                        type=record.get('type', 'feedback'),
                        reliabilityScore=record.get('reliabilityScore', -1)
                    )
                    target_client.create_feedback_record(feedback_record)
                    # print(f"  ✓ Created feedback record: {record.get('logID', '')}")
                    
                elif operation.startswith('put__update-reliability-record'):
                    # Extract source ID from operation name
                    # Format: put__update-reliability-record_<source_id>
                    source_id = operation.replace('put__update-reliability-record_', '')
                    
                    # Update reliability record
                    reliability_score = record.get('reliabilityScore', 0.0)
                    is_delta = record.get('isDelta', False)
                    info = record.get('info', '')
                    
                    target_client.update_reliability_record(
                        data_source_id=source_id,
                        reliability_score=reliability_score,
                        is_delta=is_delta,
                        info=info
                    )
                    # print(f"  ✓ Updated reliability record for {source_id}: {reliability_score}")
                    
                elif operation.startswith('post__create-log-record'):
                    # Create log record
                    log_record_input = LogRecordInput(
                        logID=record.get('logID', ''),
                        loggerID=record.get('loggerID', ''),
                        input=record.get('input', ''),
                        inputFrom=record.get('inputFrom', ''),
                        output=record.get('output', ''),
                        outputTo=record.get('outputTo', ''),
                        timestamp=record.get('timestamp', ''),
                        reserved=record.get('reserved', ''),
                        type=record.get('type', 'log'),
                        reliabilityScore=record.get('reliabilityScore', -1)
                    )
                    target_client.create_log_record(log_record_input)
                    # print(f"  ✓ Created log record: {record.get('logID', '')}")
                    
                elif operation.startswith('post__create-reliability-record'):
                    # Create reliability record
                    data_source_id = record.get('dataSourceID', '')
                    digest = record.get('digest', '')
                    reserved = record.get('reserved', '')
                    
                    target_client.create_reliability_record(
                        data_source_id=data_source_id,
                        digest=digest,
                        reserved=reserved
                    )
                    # print(f"  ✓ Created reliability record: {data_source_id}")
                    
                elif operation.startswith('post__create-reliability-records-batch'):
                    # Create reliability records in batch
                    records_json = record.get('recordsJSON', '[]')
                    records_data = json.loads(records_json)
                    
                    records = []
                    for rec_data in records_data:
                        record_obj = LogRecord(
                            logID=rec_data.get('logID', ''),
                            loggerID=rec_data.get('loggerID', ''),
                            type=rec_data.get('type', 'reliability'),
                            input=rec_data.get('input', ''),
                            inputFrom=rec_data.get('inputFrom', ''),
                            output=rec_data.get('output', ''),
                            outputTo=rec_data.get('outputTo', ''),
                            reliabilityScore=rec_data.get('reliabilityScore', 0.0),
                            timestamp=rec_data.get('timestamp', ''),
                            reserved=rec_data.get('reserved', '')
                        )
                        records.append(record_obj)
                    
                    target_client.create_reliability_records_batch(records)
                    # print(f"  ✓ Created {len(records)} reliability records in batch")
                    
                else:
                    print(f"  ⚠ Unknown operation: {operation}")
                    
            except json.JSONDecodeError as e:
                print(f"  ✗ Error parsing JSON on line {line_count}: {e}")
            except Exception as e:
                print(f"  ✗ Error processing line {line_count}: {e}")
    
    print(f"Replay completed. Processed {line_count} lines.")

def replay_logs_with_options(base_url: str, log_file: str, 
                            dry_run: bool = False, 
                            start_line: int = 1, 
                            end_line: int = None,
                            filter_operations: List[str] = None) -> None:
    """Replay the logs to the server with additional options.
    
    Args:
        base_url: Base URL of the target server
        log_file: Path to the log file to replay
        dry_run: If True, only print what would be done without actually executing
        start_line: Line number to start replaying from (1-indexed)
        end_line: Line number to stop replaying at (inclusive, None for end of file)
        filter_operations: List of operation types to include (e.g., ['post__create-feedback-record'])
    """
    # Create a new client instance for the target server
    target_client = DragLogClient(base_url=base_url, local=False)
    
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"Replaying logs from {log_file} to {base_url} ({mode})")
    if start_line > 1 or end_line:
        print(f"Line range: {start_line} to {end_line or 'end'}")
    if filter_operations:
        print(f"Filtering operations: {filter_operations}")
    
    with open(log_file, 'r') as f:
        line_count = 0
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for line in f:
            line_count += 1
            
            # Skip lines outside the specified range
            if line_count < start_line:
                continue
            if end_line and line_count > end_line:
                break
            
            try:
                log_record = json.loads(line.strip())
                operation = log_record.get('operation', '')
                record = log_record.get('record', {})
                
                # Skip if operation is not in filter
                if filter_operations and not any(op in operation for op in filter_operations):
                    skipped_count += 1
                    continue
                
                print(f"Processing line {line_count}: {operation}")
                
                if operation.startswith('post__create-feedback-record'):
                    # Create feedback record
                    feedback_record = LogRecordInput(
                        logID=record.get('logID', ''),
                        loggerID=record.get('loggerID', ''),
                        input=record.get('input', ''),
                        inputFrom=record.get('inputFrom', ''),
                        output=record.get('output', ''),
                        outputTo=record.get('outputTo', ''),
                        timestamp=record.get('timestamp', ''),
                        reserved=record.get('reserved', ''),
                        type=record.get('type', 'feedback'),
                        reliabilityScore=record.get('reliabilityScore', -1)
                    )
                    
                    if not dry_run:
                        target_client.create_feedback_record(feedback_record)
                    print(f"  ✓ Created feedback record: {record.get('logID', '')}")
                    
                elif operation.startswith('put__update-reliability-record'):
                    # Extract source ID from operation name
                    source_id = operation.replace('put__update-reliability-record_', '')
                    
                    # Update reliability record
                    reliability_score = record.get('reliabilityScore', 0.0)
                    is_delta = record.get('isDelta', False)
                    info = record.get('info', '')
                    
                    if not dry_run:
                        target_client.update_reliability_record(
                            data_source_id=source_id,
                            reliability_score=reliability_score,
                            is_delta=is_delta,
                            info=info
                        )
                    print(f"  ✓ Updated reliability record for {source_id}: {reliability_score}")
                    
                elif operation.startswith('post__create-log-record'):
                    # Create log record
                    log_record_input = LogRecordInput(
                        logID=record.get('logID', ''),
                        loggerID=record.get('loggerID', ''),
                        input=record.get('input', ''),
                        inputFrom=record.get('inputFrom', ''),
                        output=record.get('output', ''),
                        outputTo=record.get('outputTo', ''),
                        timestamp=record.get('timestamp', ''),
                        reserved=record.get('reserved', ''),
                        type=record.get('type', 'log'),
                        reliabilityScore=record.get('reliabilityScore', -1)
                    )
                    
                    if not dry_run:
                        target_client.create_log_record(log_record_input)
                    print(f"  ✓ Created log record: {record.get('logID', '')}")
                    
                elif operation.startswith('post__create-reliability-record'):
                    # Create reliability record
                    data_source_id = record.get('dataSourceID', '')
                    digest = record.get('digest', '')
                    reserved = record.get('reserved', '')
                    
                    if not dry_run:
                        target_client.create_reliability_record(
                            data_source_id=data_source_id,
                            digest=digest,
                            reserved=reserved
                        )
                    print(f"  ✓ Created reliability record: {data_source_id}")
                    
                elif operation.startswith('post__create-reliability-records-batch'):
                    # Create reliability records in batch
                    records_json = record.get('recordsJSON', '[]')
                    records_data = json.loads(records_json)
                    
                    records = []
                    for rec_data in records_data:
                        record_obj = LogRecord(
                            logID=rec_data.get('logID', ''),
                            loggerID=rec_data.get('loggerID', ''),
                            type=rec_data.get('type', 'reliability'),
                            input=rec_data.get('input', ''),
                            inputFrom=rec_data.get('inputFrom', ''),
                            output=rec_data.get('output', ''),
                            outputTo=rec_data.get('outputTo', ''),
                            reliabilityScore=rec_data.get('reliabilityScore', 0.0),
                            timestamp=rec_data.get('timestamp', ''),
                            reserved=rec_data.get('reserved', '')
                        )
                        records.append(record_obj)
                    
                    if not dry_run:
                        target_client.create_reliability_records_batch(records)
                    print(f"  ✓ Created {len(records)} reliability records in batch")
                    
                else:
                    print(f"  ⚠ Unknown operation: {operation}")
                    skipped_count += 1
                    continue
                
                processed_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"  ✗ Error parsing JSON on line {line_count}: {e}")
                error_count += 1
            except Exception as e:
                print(f"  ✗ Error processing line {line_count}: {e}")
                error_count += 1
    
    print(f"\nReplay Summary:")
    print(f"  Total lines read: {line_count}")
    print(f"  Lines processed: {processed_count}")
    print(f"  Lines skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Mode: {mode}")

def analyze_log_file(log_file: str) -> Dict[str, Any]:
    """Analyze a log file and return statistics.
    
    Args:
        log_file: Path to the log file to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'total_lines': 0,
        'operations': {},
        'error_lines': [],
        'timestamp_range': {'start': None, 'end': None}
    }
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            analysis['total_lines'] += 1
            
            try:
                log_record = json.loads(line.strip())
                operation = log_record.get('operation', 'unknown')
                timestamp = log_record.get('timestamp', '')
                
                # Count operations
                if operation not in analysis['operations']:
                    analysis['operations'][operation] = 0
                analysis['operations'][operation] += 1
                
                # Track timestamp range
                if timestamp:
                    if analysis['timestamp_range']['start'] is None or timestamp < analysis['timestamp_range']['start']:
                        analysis['timestamp_range']['start'] = timestamp
                    if analysis['timestamp_range']['end'] is None or timestamp > analysis['timestamp_range']['end']:
                        analysis['timestamp_range']['end'] = timestamp
                        
            except json.JSONDecodeError:
                analysis['error_lines'].append(line_num)
            except Exception:
                analysis['error_lines'].append(line_num)
    
    return analysis

# Example usage:
if __name__ == "__main__":
    # Create client with custom server address
    client = DragLogClient("http://localhost:8080")
    
    # Initialize ledger
    client.init_ledger()
    
    # Create a log record
    record = LogRecord(
        logID="test123",
        loggerID="logger1",
        type="log",
        input="test input",
        inputFrom="source1",
        output="test output",
        outputTo="destination1",
        reliabilityScore=-1.0,
        timestamp=datetime.now().isoformat(),
        reserved=""
    )
    client.create_log_record(record)
    
    # Get all log records
    records = client.get_all_log_records()
    for record in records:
        print(f"Found record: {record.logID}")


def doc_to_sha256(doc):
    """
    Transform a document of any length to a SHA256 hash.
    
    Args:
        doc (str): The document text to hash
        
    Returns:
        str: SHA256 hash as a hexadecimal string
    """
    # Encode the document to bytes (SHA256 requires bytes input)
    doc_bytes = doc.encode('utf-8')
    
    # Create SHA256 hash object
    sha256_hash = hashlib.sha256()
    
    # Update the hash object with the document bytes
    sha256_hash.update(doc_bytes)
    
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

def log_record_to_json(record: LogRecord) -> str:
    """
    Convert a LogRecord dataclass instance to a JSON string.
    
    Args:
        record (LogRecord): The LogRecord instance to convert
        
    Returns:
        str: JSON string representation of the LogRecord
    """
    return json.dumps({
        "logID": record.logID,
        "loggerID": record.loggerID,
        "type": record.type,
        "input": record.input,
        "inputFrom": record.inputFrom,
        "output": record.output,
        "outputTo": record.outputTo,
        "reliabilityScore": record.reliabilityScore,
        "timestamp": record.timestamp,
        "reserved": record.reserved
    })

def log_record_to_dict(record: LogRecord) -> Dict[str, Any]:
    """
    Convert a LogRecord dataclass instance to a dictionary.
    
    Args:
        record (LogRecord): The LogRecord instance to convert
        
    Returns:
        Dict: Dictionary representation of the LogRecord
    """
    return {
        "logID": record.logID,
        "loggerID": record.loggerID,
        "type": record.type,
        "input": record.input,
        "inputFrom": record.inputFrom,
        "output": record.output,
        "outputTo": record.outputTo,
        "reliabilityScore": record.reliabilityScore,
        "timestamp": record.timestamp,
        "reserved": record.reserved
    }