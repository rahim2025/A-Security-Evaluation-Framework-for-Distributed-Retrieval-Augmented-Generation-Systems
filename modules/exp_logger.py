import csv
import os
from typing import Any, Dict, List, Set

import yaml
from loguru import logger

class ExpLogger:
    """
    Logs data to the local file system in CSV and YAML formats, supporting multiple CSV and YAML loggers.

    - CSV logs are saved to `os.path.join(root_dir, log_dir_name, version)`.
    - YAML configuration files are saved to `os.path.join(root_dir, log_dir_name, version)`.

    Args:
        root_dir: The root directory for all experiments (e.g., './experiments').
        log_dir_name: The name of the subdirectory within `root_dir` for this experiment (e.g., 'run_1').
                       If `None`, logs will be stored directly in `root_dir`.
        version: Experiment version. If `None`, the logger will automatically assign the next available version
                 by inspecting the `root_dir/log_dir_name` directory. If specified, and a directory with that
                 version already exists, log files within that directory will be overwritten.

    Example:
        ```python
        exp_logger = ExpLogger(root_dir="./experiments", log_dir_name="run_1")

        # Create and use a CSV logger for metrics
        metrics_logger = exp_logger.get_csv_logger("metrics")
        metrics_logger.log({"loss": 0.235, "accuracy": 0.75})
        metrics_logger.save()

        # Create and use another CSV logger for test case details
        test_case_logger = exp_logger.get_csv_logger("test_cases")
        test_case_logger.log({"case_id": "TC001", "result": "Pass"})
        test_case_logger.save()

        # Log configurations to YAML using a specific name
        config_logger = exp_logger.get_yaml_logger("model_config")
        config_logger.log({"model": "resnet", "lr": 0.01})
        config_logger.save() # Writes to experiments/my_experiment/version_0/model_config.yaml

        # Log dataset information to another YAML file
        dataset_logger = exp_logger.get_yaml_logger("dataset_info")
        dataset_logger.log({"name": "ImageNet", "size": 1281167})
        dataset_logger.save() # Writes to experiments/my_experiment/version_0/dataset_info.yaml
        ```
    """

    def __init__(
        self,
        root_dir: str = "./",
        log_dir_name: str = "logs"
    ):
        super().__init__()
        self._root_dir = os.path.abspath(root_dir)
        self._log_dir_name = log_dir_name
        self._version = None
        self._csv_loggers: Dict[str, _CSVWriter] = {}
        self._yaml_loggers: Dict[str, _YAMLWriter] = {}  # Dictionary to store multiple YAML loggers

    @property
    def version(self) -> int:
        """Gets the experiment version."""
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def experiment_dir(self) -> str:
        """The directory path for this experiment's logs."""
        version_name = f"version_{self.version}"
        return os.path.join(self._root_dir, self._log_dir_name, version_name)

    def get_csv_logger(self, logger_name: str) -> "_CSVWriter":
        """
        Returns a CSV logger for a specific type of data.

        Args:
            logger_name: The name to identify the CSV logger (e.g., "metrics", "test_cases"). This will be used as the base name for the CSV file.

        Returns:
            The `_CSVWriter` instance for logging data of the specified type.
        """
        if logger_name not in self._csv_loggers:
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._csv_loggers[logger_name] = _CSVWriter(log_dir=self.experiment_dir, logger_name=logger_name)
        return self._csv_loggers[logger_name]

    def get_yaml_logger(self, logger_name: str) -> "_YAMLWriter":
        """
        Returns a YAML logger for a specific type of data.

        Args:
            logger_name: The name to identify the YAML logger (e.g., "config", "dataset"). This will be used as the base name for the YAML file.

        Returns:
            The `_YAMLWriter` instance for logging data of the specified type.
        """
        if logger_name not in self._yaml_loggers:
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._yaml_loggers[logger_name] = _YAMLWriter(log_dir=self.experiment_dir, logger_name=logger_name)
        return self._yaml_loggers[logger_name]

    def _get_next_version(self) -> int:
        """Determines the next available version number."""
        experiment_root = os.path.join(self._root_dir, self._log_dir_name)

        if not os.path.isdir(experiment_root):
            return 0

        existing_versions = []
        for dir_name in os.listdir(experiment_root):
            full_path = os.path.join(experiment_root, dir_name)
            if os.path.isdir(full_path) and dir_name.startswith("version_"):
                try:
                    version_num = int(dir_name.split("_")[1])
                    existing_versions.append(version_num)
                except ValueError:
                    pass

        return max(existing_versions, default=-1) + 1

class _CSVWriter:
    """CSV writer for ExpLogger."""

    def __init__(self, log_dir: str, logger_name: str) -> None:
        """
        Initializes the CSV writer.

        Args:
            log_dir: The directory where the CSV log file will be stored.
            logger_name: The base name for the CSV file (e.g., 'metrics', 'test_cases').
        """
        self.data_buffer: List[Dict[str, float]] = []
        self.fieldnames: List[str] = []

        self.log_dir = log_dir
        self.log_file_name = f"{logger_name}.csv"
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name)

    def log(self, data_dict: Dict[str, Any]) -> None:
        """Appends data to the buffer for CSV logging."""
        self.data_buffer.append(data_dict)

    def save(self) -> None:
        """Writes the buffered data to the CSV file."""
        if not self.data_buffer:
            logger.warning(f"No data to save for {self.log_file_name}.")
            return

        new_fieldnames = self._update_fieldnames()
        file_exists = os.path.isfile(self.log_file_path)

        if new_fieldnames and file_exists:
            self._rewrite_csv_with_new_header(self.fieldnames)

        with open(self.log_file_path, mode=("a" if file_exists else "w"), errors='surrogatepass', newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames, escapechar='\\')
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.data_buffer)

        self.data_buffer = []

    def _update_fieldnames(self) -> Set[str]:
        """Updates the fieldnames based on the keys in the data buffer."""
        current_fieldnames = set().union(*self.data_buffer)
        new_fieldnames = current_fieldnames - set(self.fieldnames)
        self.fieldnames.extend(new_fieldnames)
        self.fieldnames.sort()
        return new_fieldnames

    def _rewrite_csv_with_new_header(self, fieldnames: List[str]) -> None:
        """Rewrites the CSV file with a new header."""
        with open(self.log_file_path, "r", errors='surrogatepass', newline="") as csvfile:
            reader = csv.DictReader(csvfile, escapechar='\\')
            original_data = list(reader)

        with open(self.log_file_path, "w", errors='surrogatepass', newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar='\\')
            writer.writeheader()
            writer.writerows(original_data)


class _YAMLWriter:
    """YAML writer for ExpLogger."""

    def __init__(self, log_dir: str, logger_name: str) -> None:
        """
        Initializes the YAML writer.

        Args:
            log_dir: The directory where the YAML log file will be stored.
            logger_name: The base name for the YAML file (e.g., 'config', 'dataset_info').
        """
        self.data_buffer: Dict[str, Any] = {}  # Use a buffer to accumulate data
        self.log_dir = log_dir
        self.log_file_name = f"{logger_name}.yaml"
        self.log_file_path = os.path.join(self.log_dir, self.log_file_name)

    def log(self, data_dict: Dict[str, Any]) -> None:
        """
        Accumulates data for YAML logging.

        Args:
            data_dict: A dictionary containing the data to log.
        """
        self.data_buffer.update(data_dict)  # Update the buffer with new data

    def save(self) -> None:
        """Writes the accumulated data to the YAML file."""
        if not self.data_buffer:
            logger.warning(f"No data to save for {self.log_file_name}.")
            return

        with open(self.log_file_path, "w") as yamlfile:
            yaml.dump(self.data_buffer, yamlfile, default_flow_style=False)

        self.data_buffer = {}  # Clear the buffer after saving
