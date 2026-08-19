from abc import ABC, abstractmethod
from typing import Generator, Dict, Any

class BaseDatasetLoader(ABC):
    """abstract base class for all dataset loader strategies."""

    # co-located label mapping: MUST map raw labels to canonical target labels
    label_mapping: Dict[str, str] = {}

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir

    @abstractmethod
    def acquire(self) -> None:
        """
        fetches raw data:
        - public: downloads files automatically.
        - id-based: rehydrates content via APIs.
        - private: validates required local files in raw_dir.
        """
        pass

    @abstractmethod
    def check_readiness(self) -> bool:
        """verifies if raw data files exist and are ready for processing."""
        pass

    @abstractmethod
    def load_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        yields unified dictionaries while preserving original splits:
        {
            "original_id": str,
            "text": str,
            "canonical_label": str,  # translated using self.label_mapping
            "original_split": "train" | "val" | "test" | "unassigned",
            "metadata": dict
        }
        """
        pass