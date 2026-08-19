import yaml
from typing import Dict, Any, Optional

class HarmonizerValidator:
    def __init__(self, contract_path: str):
        with open(contract_path, "r") as f:
            contract = yaml.safe_load(f)
            self.allowed_labels = set(contract.get("allowed_canonical_labels", []))
            self.reserved_actions = set(contract.get("reserved_actions", []))

    def process_and_validate(self, dataset_name: str, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        canonical_label = record.get("canonical_label")

        # filter excluded or non-pertinent records
        if canonical_label in self.reserved_actions or canonical_label == "EXCLUDE":
            return None

        # enforce canonical contract schema
        if canonical_label not in self.allowed_labels:
            raise ValueError(
                f"taxonomy violation in dataset '{dataset_name}': "
                f"label '{canonical_label}' is not declared in canonical_taxonomy.yaml"
            )

        record["dataset_source"] = dataset_name
        return record