import json
import os
import random

class DataLoader:
    def __init__(self, dataset_name, split="dev"):
        self.dataset_name = dataset_name.lower()
        self.split = split

        if self.dataset_name == "spider":
            self.data_path = f"datasets/spider/spider_{split}_with_schema_new_t1.jsonl"
        elif self.dataset_name == "bird":
            self.data_path = f"datasets/bird/bird_subset_with_schema_new_t1.jsonl"
        else:
            raise ValueError("Unsupported dataset")

        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.data_path, "r") as f:
            for line in f:
                item = json.loads(line)

                # Use schema_new instead of schema
                if "schema_new" in item:
                    item["schema"] = item["schema_new"]

                data.append(item)

        return data

    def get_all(self):
        return self.data

    def get_sample(self, n=10, seed=42):
        random.seed(seed)
        return random.sample(self.data, min(n, len(self.data)))

    def get_by_db(self, db_id):
        return [x for x in self.data if x.get("db_id") == db_id]

    def __len__(self):
        return len(self.data)