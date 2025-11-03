from typing import Dict, List
import pandas as pd
import yaml
import json


def read_csv_file(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    return df


def read_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def read_jsonl(file_path: str, n: int = None) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if n is not None:
        lines = lines[:n]
    return [json.loads(line.strip()) for line in lines]
