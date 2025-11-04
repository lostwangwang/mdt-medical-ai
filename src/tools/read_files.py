import random
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


def read_jsonl(file_path: str, random_sample: int = None) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # 再处理随机抽取random_sample条（如果指定了random_sample）
    if random_sample is not None:
        # 确保抽取数量不超过现有条数
        sample_size = min(random_sample, len(lines))
        lines = random.sample(lines, sample_size)
    return [json.loads(line.strip()) for line in lines]
