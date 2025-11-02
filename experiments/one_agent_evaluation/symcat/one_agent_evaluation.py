import os
import random
from datetime import datetime
import pandas as pd
import logging
import yaml

from experiments.one_agent_evaluation.llm.llm_client import LLMClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'/mnt/e/project/LLM/baseline/demo/test_symcat/log/symcat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)


# 定义数据类（映射 CSV 列）
class PatientRecord:
    def __init__(
        self,
        patient_id,
        age,
        gender,
        race,
        ethnicity,
        num_symptoms,
        pathology,
        symptoms,
        options,
    ):
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.race = race
        self.ethnicity = ethnicity
        self.num_symptoms = num_symptoms
        self.pathology = pathology
        self.symptoms = self.parse_symptoms(symptoms)
        self.options = options

    def parse_symptoms(self, symptoms_str):
        """
        将症状字符串解析为字典: {'症状名': 数值}
        例如: "Difficulty breathing:35::::::35;Increased heart rate:34::::::34"
        -> {'Difficulty breathing': 35, 'Increased heart rate': 34}
        """
        symptom_dict = {}
        for item in symptoms_str.split(";"):
            if not item:
                continue
            parts = item.split(":")
            symptom_name = parts[0]
            try:
                value = int(parts[1])
            except ValueError:
                value = None
            symptom_dict[symptom_name] = value
        return symptom_dict


path = "/mnt/e/project/LLM/baseline/ChallengeClinicalQA/symcat_style_dataset.jsonl"

import json

records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        record = PatientRecord(
            patient_id=data.get("PATIENT"),
            age=data.get("Age"),
            gender=data.get("Gender"),
            race=data.get("Race"),
            ethnicity=data.get("Ethnicity"),
            num_symptoms=data.get("NUM_SYMPTOMS"),
            pathology=data.get("PATHOLOGY"),
            symptoms=data.get("Symptoms"),
            options=data.get("Options"),
        )
        records.append(record)

# 转换为 DataFrame，方便后续处理
df = pd.DataFrame(
    [
        {
            "PatientID": r.patient_id,
            "Age": r.age,
            "Gender": r.gender,
            "Race": r.race,
            "Ethnicity": r.ethnicity,
            "NumSymptoms": r.num_symptoms,
            "Pathology": r.pathology,
            "Symptoms": r.symptoms,
            "Options": r.options,
        }
        for r in records
    ]
)


def build_prompt(data: PatientRecord, dataset_name: str) -> str:
    path = "prompt.yaml"
    with open(path, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)
    prompt_template = prompt_data[dataset_name]["prompt"]
    role = prompt_data[dataset_name]["role"]
    content = prompt_data[dataset_name]["content"]
    options_str = "\n".join(f"- {name}" for _, name in data.options.items())
    symptoms_str = "\n".join(
        f"- {symptom}({value})" for symptom, value in data.symptoms.items()
    )
    patient_dict = {
        "age": data.age,
        "gender": "male" if data.gender == "M" else "female",
        "race": data.race,
        "ethnicity": data.ethnicity,
        "pathology": data.pathology,
        "symptoms": symptoms_str,
        "options": options_str,
    }
    for key, value in patient_dict.items():
        prompt_template = prompt_template.replace(f"{{{key}}}", str(value))
    return role, content, prompt_template


if __name__ == "__main__":
    #     # 运行一些测试代码
    logging.info("开始运行....")
    llm_client = LLMClient(
        model_name=os.getenv("MODEL_NAME"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_base=os.getenv("DASHSCOPE_BASE_URL"),
    )
    llm_client.init_client()
    random.seed()
    random_records = random.sample(records, 50)
    right_count = 0
    for i, record in enumerate(random_records, start=1):
        print(f"======第{i}个问题=====")
        role, content, prompt = build_prompt(record, "symcat")
        response = llm_client.ask_model(role="system", content=content, prompt=prompt)
        print(f"pathology:{record.pathology}\n")
        print(f"response:{response}\n")
        if response == record.pathology:
            right_count += 1
            print("\n正确\n")
        else:
            print("\n错误\n")
    accuracy = right_count / len(random_records)
    logging.info(f"总体准确率为:{accuracy:.2%}")
