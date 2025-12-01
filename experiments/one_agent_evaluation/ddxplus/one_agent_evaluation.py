import os
import random
from datetime import datetime
from typing import Any

import pandas as pd
import ast
import yaml
import logging

from evidences import decode_evidence
from experiments.one_agent_evaluation.llm.llm_client import LLMClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'ddxplus_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

# 定义数据类（映射 CSV 列）
class MedicalRecord:
    def __init__(
        self, age, differential_diagnosis, sex, pathology, evidences, initial_evidence
    ):
        self.age = age  # 年龄
        self.differential_diagnosis = differential_diagnosis  # 鉴别诊断列表（含概率）
        self.sex = sex  # 性别
        self.pathology = pathology  # 确诊疾病
        self.evidences = evidences  # 所有证据
        self.initial_evidence = initial_evidence  # 初始证据
        self.top_all_diagnosis = self.get_top_all_diagnosis()  # 计算所有诊断

    # 可选：自定义打印格式
    def __str__(self):
        return f"MedicalRecord(age={self.age}, differential_diagnosis={self.differential_diagnosis}, sex={self.sex}, diagnosis={self.pathology}, pathology={self.pathology}, evidences={self.evidences}, initial_evidence={self.initial_evidence})"

    # 核心方法：计算 top 5 诊断
    def get_top_all_diagnosis(self):
        if not self.differential_diagnosis:  # 处理空列表情况
            return []
        # 按概率降序排序，取前5个诊断名称
        sorted_diagnosis = sorted(
            self.differential_diagnosis, key=lambda x: x[1], reverse=True
        )
        return [item[0] for item in sorted_diagnosis]


path = "../../../data/examples/ddxplus/release_test_patients.csv"

# 读取并解析 CSV
df = pd.read_csv(path, encoding="utf-8")
list_columns = ["DIFFERENTIAL_DIAGNOSIS", "EVIDENCES"]
for col in list_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 转换为自定义类对象列表
medical_objects = []

for _, row in df.iterrows():
    record = MedicalRecord(
        age=row["AGE"],
        differential_diagnosis=row["DIFFERENTIAL_DIAGNOSIS"],
        sex=row["SEX"],
        pathology=row["PATHOLOGY"],
        evidences=row["EVIDENCES"],
        initial_evidence=row["INITIAL_EVIDENCE"],
    )
    medical_objects.append(record)
print("数据集大小：", len(medical_objects))


def build_prompt(data: dict) -> tuple[Any, Any, Any]:
    data_path = "prompt.yaml"
    with open(data_path, "r", encoding="utf-8") as f:
        prompt_data = yaml.safe_load(f)
    prompt_template = prompt_data["ddxplus"]["prompt"]
    system_tole = prompt_data["ddxplus"]["role"]
    role_content = prompt_data["ddxplus"]["content"]
    for key, value in data.items():
        prompt_template = prompt_template.replace(f"{{{key}}}", str(value))
    return system_tole, role_content, prompt_template


if __name__ == "__main__":
    # 运行一些测试代码
    llm_client = LLMClient(
        model_name=model_name,
        api_key=api_key,
        api_base=base_url,
    )
    llm_client.init_client()
    # SEED = 42
    # random.seed(SEED)
    random.seed()
    # 随机抽取50条样本
    sample_records = random.sample(medical_objects, 50)
    logging.debug(f"抽样后数量：{len(sample_records)}")  # 确认是否为 50
    logging.debug("前5条抽样结果")  # 打印部分结果查看
    for record in sample_records:
        logging.debug(f"{record}\n")

    right_count = 0
    for i, record in enumerate(sample_records):
        print(f"======第{i + 1}个问题=====")
        print(f"age:{record.age}, sex:{record.sex}")
        init_evidence = decode_evidence(record.initial_evidence)
        evidences = [decode_evidence(e) for e in record.evidences]
        formatted_evidences = "\n".join(f"- {item}" for item in evidences)
        formatted_all_diagnosis = "\n".join(
            f"- {item}" for item in record.top_all_diagnosis
        )
        example_data = {
            "age": record.age,
            "gender": "male" if record.sex == "M" else "female",
            "initial_evidence": init_evidence,
            "clinical_diagnosis": formatted_evidences,
            "all_diagnosis": formatted_all_diagnosis,
        }
        role, content, prompt = build_prompt(example_data)
        response = llm_client.ask_model(role="system", content=content, prompt=prompt)
        logging.info(f"第{i}个response:{response}")
        logging.info(f"第{i}个pathlogy:{record.pathology}")
        print("pathlogy:", record.pathology)
        print("response:", response)
        if record.pathology == response:
            right_count += 1
    accuracy = right_count / 50
    logging.info(f"总体准确率为:{accuracy:.2%}")
