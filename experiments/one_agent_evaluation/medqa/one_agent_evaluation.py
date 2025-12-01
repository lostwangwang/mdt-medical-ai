import os
import re
import re
import json
import csv
import random

from openai import OpenAI
from time import sleep
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

# ✅ 初始化阿里云 DashScope 客户端
client = OpenAI(
    api_key=api_key,  # 你的 Agent Token
    base_url=base_url,
)

import json


def read_jsonl(file_path: str, n: int = None, seed: int = None) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 设置随机种子（保证可复现性）
    if seed is not None:
        random.seed(seed)

    if n is not None:
        lines = lines[:n]
    return [json.loads(line.strip()) for line in lines]


def extract_answer(text: str) -> str:
    match = re.search(r"\b([ABCDE])\b", text)
    return match.group(1) if match else None


def ask_model(case: dict) -> dict:
    prompt = f"""
你是一名医学智能体，现在要回答医学考试选择题。

题目：
{case['question']}

选项：
{', '.join([f'{k}: {v}' for k, v in case['options'].items()])}

"请直接回答选项的字母（A/B/C/D/E）。"
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一名医学考试智能体，请仅用于医学知识问答，不进行真实诊断。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"[Error] {e}"

    # 提取答案
    pred = extract_answer(content)
    correct = pred == case["answer_idx"]

    return {
        "question": case["question"],
        "pred_answer": pred,
        "true_answer": case["answer_idx"],
        "is_correct": correct,
        "raw_output": content,
    }


def evaluate_dataset(dataset):
    results = []
    correct = 0

    for i, case in enumerate(dataset, 1):
        print(f"👉 正在推理第 {i}/{len(dataset)} 题...")
        result = ask_model(case)
        print("=== 模型原始输出 ===")
        print(result["raw_output"])
        results.append(result)

        if result["is_correct"]:
            correct += 1

        print(
            f"模型答案: {result['pred_answer']} | 正确答案: {result['true_answer']} | 是否正确: {result['is_correct']}"
        )
        sleep(1.5)  # 为防止速率限制，可适当延迟

    accuracy = correct / len(dataset)
    print(f"\n✅ 总体正确率: {accuracy:.2%}")
    return results, accuracy


# ✅ 保存结果到 CSV
import time


def save_to_csv(results, filename=f"results_{int(time.time())}.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "pred_answer",
                "true_answer",
                "is_correct",
                "raw_output",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"📁 已保存结果到: {filename}")


if __name__ == "__main__":
    path = "../../../data/examples/medqa/data_clean/questions/US/dev.jsonl"
    total_count = 50
    # cases = read_jsonl(path, total_count)
    # cases = [{"question": "A 23-year-old woman presents to the emergency department with abdominal pain. The patient states that she has pain in her back that radiates to her abdomen. The pain started yesterday and is episodic. She describes the pain as a 10/10 when it occurs. The patient has a past medical history of asthma, anxiety, and chlamydia that was appropriately treated. Her current medications include fluoxetine. The patient's lab values are ordered as seen below.\n\nHemoglobin: 14 g/dL\nHematocrit: 40%\nLeukocyte count: 8,500 cells/mm^3 with normal differential\nPlatelet count: 225,000/mm^3\n\nSerum:\nNa+: 140 mEq/L\nCl-: 100 mEq/L\nK+: 4.9 mEq/L\nHCO3-: 24 mEq/L\nBUN: 22 mg/dL\nGlucose: 89 mg/dL\nCreatinine: 1.2 mg/dL\nCa2+: 10.0 mg/dL\nAST: 11 U/L\nALT: 11 U/L\n\nUrine:\nColor: amber\nNitrites: negative\nNa+: 12 mmol/24 hours\nBeta-hCG: positive\nRed blood cells: 11/hpf\n\nWhich of the following is the best test for this patient's most likely diagnosis?", "answer": "Renal ultrasound", "options": {"A": "Abdominal radiograph", "B": "Abdominal CT", "C": "Transvaginal ultrasound", "D": "Abdominal ultrasound", "E": "Renal ultrasound"}, "meta_info": "step2&3", "answer_idx": "E"}]
    # cases = [{"question": "A 53-year-old man with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that his physician believes may be beneficial to his condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. A dose-response trial had the following results:\n\n10 mg dose - 6/59 patients demonstrated improvement\n20 mg dose - 19/49 patients demonstrated improvement\n30 mg dose - 26/53 patients demonstrated improvement\n40 mg dose - 46/51 patients demonstrated improvement\n\nThe same trial also had the following safety profile:\n\n20 mg dose - 5/49 patients had a treatment related adverse event\n40 mg dose - 11/51 patients had a treatment related adverse event\n60 mg dose - 15/42 patients had a treatment related adverse event\n80 mg dose - 23/47 patients had a treatment related adverse event\n100 mg dose - 47/52 patients had a treatment related adverse event\n\nBased on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?", "answer": "2.67", "options": {"A": "0.375", "B": "0.5", "C": "2", "D": "2.5", "E": "2.67"}, "meta_info": "step1", "answer_idx": "E"}]
    # cases = [{"question": "A 53-year-old man with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that his physician believes may be beneficial to his condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. A dose-response trial had the following results:\n\n10 mg dose - 6/59 patients demonstrated improvement\n20 mg dose - 19/49 patients demonstrated improvement\n30 mg dose - 26/53 patients demonstrated improvement\n40 mg dose - 46/51 patients demonstrated improvement\n\nThe same trial also had the following safety profile:\n\n20 mg dose - 5/49 patients had a treatment related adverse event\n40 mg dose - 11/51 patients had a treatment related adverse event\n60 mg dose - 15/42 patients had a treatment related adverse event\n80 mg dose - 23/47 patients had a treatment related adverse event\n100 mg dose - 47/52 patients had a treatment related adverse event\n\nBased on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?", "answer": "2.67", "options": {"A": "0.375", "B": "0.5", "C": "2", "D": "2.5", "E": "2.67"}, "meta_info": "step1", "answer_idx": "E"}]
    cases = [
        {
            "question": "A 53-year-old man with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that his physician believes may be beneficial to his condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. A dose-response trial had the following results:\n\n10 mg dose - 6/59 patients demonstrated improvement\n20 mg dose - 19/49 patients demonstrated improvement\n30 mg dose - 26/53 patients demonstrated improvement\n40 mg dose - 46/51 patients demonstrated improvement\n\nThe same trial also had the following safety profile:\n\n20 mg dose - 5/49 patients had a treatment related adverse event\n40 mg dose - 11/51 patients had a treatment related adverse event\n60 mg dose - 15/42 patients had a treatment related adverse event\n80 mg dose - 23/47 patients had a treatment related adverse event\n100 mg dose - 47/52 patients had a treatment related adverse event\n\nBased on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?",
            "answer": "2.67", "options": {"A": "0.375", "B": "0.5", "C": "2", "D": "2.5", "E": "2.67"},
            "meta_info": "step1", "answer_idx": "E"}

    ]
    print(f"=== 读取了 {len(cases)} 条病例数据 ===")
    results, acc = evaluate_dataset(cases)
    save_to_csv(results)

# if __name__ == "__main__":

#     path = "dev.jsonl"
#     total_count = 100
#     cases = read_jsonl(path, total_count)
#     print(f"=== 读取了 {len(cases)} 条病例数据 ===")
#     right_count = 0
#     for i, case in enumerate(cases, 1):
#         print(f"\n=== 第 {i} 条病例 ===")
#         print(case)
#         prompt = build_prompt(case)
#         response = client.chat.completions.create(
#             model="qwen3-max",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "你是一名医学考试智能体，请仅用于医学知识问答，不进行真实诊断。",
#                 },
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.2,
#         )

#         # ✅ 调用模型
#         output = response.choices[0].message.content.strip()
#         print("=== 模型原始输出 ===")
#         print(output)

#         # ✅ 提取模型答案（A/B/C/D/E）
#         match = re.search(r"[ABCDE]", output)
#         pred_answer = match.group(0) if match else None
#         print(f"=== 提取的模型答案: {pred_answer} ===")
#         # ✅ 正确答案
#         true_answer = case_data["answer_idx"]

#         # ✅ 计算正确性
#         is_correct = pred_answer == true_answer
#         if is_correct:
#             right_count += 1
#         print("\n=== 自动评估结果 ===")
#         print(f"模型预测答案: {pred_answer}")
#         print(f"真实答案: {true_answer}")
#         print(f"是否正确: {is_correct}")
#     accuracy = right_count / len(cases)
#     print(f"\n=== 总体准确率: {accuracy:.2%} ===")
