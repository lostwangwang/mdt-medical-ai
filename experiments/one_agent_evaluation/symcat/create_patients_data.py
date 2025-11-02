import pandas as pd
import random
import json

path = "/mnt/e/project/LLM/baseline/synthea/output/symptoms/csv/symptoms.csv"
# 1️⃣ 读取 CSV
df = pd.read_csv(path)

# 2️⃣ 准备疾病池（用于干扰选项）s
disease_pool = df["PATHOLOGY"].unique().tolist()


# 3️⃣ 生成单条 SymCat 风格样本
def generate_symcat_sample(row, disease_pool):
    patient_id = row["PATIENT"]
    age = row["AGE_BEGIN"]
    gender = row["GENDER"]
    race = row["RACE"]
    ethnicity = row["ETHNICITY"]
    num_symptoms = row["NUM_SYMPTOMS"]
    correct_pathology = row["PATHOLOGY"]

    # 处理症状
    symptoms_list = []
    for s in str(row["SYMPTOMS"]).split(";"):
        parts = s.split(":")
        # 填充占位符到 7 层，再加上原来的数字
        symptom_str = ":".join(parts + [""] * (7 - len(parts))) + f":{parts[-1]}"
        symptoms_list.append(symptom_str)
    symptoms = ";".join(symptoms_list)

    # 构建选项：1 正确 + 3 干扰
    distractors = random.sample([d for d in disease_pool if d != correct_pathology], 3)
    options = [correct_pathology] + distractors
    random.shuffle(options)
    option_dict = {chr(65 + i): opt for i, opt in enumerate(options)}  # A, B, C, D

    # 返回完整样本
    return {
        "PATIENT": patient_id,
        "Age": age,
        "Gender": gender,
        "Race": race,
        "Ethnicity": ethnicity,
        "NUM_SYMPTOMS": num_symptoms,
        "PATHOLOGY": correct_pathology,
        "Symptoms": symptoms,
        "Options": option_dict,
    }


# 4️⃣ 生成所有样本
samples = [generate_symcat_sample(row, disease_pool) for _, row in df.iterrows()]

# 5️⃣ 写入 JSONL 文件，每行一个样本
with open("symcat_style_dataset.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"成功生成 {len(samples)} 条 SymCat 风格样本，保存到 symcat_style_dataset.jsonl")
