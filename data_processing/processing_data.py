from src.tools.read_files import read_all_jsonl

"""
/home/wangwang/miniconda3/bin/conda run -n mdt-agent-ai --no-capture-output python /mnt/e/project/LLM/mdt_medical_ai/data_processing/processing_data.py 
原始数据量: 1273
过滤后数据量: 911
删除了 362 条含图像引用的题目
清洗后的数据已保存至: ../data/examples/medqa/data_clean/questions/US/test_update_no_image.jsonl
"""
# 定义与图像相关的关键词（不区分大小写）
IMAGE_KEYWORDS = [
    "shown", "image", "figure", "radiograph", "x-ray", "xray",
    "ecg", "ekg", "electrocardiogram", "photograph", "photo",
    "as depicted", "seen below", "shown below", "illustration",
    "blood smear", "chest x-ray", "ct scan", "mri", "ultrasound"
]

def contains_image_reference(text: str) -> bool:
    text_lower = text.lower()
    for kw in IMAGE_KEYWORDS:
        if kw in text_lower:
            return True
    return False

# 读取原始数据
path = "../data/examples/medqa/data_clean/questions/US/test_update.jsonl"
datas = read_all_jsonl(path)
print(f"原始数据量: {len(datas)}")

# 过滤：仅保留不包含图像引用的题目
cleaned_datas = []
for item in datas:
    question = item.get("question", "")
    if not contains_image_reference(question):
        cleaned_datas.append(item)

print(f"过滤后数据量: {len(cleaned_datas)}")
print(f"删除了 {len(datas) - len(cleaned_datas)} 条含图像引用的题目")

# 可选：保存清洗后的数据
import json

output_path = "../data/examples/medqa/data_clean/questions/US/test_update_no_image.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in cleaned_datas:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"清洗后的数据已保存至: {output_path}")