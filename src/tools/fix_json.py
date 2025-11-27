import re
import json


def fix_and_parse_single_json(raw_text):
    """
    处理“或”关系的单个JSON：修复末尾多}、concerns多]、concerns少]的错误，再解析
    :param raw_text: 大模型返回的单个JSON字符串（含可能错误）
    :return: 解析后的JSON字典
    """
    # 新增步骤：移除首尾的```json和```标记
    # 1. 先去除字符串前后的所有空白字符（包括换行符）
    content = raw_text.strip()
    # 2. 检查并移除开头的```json标记
    if content.startswith("```json"):
        content = content[7:]  # 从第7个字符开始截取（跳过"```json"）
    # 3. 检查并移除结尾的```标记
    if content.endswith("```"):
        content = content[:-3]  # 截取到倒数第3个字符之前（去掉"```"）
    # 4. 再次去除可能因移除标记而产生的首尾空白
    content = content.strip()

    # 1. 修复“concerns列表缺少闭合]”（核心新增逻辑）
    # 正则逻辑：匹配 "concerns": [xxx} 结构，在}前补全]
    fixed_text = re.sub(r'("concerns": \[.*?)(\s*})', r'\1]\2', content.strip(), flags=re.DOTALL)

    # 2. 原有逻辑：修复末尾多}
    fixed_text = re.sub(r'(\})\s*\}', r'\1', fixed_text, count=1)

    # 3. 原有逻辑：修复concerns多]
    fixed_text = re.sub(r'("concerns": \[.*?)\]\]', r'\1]', fixed_text, flags=re.DOTALL)

    # 4. 容错解析：若仍有错误，尝试移除多余逗号（可选补充）
    fixed_text = re.sub(r',\s*}', '}', fixed_text)  # 移除对象末尾多余逗号
    fixed_text = re.sub(r',\s*]', ']', fixed_text)  # 移除列表末尾多余逗号

    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        print(f"解析失败：{e}")
        print(f"修复后的JSON文本：{fixed_text}")
        raise e


if __name__ == "__main__":
    # ------------------- 调用示例 -------------------
    # 情况1：返回第一个JSON（末尾多}）
    raw_json1 = """{
        "treatment_preferences": {"A": 0.8, "B": -0.9, "C": 0.7, "D": 0.6, "E": 0.5},
        "reasoning": "题目要求选择不正确的描述，B明显错误，因肽聚糖为原核细胞壁成分，非真核细胞特有",
        "confidence": 0.95,
        "concerns": ["B选项错误明显", "需确认题目考查意图", "非肿瘤治疗相关"]}
    }"""
    parsed1 = fix_and_parse_single_json(raw_json1)
    print("解析结果1：", parsed1)

    raw_json3 = """```json
    {
        "scores": {
            "A": -0.8,
            "B": -0.3,
            "C": -1.0,
            "D": 1.0,
            "E": -0.2
        },
        "reasoning": "The patient presents with back pain, weight loss, fatigue, and elevated alkaline phosphatase, with multiple sclerotic vertebral lesions on imaging. These findings are classic for prostate cancer with bone metastases, particularly in an older male with a 50 pack-year smoking history. A transrectal ultrasound-guided prostate biopsy is the most direct diagnostic test for prostate cancer. Other tests are either irrelevant or less specific to the likely diagnosis.",
        "evidence_strength": 0.9,
        "evidences": [
            "Sclerotic bone lesions in lumbar spine suggest metastatic prostate cancer.",
            "Elevated alkaline phosphatase is consistent with bone metastases.",
            "Prostate biopsy is the gold standard for diagnosing prostate cancer."
        ]
    }
    ```"""
    parsed3 = fix_and_parse_single_json(raw_json3)
    print("解析结果3：", parsed3)
