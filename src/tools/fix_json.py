import re
import json

def fix_and_parse_single_json(raw_text):
    """
    处理“或”关系的单个JSON：修复末尾多}、concerns多]的错误，再解析
    :param raw_text: 大模型返回的单个JSON字符串（含可能错误）
    :return: 解析后的JSON字典
    """
    # 新增：移除首尾的```json和```标记
    # 正则匹配：去掉开头的```json（可选换行）和结尾的```（可选换行）
    # raw_text = re.sub(r'^```json\s*', '', raw_text.strip())  # 移除开头的```json
    # raw_text = re.sub(r'\s*```$', '', raw_text)  # 移除结尾的```
    # 1. 第一步：修复“末尾多}”的错误（匹配最后一个}后多余的}，并删除）
    # 正则逻辑：匹配字符串末尾的“任意空格+}”，只保留1个}
    fixed_text = re.sub(r'(\})\s*\}', r'\1', raw_text.strip(), count=1)
    
    # 2. 第二步：修复“concerns多]”的错误（匹配列表末尾的“]]”，替换为“]”）
    # 正则逻辑：匹配 "concerns": [...] 结构中末尾的多余]
    fixed_text = re.sub(r'("concerns": \[.*?)\]\]', r'\1]', fixed_text, flags=re.DOTALL)
    
    # 3. 第三步：标准JSON解析（若仍有小错误，可加容错处理）
    try:
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        # 打印错误信息和修复后的文本，方便调试
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

    # 情况2：返回第二个JSON（concerns多]）
    raw_json2 = """{
        "treatment_preferences": {"A": -0.3, "B": -0.2, "C": -0.6, "D": 0.8, "E": -0.5},
        "reasoning": "急性期首要控制炎症与疼痛，非甾体抗炎药（D）可快速缓解症状，利于早期功能维持",
        "confidence": 0.9,
        "concerns": ["注意胃肠道副作用", "避免长期使用", "监测肾功能"]]
    }"""
    parsed2 = fix_and_parse_single_json(raw_json2)
    print("解析结果2：", parsed2)