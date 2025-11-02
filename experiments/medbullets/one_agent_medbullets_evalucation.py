import os
import re
import json
import csv
from time import sleep
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

# 优先从 DashScope 兼容模式环境变量读取，其次回退到常见的 OPENAI_API_KEY
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("未检测到 API Key。请在系统环境或 .env 中设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。")

# 模型可通过环境变量覆盖，默认使用 DashScope 预览模型
MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-max-preview")

# ✅ 初始化阿里云 DashScope 客户端（OpenAI 兼容模式）
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "examples" / "medbullets" / "medbullets_op4.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_medbullets_op4(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    读取 medbullets_op4.json 并转换为统一结构：
    [{
        "question": str,
        "options": {"A": str, "B": str, "C": str, "D": str},
        "answer_idx": str  # e.g., "A"/"B"/"C"/"D"
    }, ...]

    原始文件为单个 JSON，字段为列式映射：
    question: {"0": q0, "1": q1, ...}
    opa/opb/opc/opd: {"0": a0, ...}
    answer_idx: {"0": "A", ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MedBullets 文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 必要字段检查
    required_cols = ["question", "opa", "opb", "opc", "opd", "answer_idx"]
    for col in required_cols:
        if col not in obj:
            raise ValueError(f"输入JSON缺少必要字段: {col}")

    q_map = obj["question"]
    a_map = obj["answer_idx"]
    opa = obj["opa"]
    opb = obj["opb"]
    opc = obj["opc"]
    opd = obj["opd"]

    # 索引集合（字符串索引，如 "0","1",...）
    ids = list(q_map.keys())
    try:
        ids = sorted(ids, key=lambda x: int(x))
    except Exception:
        ids = sorted(ids)

    dataset: List[Dict[str, Any]] = []
    for i, k in enumerate(ids):
        if limit is not None and len(dataset) >= limit:
            break
        q = str(q_map.get(k, "")).strip()
        ai = a_map.get(k)
        # 允许数字/字母，统一转成字母
        if isinstance(ai, int):
            idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
            ai_letter = idx_to_letter.get(ai)
        elif isinstance(ai, str):
            ai_letter = ai.strip().upper()
        else:
            ai_letter = None

        options = {
            "A": str(opa.get(k, "")).strip(),
            "B": str(opb.get(k, "")).strip(),
            "C": str(opc.get(k, "")).strip(),
            "D": str(opd.get(k, "")).strip(),
        }

        # 跳过不完整或无效样本
        if not q or not all(options.values()) or ai_letter not in {"A", "B", "C", "D"}:
            continue
        dataset.append({
            "id": k,
            "question": q,
            "options": options,
            "answer_idx": ai_letter,
        })

    return dataset


def build_prompt(case: Dict[str, Any]) -> str:
    options_str = "\n".join([f"{key}. {value}" for key, value in case["options"].items()])
    prompt = (
        "请根据以下医疗问题描述，选择最合适的选项答案。\n\n"
        f"题目：{case['question']}\n\n"
        f"选项：\n{options_str}\n\n"
        "请直接回答选项的字母（A/B/C/D）。"
    )
    return prompt


def extract_answer_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def ask_model(case: Dict[str, Any]) -> Dict[str, Any]:
    # 更强约束的提示，鼓励只输出字母
    option_lines = ", ".join([f"{k}: {v}" for k, v in case["options"].items()])
    prompt = (
        "你是一名医学智能体，现在要回答医学考试选择题。\n\n"
        f"题目：\n{case['question']}\n\n"
        f"选项：\n{option_lines}\n\n"
        "只输出一个选项字母作为最终答案（例如 D）。"
    )

    raw_output: str
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一名医学考试智能体，请仅用于医学知识问答，不进行真实诊断。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=16,
        )
        raw_output = response.choices[0].message.content.strip()
    except Exception as e:
        raw_output = f"[Error] {e}"

    pred = extract_answer_letter(raw_output)
    correct = (pred == case["answer_idx"]) if pred else False

    return {
        "id": case.get("id"),
        "question": case["question"],
        "pred_answer": pred,
        "true_answer": case["answer_idx"],
        "is_correct": correct,
        "raw_output": raw_output,
    }


def evaluate_dataset(dataset: List[Dict[str, Any]], sleep_sec: float = 1.0) -> (List[Dict[str, Any]], float):
    results: List[Dict[str, Any]] = []
    correct = 0
    total = len(dataset)

    for i, case in enumerate(dataset, 1):
        print(f"👉 正在推理第 {i}/{total} 题...")
        result = ask_model(case)
        print("=== 模型原始输出 ===")
        print(result["raw_output"])  # 供调试/检查用
        results.append(result)
        if result["is_correct"]:
            correct += 1
        print(f"模型答案: {result['pred_answer']} | 正确答案: {result['true_answer']} | 是否正确: {result['is_correct']}")
        sleep(sleep_sec)

    acc = correct / max(1, total)
    print(f"\n✅ 总体正确率: {acc:.2%}")
    return results, acc


def save_to_csv(results: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
    if filename is None:
        filename = RESULTS_DIR / f"medbullets_one_agent_{int(__import__('time').time())}.csv"
    else:
        filename = Path(filename)
        if not filename.is_absolute():
            filename = RESULTS_DIR / filename
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "pred_answer", "true_answer", "is_correct", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"📁 已保存结果到: {filename}")
    return str(filename)


if __name__ == "__main__":
    # 可通过环境变量 LIMIT 控制样本数量
    limit_env = os.getenv("MEDBULLETS_LIMIT")
    limit = int(limit_env) if limit_env and limit_env.isdigit() else None

    print(f"读取数据: {DATA_PATH}")
    dataset = load_medbullets_op4(str(DATA_PATH), limit=limit)
    print(f"=== 读取了 {len(dataset)} 条题目 ===")
    results, acc = evaluate_dataset(dataset, sleep_sec=1.0)
    save_to_csv(results)