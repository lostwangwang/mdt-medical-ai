# -*- coding: utf-8 -*-
"""
EHR → 病情管理记忆库（JSON版）
读取 example.json，提取一级/二级核心信息，
并调用 Qwen2.5-Med 生成**总体**病情摘要，输出
{subject_id}_clinical_memory.json
"""
import json
import datetime
from pydoc import cli
import sys
from typing import Dict, List, Any
from pathlib import Path
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import re

# -------------------------- 配置 --------------------------
INPUT_JSON = "breast.json"          # 同目录下的 MIMIC 示例
OUTPUT_DIR = Path("memory_output")   # 输出文件夹
OUTPUT_DIR.mkdir(exist_ok=True)

LLM_API_KEY = os.getenv("QWEN_API_KEY")
LLM_BASE_URL = os.getenv("BASE_URL")
LLM_MODEL = "qwen-plus"

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# -------------------------- 向量库配置 --------------------------
FAISS_DEVICE = "cuda"  # 使用 GPU 设备以加速嵌入计算
FAISS_DB_PATH = "clinical_memory_db"  # 全局库路径（所有患者共用）

def get_embeddings():
    """初始化中文句向量模型（与 demo.py 保持一致的系列）"""
    return SentenceTransformerEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": FAISS_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )

# ---------- 原脚本工具函数（完整搬过来） ----------
def parse_json_data(json_str, data_type):
    try:
        # 兼容字符串/列表/None
        if json_str is None:
            return []
        if isinstance(json_str, list):
            data_list = json_str
        else:
            s = str(json_str).strip()
            if s in ["[]", "null", ""]:
                return []
            cleaned = s.replace('""', '"')
            if not cleaned.startswith('['):
                cleaned = f"[{cleaned}]"
            data_list = json.loads(cleaned)
        parsed = []
        for item in data_list:
            time_str = item.get("charttime") or item.get("starttime") or item.get("storetime") or item.get("chart_time")
            if not time_str:
                continue
            try:
                charttime = datetime.datetime.fromisoformat(time_str)
                date = charttime.strftime("%Y-%m-%d")
            except ValueError:
                continue
            # 名称字段兼容
            if data_type == "lab":
                item_name = item.get("lab_name") or str(item.get("itemid", "Unknown_Lab"))
            else:
                item_name = item.get("vital_name") or item.get("label") or str(item.get("itemid", "Unknown_Vital"))
            # 数值解析：优先 valuenum；否则从 value 中提取数字
            val = item.get("valuenum")
            if isinstance(val, str):
                try:
                    val = float(val)
                except Exception:
                    val = None
            if val is None:
                v_str = item.get("value")
                if isinstance(v_str, (str, bytes)):
                    if isinstance(v_str, bytes):
                        v_str = v_str.decode("utf-8", errors="ignore")
                    m = re.search(r"([+-]?\d+(?:\.\d+)?)", v_str)
                    if m:
                        try:
                            val = float(m.group(1))
                        except Exception:
                            val = None
            item_info = {
                "date": date,
                "data_type": data_type,
                "item_name": item_name,
                "valuenum": val,
                "valueuom": item.get("valueuom", "") or item.get("uom", ""),
                "flag": item.get("flag")
            }
            if isinstance(item_info["valuenum"], (int, float)):
                parsed.append(item_info)
        return parsed
    except Exception as e:
        print(f"[{data_type}] JSON解析失败: {e}")
        return []

def aggregate_daily_data(parsed_labs, parsed_vitals):
    daily_data = {}
    for item in parsed_labs + parsed_vitals:
        date = item["date"]
        key = f"{item['data_type']}_{item['item_name'].replace(' ', '_').replace('/', '_')}"
        daily_data.setdefault(date, {}).setdefault(key, {"values": [], "uom": item["valueuom"], "flag": None})
        if isinstance(item.get("valuenum"), (int, float)):
            daily_data[date][key]["values"].append(float(item["valuenum"]))
        if item["flag"] == "abnormal":
            daily_data[date][key]["flag"] = "abnormal"
    aggregated = []
    for date in sorted(daily_data):
        day_dict = {"date": date}
        for k, v in daily_data[date].items():
            if not v["values"]:
                continue
            mean_val = round(sum(v["values"]) / max(len(v["values"]), 1), 2)
            day_dict[f"{k}_mean"] = mean_val
            day_dict[f"{k}_uom"] = v["uom"]
            day_dict[f"{k}_flag"] = v["flag"] or "normal"
        aggregated.append(day_dict)
    return aggregated

# ---------- 安全 JSON 工具 ----------
def safe_list_from_json(value):
    """将可能为 None/空串/JSON串/已是列表 的输入安全转为 list。
    解析失败时返回空列表。
    """
    try:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            v = value.strip()
            if v == "" or v.lower() in ("null", "none"):
                return []
            return json.loads(v)
        # 其他类型直接空列表
        return []
    except Exception:
        return []

# ---------- LLM ----------
def call_qwen(prompt: str) -> str:
    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是临床数据分析师，需基于患者EHR的每日实验室指标和生命体征，客观分析病情变化。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"【LLM异常】{e}"

# ---------- 核心 ----------
def load_patients() -> List[Dict[str, Any]]:
    with open(INPUT_JSON, encoding="utf-8") as f:
        return json.load(f)

def overall_summary(p_info: Dict, daily_agg: List[Dict]) -> Dict[str, str]:
    """生成住院期间总体总结（无结构化日记录时使用出院生命体征作为依据）"""
    if not daily_agg:
        # 无结构化日记录：依据 baseline_vitals 给出简要评估
        bv = p_info.get("baseline_vitals", {}) if isinstance(p_info, dict) else {}
        parts = []
        if "temp" in bv:
            parts.append(f"体温{bv['temp']}")
        if "heart_rate" in bv:
            parts.append(f"心率{bv['heart_rate']}")
        if "sbp" in bv and "dbp" in bv:
            parts.append(f"血压{bv['sbp']}/{bv['dbp']}")
        if "resp_rate" in bv:
            parts.append(f"呼吸{bv['resp_rate']}")
        if "spo2" in bv:
            parts.append(f"血氧{bv['spo2']}%")
        if parts:
            return {"summary": "结构化记录较少，依据出院生命体征：" + "，".join(parts) + "。整体趋于稳定。", "trend": "stable"}
        return {"summary": "结构化记录较少，整体趋于稳定。", "trend": "stable"}

    chunks = []
    for day in daily_agg:
        d = day["date"]
        labs = [f"{k}:{v}" for k, v in day.items() if k.startswith("lab_") and "_mean" in k]
        vitals = [f"{k}:{v}" for k, v in day.items() if k.startswith("vital_") and "_mean" in k]
        chunks.append(f"{d} 实验室：{'；'.join(labs)} 生命体征：{'；'.join(vitals)}")
    prompt = f"""
患者{p_info['gender']}{p_info['anchor_age']}岁，诊断{p_info['chronic_diseases'][:200]}。
住院期间指标如下：
""" + "\n".join(chunks) + """
请用2-3句话总体总结住院期间病情变化趋势（改善/恶化/稳定），并指出最关键异常。只给总结，不展开建议。"""
    summary = call_qwen(prompt)
    trend = "improve" if "改善" in summary else "worsen" if "恶化" in summary else "stable"
    return {"summary": summary, "trend": trend}

def build_memory(patient: Dict) -> Dict:
    mem = {
        "subject_id": patient["subject_id"],
        "gender": patient["gender"],
        "anchor_age": patient["anchor_age"],
        "allergies": [],
        "cancer_history": [],
        "chronic_diseases": [],
        "baseline_labs": {},
        "baseline_vitals": {},
        "discharge_medications": [],
        "daily_summaries": {}
    }

    # 过敏
    try:
        discharge_list = safe_list_from_json(patient.get("discharge_json"))
        if discharge_list:
            note = discharge_list[0].get("text_snippet", "")
            if "Allergies:" in note:
                mem["allergies"] = [a.strip() for a in note.split("Allergies:")[1].split("\n")[0].split("/")]
    except Exception:
        pass

    # 诊断
    try:
        for d in safe_list_from_json(patient.get("diagnoses")):
            code = str(d.get("icd_code", "")).strip()
            desc = str(d.get("desc", "")).strip()
            if not code:
                continue
            if code.startswith("1"):
                mem["cancer_history"].append({"icd9": code, "desc": desc})
            elif code[:1] in "2345":
                mem["chronic_diseases"].append({"icd9": code, "desc": desc})
    except Exception:
        pass

    # 基线检验 & 生命体征
    labs = safe_list_from_json(patient.get("labs_json"))
    baseline_labs = {}
    for k in labs:
        name = k.get("lab_name") or str(k.get("itemid", "")).strip()
        if not name:
            continue
        if k.get("flag") == "abnormal":
            continue
        val = k.get("valuenum")
        if isinstance(val, str):
            try:
                val = float(val)
            except Exception:
                val = None
        if val is None:
            v_str = k.get("value")
            if isinstance(v_str, (str, bytes)):
                if isinstance(v_str, bytes):
                    v_str = v_str.decode("utf-8", errors="ignore")
                m = re.search(r"([+-]?\d+(?:\.\d+)?)", v_str)
                if m:
                    try:
                        val = float(m.group(1))
                    except Exception:
                        val = None
        if not isinstance(val, (int, float)):
            continue
        baseline_labs[name.lower().replace(" ", "_")] = val
    mem["baseline_labs"] = baseline_labs
    # 解析出院记录 VS 生命体征
    discharge_list = safe_list_from_json(patient.get("discharge_json"))
    if discharge_list:
        note = discharge_list[0].get("text_snippet", "")
        m_line = re.search(r"VS:.*", note)
        vit = {}
        if m_line:
            line = m_line.group(0)
            tokens = re.split(r"\s+", line.replace("VS:", "").strip())
            try:
                if len(tokens) >= 1:
                    t = re.search(r"([+-]?\d+(?:\.\d+)?)", tokens[0])
                    if t:
                        vit["temp"] = float(t.group(1))
                if len(tokens) >= 2:
                    h = re.search(r"(\d+)", tokens[1])
                    if h:
                        vit["heart_rate"] = int(h.group(1))
                if len(tokens) >= 3 and "/" in tokens[2]:
                    sbp_dbp = tokens[2].split("/")
                    try:
                        vit["sbp"] = int(re.sub(r"\D", "", sbp_dbp[0]))
                        vit["dbp"] = int(re.sub(r"\D", "", sbp_dbp[1]))
                    except Exception:
                        pass
                if len(tokens) >= 4:
                    r = re.search(r"(\d+)", tokens[3])
                    if r:
                        vit["resp_rate"] = int(r.group(1))
                if len(tokens) >= 5:
                    s = re.search(r"(\d+)", tokens[4])
                    if s:
                        vit["spo2"] = int(s.group(1))
            except Exception:
                pass
        if vit:
            mem["baseline_vitals"] = vit
    if not mem["baseline_vitals"]:
        mem["baseline_vitals"] = {k: v for k, v in {"heart_rate": 75, "sbp": 120, "dbp": 80, "spo2": 98}.items()}

    # 出院带药
    try:
        rx = safe_list_from_json(patient.get("prescriptions_json"))
        mem["discharge_medications"] = [{"drug": r.get("drug"), "dose": r.get("dose_val_rx"), "route": r.get("route")} for r in rx]
    except Exception:
        pass

    # 总体摘要
    parsed_labs = parse_json_data(patient["labs_json"], "lab")
    parsed_vitals = parse_json_data(patient["vitals_json"], "vital")
    daily_agg = aggregate_daily_data(parsed_labs, parsed_vitals)
    mem["daily_summaries"] = overall_summary(mem, daily_agg)
    return mem


def save_patient_memory_to_faiss(subject_id: str, memory_obj: Dict[str, Any]) -> str:
    """将患者病情 JSON 保存到全局 FAISS 库（统一索引，按 subject_id 标注）"""
    embeddings = get_embeddings()
    text = json.dumps(memory_obj, ensure_ascii=False)
    doc_id = f"clinical_memory_{subject_id}"

    # 若库已存在则加载并追加；否则以当前文档初始化
    if Path(FAISS_DB_PATH).exists():
        db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        db.add_texts([text], metadatas=[{"subject_id": subject_id}], ids=[doc_id])
    else:
        db = FAISS.from_texts([text], embeddings, metadatas=[{"subject_id": subject_id}], ids=[doc_id])
    db.save_local(FAISS_DB_PATH)
    return FAISS_DB_PATH


def get_memory_by_subject_id(subject_id: str) -> Dict[str, Any]:
    """在全局 FAISS 库中按 subject_id 获取对应患者的 JSON 内容"""
    embeddings = get_embeddings()
    if not Path(FAISS_DB_PATH).exists():
        raise FileNotFoundError(f"未找到全局FAISS数据库：{FAISS_DB_PATH}")
    db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # 1) 先按唯一ID命中
    doc_id = f"clinical_memory_{subject_id}"
    doc = None
    try:
        doc = db.docstore.search(doc_id)
    except Exception:
        doc = getattr(db.docstore, "_dict", {}).get(doc_id)

    # 2) 回退：遍历按 metadata 匹配 subject_id
    if not doc and hasattr(db.docstore, "_dict"):
        for d in db.docstore._dict.values():
            if isinstance(getattr(d, "metadata", None), dict) and d.metadata.get("subject_id") == subject_id:
                doc = d
                break

    if not doc:
        raise RuntimeError(f"全局库中未找到 subject_id={subject_id} 的临床记忆文档")

    try:
        return json.loads(doc.page_content)
    except Exception:
        return {"raw": doc.page_content, "metadata": doc.metadata}

def main():
    pts = load_patients()
    for p in pts:
        sid = p["subject_id"]
        memory = build_memory(p)
        out = OUTPUT_DIR / "clinical_memory" / f"{sid}_clinical_memory.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
        print(f"✅ 已生成 {out}")

        # 保存到 FAISS（所有患者共用同一库）
        db_path = save_patient_memory_to_faiss(sid, memory)
        print(f"📦 已保存到全局 FAISS 数据库: {db_path}")

if __name__ == "__main__":
    patient_memory = get_memory_by_subject_id(11489167)
    print(patient_memory)
    # main()