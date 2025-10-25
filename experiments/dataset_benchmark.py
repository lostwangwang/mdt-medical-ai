import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging
import random
from datetime import datetime
import re
import sys
from pathlib import Path

# 确保可以从项目根目录导入 main_integrated 等模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---- Task schemas ----

@dataclass
class MCQExample:
    id: str
    question: str
    options: List[str]
    answer_idx: int  # ground truth index


@dataclass
class YNExample:
    id: str
    question: str
    context: Optional[str]
    answer: str  # "yes" | "no" | "maybe"


# ---- Baseline interfaces ----

class TaskBaseline:
    name: str
    def predict_mcq(self, ex: MCQExample) -> int:
        raise NotImplementedError
    def predict_yn(self, ex: YNExample) -> str:
        raise NotImplementedError


class RandomBaseline(TaskBaseline):
    def __init__(self):
        self.name = "RandomBaseline"

    def predict_mcq(self, ex: MCQExample) -> int:
        return random.randint(0, len(ex.options) - 1)

    def predict_yn(self, ex: YNExample) -> str:
        return random.choice(["yes", "no", "maybe"])


class MajorityBaseline(TaskBaseline):
    def __init__(self, majority_label: Optional[str] = None):
        self.name = "MajorityBaseline"
        self.majority_label = majority_label or "yes"

    def predict_mcq(self, ex: MCQExample) -> int:
        # MCQ没有全局多数类概念，回退随机
        return random.randint(0, len(ex.options) - 1)

    def predict_yn(self, ex: YNExample) -> str:
        return self.majority_label


class KeywordHeuristicBaseline(TaskBaseline):
    def __init__(self):
        self.name = "KeywordHeuristicBaseline"
        self.pos_keywords = {"improve", "effective", "recommended", "beneficial"}
        self.neg_keywords = {"contraindicated", "ineffective", "harm", "not recommended"}

    def predict_mcq(self, ex: MCQExample) -> int:
        # 简单启发：若题干出现某选项关键词，优先该选项；否则随机
        lower_q = ex.question.lower()
        # 尝试按选项的子串匹配（实际数据需更稳健的映射）
        for i, opt in enumerate(ex.options):
            token = opt.lower().strip()
            if token and token in lower_q:
                return i
        return random.randint(0, len(ex.options) - 1)

    def predict_yn(self, ex: YNExample) -> str:
        txt = (ex.question + " " + (ex.context or "")).lower()
        if any(k in txt for k in self.pos_keywords):
            return "yes"
        if any(k in txt for k in self.neg_keywords):
            return "no"
        return "maybe"


# ---- MDT Baseline Adapter ----

class MDTBaseline(TaskBaseline):
    """
    使用项目中的MDT系统（优先 main_integrated.EnhancedPatientDialogueManager，
    回退到核心 MultiAgentDialogueManager），若无法给出结构化答案则回退到项目自带LLM接口，
    最后兜底使用启发式。
    """
    def __init__(self,
                 faiss_db_path: str = "clinical_memory_db",
                 patient_id: str = "MEDQA_PSEUDO"):
        self.name = "MDTBaseline"
        self.patient_id = patient_id
        self.faiss_db_path = faiss_db_path
        self._init_ok = False
        self._mode = "none"  # enhanced | core | none
        self._enhanced_mgr = None
        self._core_mgr = None
        self._llm = None
        self._init_mdt()

    def _init_mdt(self) -> None:
        """尝试初始化增强版MDT管理器，失败则回退到核心对话管理器，并准备LLM回退。"""
        # 优先增强版
        try:
            from main_integrated import EnhancedPatientDialogueManager  # type: ignore
            from src.knowledge.enhanced_faiss_integration import EnhancedFAISSManager  # type: ignore
            from src.consensus.consensus_matrix import ConsensusMatrix  # type: ignore
            from src.rl.rl_environment import MDTReinforcementLearning  # type: ignore
            from src.utils.llm_interface import LLMInterface, LLMConfig  # type: ignore

            faiss_mgr = EnhancedFAISSManager(db_path=self.faiss_db_path)
            consensus = ConsensusMatrix()
            rl_env = MDTReinforcementLearning()
            self._enhanced_mgr = EnhancedPatientDialogueManager(
                faiss_manager=faiss_mgr,
                consensus_system=consensus,
                rl_environment=rl_env,
            )
            # 预备LLM回退
            try:
                cfg = LLMConfig()
                self._llm = LLMInterface(cfg)
            except Exception:
                self._llm = None

            self._mode = "enhanced"
            self._init_ok = True
            logger.info("MDTBaseline: 使用 EnhancedPatientDialogueManager")
            return
        except Exception as e:
            logger.warning(f"MDTBaseline: 增强版管理器不可用，回退到核心MDT。原因: {e}")

        # 回退核心版
        try:
            from src.knowledge.rag_system import MedicalKnowledgeRAG  # type: ignore
            from src.knowledge.enhanced_faiss_integration import EnhancedFAISSManager  # type: ignore
            from src.consensus.dialogue_manager import MultiAgentDialogueManager  # type: ignore
            from src.utils.llm_interface import LLMInterface, LLMConfig  # type: ignore

            faiss_mgr = EnhancedFAISSManager(db_path=self.faiss_db_path)
            rag = MedicalKnowledgeRAG(faiss_manager=faiss_mgr)
            # LLM回退
            try:
                cfg = LLMConfig()
                self._llm = LLMInterface(cfg)
            except Exception:
                self._llm = None
            self._core_mgr = MultiAgentDialogueManager(rag_system=rag, llm_interface=self._llm)
            self._mode = "core"
            self._init_ok = True
            logger.info("MDTBaseline: 使用核心 MultiAgentDialogueManager")
        except Exception as e:
            logger.error(f"MDTBaseline: 核心管理器初始化失败: {e}")
            self._init_ok = False

    # ---- 工具方法 ----
    @staticmethod
    def _letter_for_index(idx: int) -> str:
        return chr(ord('A') + idx)

    @staticmethod
    def _index_for_letter(letter: str) -> Optional[int]:
        up = letter.strip().upper()
        if up and 'A' <= up <= 'Z':
            return ord(up) - ord('A')
        return None

    @staticmethod
    def _simple_overlap_score(q: str, opt: str) -> float:
        # 基于词重叠的极简分数作为最后回退
        def toks(s: str) -> List[str]:
            return [t for t in re.split(r"[^a-zA-Z]+", s.lower()) if t]
        qs = set(toks(q))
        os_ = set(toks(opt))
        if not qs or not os_:
            return 0.0
        inter = len(qs & os_)
        union = len(qs | os_)
        return inter / max(1, union)

    def _ask_mdt(self, query: str) -> str:
        """调用MDT系统返回文本回答。失败则返回空字符串。"""
        if not self._init_ok:
            return ""
        try:
            if self._mode == "enhanced" and self._enhanced_mgr is not None:
                result = self._enhanced_mgr.query_patient_info_with_mdt(self.patient_id, query)
                if isinstance(result, dict):
                    return str(result.get("response", ""))
                return str(result)
            elif self._mode == "core" and self._core_mgr is not None:
                # 核心管理器没有直接问答接口，这里返回空，留给LLM回退
                return ""
        except Exception as e:
            logger.warning(f"MDTBaseline: 调用MDT失败: {e}")
        return ""

    def _ask_llm_letter(self, prompt: str, num_options: int) -> Optional[int]:
        """使用项目内置LLM直答一个字母，解析成index。"""
        if not self._llm:
            return None
        try:
            # 通过 LLMInterface 走一个最小调用路径：使用 generate_professional_reasoning 可能会输出自然语言
            # 这里我们直接尝试 client 接口，如果不可用则返回None
            client = getattr(self._llm, 'client', None)
            cfg = getattr(self._llm, 'config', None)
            if client and cfg:
                resp = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict MCQ solver. Reply with a single capital letter only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4,
                )
                text = resp.choices[0].message.content.strip()
            else:
                # 无真实LLM时，退化为模板：返回空以继续下游兜底
                return None
        except Exception as e:
            logger.warning(f"MDTBaseline: LLM回退失败: {e}")
            return None

        # 解析字母
        m = re.search(r"\b([A-Z])\b", text)
        if m:
            idx = self._index_for_letter(m.group(1))
            if idx is not None and 0 <= idx < num_options:
                return idx
        return None

    @staticmethod
    def _extract_option_letter(text: str, num_options: int) -> Optional[int]:
        if not text:
            return None
        m = re.search(r"(?i)answer\s*[:：]\s*([A-Z])", text)
        if m:
            idx = MDTBaseline._index_for_letter(m.group(1))
            if idx is not None and 0 <= idx < num_options:
                return idx
        letters = re.findall(r"\b([A-Z])\b", text)
        for ch in reversed(letters):
            idx = MDTBaseline._index_for_letter(ch)
            if idx is not None and 0 <= idx < num_options:
                return idx
        return None

    def predict_mcq(self, ex: MCQExample) -> int:
        # 构造明确格式化的提问，强约束输出字母
        option_lines = []
        for i, opt in enumerate(ex.options):
            option_lines.append(f"{self._letter_for_index(i)}. {opt}")
        prompt = (
            "You are an MDT system. Read the USMLE-style question and choose the single best answer.\n"
            "Return ONLY the option letter. No explanation.\n\n"
            f"Question: {ex.question}\nOptions:\n" + "\n".join(option_lines) +
            "\n\nAnswer with a single letter from [" + ", ".join(self._letter_for_index(i) for i in range(len(ex.options))) + "].\n"
            "Output format: just the letter."
        )
        text = self._ask_mdt(prompt)
        idx = self._extract_option_letter(text, len(ex.options))
        if idx is not None:
            return idx
        # 回退：项目LLM直答
        llm_idx = self._ask_llm_letter(prompt, len(ex.options))
        if llm_idx is not None:
            return llm_idx
        # 最后兜底：使用极简重叠分数
        scores = [self._simple_overlap_score(ex.question, opt) for opt in ex.options]
        best = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        return best

    def predict_yn(self, ex: YNExample) -> str:
        prompt = (
            "You are an MDT system. Answer the question as yes/no/maybe.\n"
            "Return ONLY one of: yes, no, maybe.\n\n"
            f"Question: {ex.question}\nContext: {ex.context or ''}\n\nAnswer:"
        )
        text = self._ask_mdt(prompt).strip().lower()
        for cand in ["yes", "no", "maybe"]:
            if cand in text:
                return cand
        # 回退：尝试LLM
        if self._llm:
            try:
                client = getattr(self._llm, 'client', None)
                cfg = getattr(self._llm, 'config', None)
                if client and cfg:
                    resp = client.chat.completions.create(
                        model=cfg.model_name,
                        messages=[
                            {"role": "system", "content": "Reply with exactly one token: yes or no or maybe."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=2,
                    )
                    txt = resp.choices[0].message.content.strip().lower()
                    if txt in {"yes", "no", "maybe"}:
                        return txt
            except Exception as e:
                logger.warning(f"MDTBaseline: YN LLM回退失败: {e}")
        # 兜底：关键词
        kh = KeywordHeuristicBaseline()
        return kh.predict_yn(ex)


# ---- Datasets loaders ----

def load_medqa(path: str, max_samples: Optional[int] = None) -> List[MCQExample]:
    """
    期望格式（示例）:
    每行一个JSON: { "id": "...", "question": "...", "options": ["A", "B", "C", "D"], "answer_idx": 2 }
    实际MedQA-USMLE官方格式为不同字段命名，请先转换为上述统一格式。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MedQA file not found: {path}")
    examples: List[MCQExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # ---- 兼容官方与统一格式 ----
            q = obj.get("question", "")
            opts = obj.get("options")
            options_list: List[str] = []
            ans_idx: Optional[int] = None

            if isinstance(opts, dict):
                # 官方格式: options 是字典（A/B/C...）
                keys = sorted(opts.keys())
                options_list = [opts[k] for k in keys]

                # 1) answer_idx 是字母
                if isinstance(obj.get("answer_idx"), str) and obj["answer_idx"] in keys:
                    ans_idx = keys.index(obj["answer_idx"])
                # 2) answer 是字母
                elif isinstance(obj.get("answer"), str) and obj["answer"] in keys:
                    ans_idx = keys.index(obj["answer"])
                # 3) answer 是正确文本（与选项完全匹配）
                elif isinstance(obj.get("answer"), str):
                    try:
                        ans_idx = options_list.index(obj["answer"])
                    except ValueError:
                        ans_idx = None

            elif isinstance(opts, list):
                # 统一格式: options 是列表
                options_list = opts
                if isinstance(obj.get("answer_idx"), int):
                    ans_idx = int(obj["answer_idx"])
                elif isinstance(obj.get("answer_idx"), str) and obj["answer_idx"].isdigit():
                    ans_idx = int(obj["answer_idx"])
                elif isinstance(obj.get("answer"), str):
                    # 尝试把 answer 作为选项文本匹配
                    try:
                        ans_idx = options_list.index(obj["answer"])
                    except ValueError:
                        ans_idx = None

            # 生成样本
            if q and options_list and ans_idx is not None and 0 <= ans_idx < len(options_list):
                ex_id = obj.get("id", f"medqa_{i+1:06d}")
                examples.append(MCQExample(
                    id=ex_id,
                    question=q,
                    options=options_list,
                    answer_idx=int(ans_idx),
                ))

            if max_samples and len(examples) >= max_samples:
                break
    return examples


def load_pubmedqa(path: str, max_samples: Optional[int] = None) -> List[YNExample]:
    """
    期望格式（示例）:
    每行一个JSON: { "id": "...", "question": "...", "context": "...", "answer": "yes|no|maybe" }
    官方数据包含abstract与question，需先转换为统一格式。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PubMedQA file not found: {path}")
    examples: List[YNExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ans = obj["answer"].lower()
            if ans not in {"yes", "no", "maybe"}:
                continue
            examples.append(YNExample(
                id=obj["id"],
                question=obj["question"],
                context=obj.get("context"),
                answer=ans,
            ))
            if max_samples and len(examples) >= max_samples:
                break
    return examples


# TODO: 扩展加载器（占位符）
def load_ddxplus(path: str, max_samples: Optional[int] = None):
    raise NotImplementedError("DDXPlus loader not implemented yet")
def load_symcat(path: str, max_samples: Optional[int] = None):
    raise NotImplementedError("SymCat loader not implemented yet")
def load_jama(path: str, max_samples: Optional[int] = None):
    raise NotImplementedError("JAMA loader not implemented yet")
def load_medbullets(path: str, max_samples: Optional[int] = None):
    raise NotImplementedError("MedBullets loader not implemented yet")


# ---- Evaluation ----

def eval_mcq(baseline: TaskBaseline, examples: List[MCQExample]) -> Dict[str, Any]:
    correct = 0
    preds: List[int] = []
    for ex in examples:
        pred = baseline.predict_mcq(ex)
        preds.append(pred)
        if pred == ex.answer_idx:
            correct += 1
    acc = correct / max(1, len(examples))
    return {
        "task": "MCQ",
        "model_name": baseline.name,
        "num_examples": len(examples),
        "accuracy": round(acc, 4),
        "timestamp": datetime.now().isoformat(),
    }


def eval_yn(baseline: TaskBaseline, examples: List[YNExample]) -> Dict[str, Any]:
    labels = ["yes", "no", "maybe"]
    correct = 0
    for ex in examples:
        pred = baseline.predict_yn(ex)
        if pred == ex.answer:
            correct += 1
    acc = correct / max(1, len(examples))
    return {
        "task": "YN",
        "labels": labels,
        "model_name": baseline.name,
        "num_examples": len(examples),
        "accuracy": round(acc, 4),
        "timestamp": datetime.now().isoformat(),
    }


# ---- Runner ----

def run_dataset_benchmark(
    dataset: str,
    data_path: str,
    baseline_name: str,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    # 选择baseline
    if baseline_name.lower() == "random":
        baseline = RandomBaseline()
    elif baseline_name.lower() in {"majority", "majority_yes", "majority_no"}:
        baseline = MajorityBaseline(majority_label="no" if "no" in baseline_name.lower() else "yes")
    elif baseline_name.lower() == "keyword":
        baseline = KeywordHeuristicBaseline()
    elif baseline_name.lower() in {"mdt", "mdt_system", "our_mdt"}:
        baseline = MDTBaseline()
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")

    # 加载数据并评估
    dataset = dataset.lower()
    if dataset == "medqa":
        examples = load_medqa(data_path, max_samples)
        result = eval_mcq(baseline, examples)
    elif dataset == "pubmedqa":
        examples = load_pubmedqa(data_path, max_samples)
        result = eval_yn(baseline, examples)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Dataset Benchmark Runner")
    parser.add_argument("--dataset", type=str, required=True, help="medqa | pubmedqa | ddxplus | symcat | jama | medbullets")
    parser.add_argument("--data-path", type=str, required=True, help="Path to preprocessed JSONL file")
    parser.add_argument("--baseline", type=str, default="random", help="random | majority | majority_no | keyword | mdt")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", type=str, default="results/dataset_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        result = run_dataset_benchmark(args.dataset, args.data_path, args.baseline, args.max_samples)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Benchmark complete. Saved to: {args.output}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        print(f"❌ Benchmark failed: {e}")


if __name__ == "__main__":
    main()