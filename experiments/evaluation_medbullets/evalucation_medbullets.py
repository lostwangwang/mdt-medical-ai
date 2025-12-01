import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from experiments import medqa_types
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface

# 移除不再需要的抓取相关依赖
# import re
# import requests
# from bs4 import BeautifulSoup

# 获取当前脚本所在目录（experiments/）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（MDT-Agent 的父目录）
project_root = os.path.dirname(current_script_dir)
# 将项目根目录添加到 Python 搜索路径
sys.path.append(project_root)

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s',
    filename=f'medbullets_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)


def load_medbullets(path: str, n: Optional[int] = None) -> List[Dict]:
    """
    读取 MedBullets 数据文件（JSON）。
    仅支持直接包含字段映射：
      - question / opa / opb / opc / opd / answer_idx
    可选保留 link 字段用于元信息，但不进行任何抓取。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MedBullets file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    q_map = obj.get('question', {}) or {}
    # 直接字段（若存在）：
    opa_map = obj.get('opa', {}) or {}
    opb_map = obj.get('opb', {}) or {}
    opc_map = obj.get('opc', {}) or {}
    opd_map = obj.get('opd', {}) or {}
    ans_idx_map = obj.get('answer_idx', {}) or {}

    link_map = obj.get('link', {}) or {}

    def _sort_key(k: str):
        try:
            return int(k)
        except Exception:
            return k

    indices = sorted(q_map.keys(), key=_sort_key)
    items: List[Dict] = []

    for idx_key in indices:
        question = q_map.get(idx_key)
        link = link_map.get(idx_key)

        options_dict: Optional[Dict[str, str]] = None
        answer_idx: Optional[str] = None
        answer_text: Optional[str] = None

        # 仅使用直接字段
        if (idx_key in opa_map) and (idx_key in opb_map) and (idx_key in opc_map) and (idx_key in opd_map) and (
                idx_key in ans_idx_map):
            options_dict = {
                'A': str(opa_map.get(idx_key)),
                'B': str(opb_map.get(idx_key)),
                'C': str(opc_map.get(idx_key)),
                'D': str(opd_map.get(idx_key)),
            }
            letter = str(ans_idx_map.get(idx_key)).strip().upper()
            if letter in options_dict:
                answer_idx = letter
                answer_text = options_dict[letter]

        meta_info = f"link: {link}" if link else ""

        if question and options_dict and answer_idx:
            items.append({
                "id": str(idx_key),
                "question": question,
                "options": options_dict,
                "answer": answer_text or "",
                "answer_idx": answer_idx,
                "meta_info": meta_info,
            })

        if n is not None and len(items) >= n:
            break

    return items


if __name__ == "__main__":
    # path = os.path.join(project_root, "data/examples/medbullets/medbullets_op4.json")
    path = "/mnt/e/project/LLM/mdt_medical_ai/data/examples/medbullets/medbullets_op4.json"
    # 读取前若干条样本以快速验证
    print(path)
    data = load_medbullets(path, n=50)
    print(data)
    print(f"载入样本数: {len(data)}")
    if len(data) == 0:
        print("未找到可评估的样本：数据文件缺少 options/answer 字段。\n"
              "请先转换为统一格式：question + options(A/B/...) + answer。\n"
              "当前文件似乎仅包含 link 与 question，无法计算准确率。")
        sys.exit(0)

    right_cnt = 0
    for idx, item in enumerate(data, start=1):
        print(f"执行第{idx}个问题: {item['question']}")

        # 初始化动态枚举（A/B/C...）
        medqa_types.init_question_option(item["options"])
        question_options = list(medqa_types.QuestionOption)

        # 构建问题状态
        question_state = medqa_types.MedicalQuestionState(
            patient_id=str(idx),
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            meta_info="",
            answer_idx=item["answer_idx"],
        )

        # LLM与RAG系统
        llm_config = LLMConfig(model_name=model_name, api_key=api_key, base_url=base_url)
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)

        # 执行MDT协同讨论（复用MedQA流程）
        final_result = dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options,
                                                                     dataset_name="medbullets")
        df = final_result["final_consensus"]["df"]
        logging.info(f"第{idx}个问题的共识矩阵: {df}")
        mdt_leader_final_summary = final_result["mdt_leader_final_summary"]
        print(mdt_leader_final_summary["label"])
        label = mdt_leader_final_summary["label"]
        if label == question_state.answer_idx:
            logging.info(f"第{idx}个问题的智能体给的答案: {label}，回答正确")
            right_cnt += 1
        else:
            logging.info(f"第{idx}个问题正确的答案: {question_state.answer_idx}，回答错误")
        logging.info(f"第{idx}个问题的最终答案标签: {mdt_leader_final_summary['label']}")
        logging.info(f"第{idx}个问题最终答案的内容: {mdt_leader_final_summary['content']}")
        logging.info(f"第{idx}个问题的最终摘要: {mdt_leader_final_summary['decision_reasoning']}")
        logging.info(f"当前已经答对的问题的数量: {right_cnt}")
        # logging.info(f"第{idx}个问题的共识矩阵: {df}")
        #
        # best_treatment = df['mean'].idxmax()
        # logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}")
        # logging.info(f"第{idx}个问题的平均投票: {df['mean']}")
        #
        # # 将最佳方案的值映射回枚举，取其name（字母），与正确答案对比
        # try:
        #     chosen_letter = medqa_types.QuestionOption(best_treatment).name
        # except Exception:
        #     # 若idxmax返回的是枚举成员而非值，兜底处理
        #     chosen_letter = getattr(best_treatment, 'name', str(best_treatment))
        #
        # if chosen_letter == question_state.answer_idx:
        #     logging.info(f"第{idx}个问题的智能体给的答案: {chosen_letter}，正确")
        #     right_cnt += 1
        # else:
        #     logging.info(f"第{idx}个问题的智能体给的答案: {chosen_letter}，错误（正确: {question_state.answer_idx}）")

    logging.info(f"总体准确率: {right_cnt / max(1, len(data)):.2f}")
