import os
import random
import sys
import logging

# 获取当前脚本所在目录（experiments/）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（mdt_medical_ai/，即 experimentsexperiments 的父目录）
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)
from datetime import datetime
import experiments.medqa_types as medqa_types
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'/mnt/e/project/LLM/mdt_medical_ai/data/result/pubmedqa_log/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)


def get_datas(seed=42):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return ds


if __name__ == "__main__":
    ds = get_datas()
    total = 50
    right_cnt = 0
    dataset_name = "pubmedqa"
    for idx in range(1, total + 1):
        print("第{}个问题".format(idx))
        case = ds["train"][idx]
        options = {"A": "yes", "B": "no", "C": "maybe"}
        context_text = ""
        print(case["context"]["contexts"])
        for ctx in case["context"]["contexts"]:
            context_text += ctx + "\n"

        question_context = "Context: " + context_text + "\nQuestion: " + case["question"]
        reversed_options = {v: k for k, v in options.items()}
        question_state = medqa_types.MedicalQuestionState(
            patient_id=case["pubid"],
            question=question_context,
            options=options,
            answer=case["final_decision"],
            meta_info=context_text,
            answer_idx=reversed_options[case["final_decision"]],
        )
        print("问题的状态:", question_state)
        medqa_types.init_question_option(options)
        print("枚举成员列表：", list(medqa_types.QuestionOption))
        question_options = list(medqa_types.QuestionOption)
        print(ds)
        question_options = list(medqa_types.QuestionOption)
        llm_config = LLMConfig(model_name=model_name, api_key=api_key, base_url=base_url)
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
        # # 打印结果
        final_result = dialogue_manager.conduct_mdt_discussion_medqa_demo(
            question_state, question_options, dataset_name
        )
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
    logging.info(f"总体准确率: {right_cnt / total:.2f}")
