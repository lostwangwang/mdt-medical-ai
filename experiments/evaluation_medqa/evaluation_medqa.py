# from experiments.medqa_types import MedicalQuestionState, QuestionOption
import os
import sys
import logging
import random

# # 获取当前脚本所在目录（experiments/）
# current_script_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录（mdt_medical_ai/
# project_root = os.path.dirname(current_script_dir)
# # 将项目根目录添加到 Python 搜索路径
# sys.path.append(project_root)
from datetime import datetime
import experiments.medqa_types as medqa_types
from typing import Dict, List
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface
import json

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s',
    filename=f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode='a'  # 追加模式（默认）
)

def read_jsonl(file_path: str, random_sample: int = None) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if random_sample is not None:
        # 确保抽取数量不超过现有条数
        sample_size = min(random_sample, len(lines))
        lines = random.sample(lines, sample_size)
    return [json.loads(line.strip()) for line in lines]

if __name__ == "__main__":

    # path = os.path.join(project_root, "data/examples/medqa/data_clean/questions/Mainland/dev.jsonl")
    path = "../../" + "data/examples/medqa/data_clean/questions/Mainland/dev.jsonl"
    print(path)
    data = read_jsonl(path, 1)
    right_cnt = 0
    for idx, item in enumerate(data, start=1):
        print(f"执行第{idx}个问题: {item['question']}")
        question_state = medqa_types.MedicalQuestionState(
            patient_id=str(idx),
            question=item["question"],
            options=item["options"],
            answer=item["answer"],
            meta_info=item["meta_info"],
            answer_idx=item["answer_idx"],
        )
        medqa_types.init_question_option(item["options"])
        print("枚举成员列表：", list(medqa_types.QuestionOption))
        question_options = list(medqa_types.QuestionOption)
        llm_config = LLMConfig(model_name=model_name, api_key=api_key, base_url=base_url)
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
        # 打印结果
        final_result = dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options)
        df = final_result["final_consensus"]["df"]
        logging.info(f"第{idx}个问题的共识矩阵: {df}")
        best_treatment = df['mean'].idxmax()
        logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}")
        logging.info(f"第{idx}个问题的平均投票: {df['mean']}")
        if medqa_types.QuestionOption(best_treatment).name == question_state.answer_idx:
            logging.info(f"第{idx}个问题的智能体给的答案: {best_treatment}，正确")
            right_cnt += 1
        else:
            logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}，错误")
        logging.info(f"第{idx}个问题的正确答案: {question_state.answer_idx}")
    
    logging.info(f"总体准确率: {right_cnt / len(data):.2f}")
