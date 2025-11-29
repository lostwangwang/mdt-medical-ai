# from experiments.medqa_types import MedicalQuestionState, QuestionOption
import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv

import experiments.medqa_types as medqa_types
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s',
    filename=f'medqa_app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode='a'  # 追加模式（默认）
)


def read_jsonl(file_path: str, random_sample: int = None, seed: int = 42) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if random_sample is not None:
        # 设置固定随机种子
        random.seed(seed)
        # 确保抽取数量不超过现有条数
        sample_size = min(random_sample, len(lines))
        lines = random.sample(lines, sample_size)
    return [json.loads(line.strip()) for line in lines]


if __name__ == "__main__":
    # path = os.path.join(project_root, "data/examples/medqa/data_clean/questions/Mainland/dev.jsonl")
    # path = "../../" + "data/examples/medqa/data_clean/questions/Mainland/dev.jsonl"
    path = "../../data/examples/medqa/data_clean/questions/US/test_update_no_image.jsonl"
    print(path)
    # data = [
    #     {
    #         "question": "A 68-year-old African American male presents to his primary care provider complaining of progressively worsening back pain. The pain is localized to the lower back and sometimes wakes him up from sleep. He has a history of hypertension and admits to a 50 pack-year smoking history. Further questioning reveals that he has also experienced fatigue and an unintentional weight loss of 18 pounds over the past year. Laboratory analysis is notable for an elevated alkaline phosphatase. A radiograph of the patient’s spine demonstrates multiple sclerotic lesions in the lumbar vertebral bodies. Which of the following tests would most likely confirm the diagnosis in this patient?",
    #         "answer": "Transrectal ultrasound-guided prostate biopsy",
    #         "options": {"A": "Renal biopsy", "B": "Serum protein electrophoresis",
    #                     "C": "Fine needle aspiration of the thyroid",
    #                     "D": "Transrectal ultrasound-guided prostate biopsy", "E": "Chest radiograph"},
    #         "meta_info": "step1", "answer_idx": "D"
    #     }
    #
    # ]
    data = read_jsonl(path, random_sample=5, seed=42)
    right_cnt = 0
    for idx, item in enumerate(data, start=1):
        print(f"执行第{idx}个问题: {item["question"]}")
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
        final_result = dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options,
                                                                     dataset_name="medqa")
        df = final_result["final_consensus"]["df"]
        logging.info(f"第{idx}个问题的共识矩阵: {df}")
        best_treatment = df['mean'].idxmax()
        logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}")
        logging.info(f"第{idx}个问题的平均分数: {df['mean']}")
        if medqa_types.QuestionOption(best_treatment).name == question_state.answer_idx:
            logging.info(f"第{idx}个问题的智能体给的答案: {best_treatment}，正确")
            right_cnt += 1
        else:
            logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}，错误")
        print(f"当前已经答对的问题数: {right_cnt}, 当前是第{idx}个问题")
        logging.debug(f"当前已经答对的问题数: {right_cnt}")
        logging.info(f"第{idx}个问题的正确答案: {question_state.answer_idx}")
        logging.info(f"第{idx}个问题的智能体给的最终方案: {final_result["mdt_leader_final_summary"]}")
    logging.info(f"总体准确率: {right_cnt / len(data):.2f}")
