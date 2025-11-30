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
    #         "question": "A 23-year-old woman presents to the emergency department because of nausea. She has been experiencing severe intermittent nausea and diarrhea for the last year and refuses to leave the emergency department until she is told what is causing her symptoms. She hates hospitals but has reluctantly undergone numerous workups and imaging studies in order to discern what is wrong. All of the studies have been normal. She says that these symptoms have been causing her severe distress and is impairing her ability to perform her job. She says that she feels as if the nausea is worse when she is trying to work but says that she will continue to work as an insurance agent despite the physical discomfort. She has started restricting her diet to only graham crackers because she thinks that helps her symptoms. She has also started avoiding eating lunch because of the nausea. Which of the following disorders is most consistent with this patient's presentation?",
    #         "answer": "Somatic symptom disorder",
    #         "options": {"A": "Conversion disorder", "B": "Factitious disorder", "C": "Illness anxiety disorder",
    #                     "D": "Malingering", "E": "Somatic symptom disorder"}, "meta_info": "step1", "answer_idx": "E"}
    #
    # ]
    # data = [{"question": "An 83-year-old man is admitted to the hospital with fever, weakness, and decreased responsiveness. He is diagnosed with urosepsis based on urinalysis and culture and started on ceftriaxone and intravenous fluids. By hospital day 3, he is clinically improving. During the evening, the patient becomes irritable. He is talking to someone despite nobody being present in the room. He is easily agitated and attempts to strike a nurse with a remote control to his TV. Subsequently, the patient keeps getting out of bed and trying to walk away despite being a fall risk. Which of the following is the most appropriate next step in management?", "answer": "Olanzapine", "options": {"A": "Diphenhydramine", "B": "Lorazepam", "C": "Olanzapine", "D": "Phenobarbital", "E": "Physical restraints"}, "meta_info": "step2&3", "answer_idx": "C"}]
    data = read_jsonl(path, random_sample=50, seed=42)
    # data = [
    #     {
    #         "question": "A 53-year-old man with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that his physician believes may be beneficial to his condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. A dose-response trial had the following results:\n\n10 mg dose - 6/59 patients demonstrated improvement\n20 mg dose - 19/49 patients demonstrated improvement\n30 mg dose - 26/53 patients demonstrated improvement\n40 mg dose - 46/51 patients demonstrated improvement\n\nThe same trial also had the following safety profile:\n\n20 mg dose - 5/49 patients had a treatment related adverse event\n40 mg dose - 11/51 patients had a treatment related adverse event\n60 mg dose - 15/42 patients had a treatment related adverse event\n80 mg dose - 23/47 patients had a treatment related adverse event\n100 mg dose - 47/52 patients had a treatment related adverse event\n\nBased on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?",
    #         "answer": "2.67", "options": {"A": "0.375", "B": "0.5", "C": "2", "D": "2.5", "E": "2.67"},
    #         "meta_info": "step1", "answer_idx": "E"}
    # ]
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
        mdt_leader_final_summary = final_result["mdt_leader_final_summary"]
        print(mdt_leader_final_summary["label"])
        label = mdt_leader_final_summary["label"]
        if label == question_state.answer_idx:
            logging.info(f"第{idx}个问题的智能体给的答案: {label}，正确")
            right_cnt += 1
        else:
            logging.info(f"第{idx}个问题正确的答案: {question_state.answer_idx}，错误")
        logging.info(f"第{idx}个问题的最终答案标签: {mdt_leader_final_summary['label']}")
        logging.info(f"第{idx}个问题最终答案的内容: {mdt_leader_final_summary['content']}")
        logging.info(f"第{idx}个问题的最终摘要: {mdt_leader_final_summary['decision_reasoning']}")
        # best_treatment = df['mean'].idxmax()
        # logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}")
        # logging.info(f"第{idx}个问题的平均分数: {df['mean']}")
        # if medqa_types.QuestionOption(best_treatment).name == question_state.answer_idx:
        #     logging.info(f"第{idx}个问题的智能体给的答案: {best_treatment}，正确")
        #     right_cnt += 1
        # else:
        #     logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}，错误")
        # print(f"当前已经答对的问题数: {right_cnt}, 当前是第{idx}个问题")
        # logging.debug(f"当前已经答对的问题数: {right_cnt}")
        # logging.info(f"第{idx}个问题的正确答案: {question_state.answer_idx}")
        # logging.info(f"第{idx}个问题的智能体给的最终方案: {final_result["mdt_leader_final_summary"]}")

    logging.info(f"总体准确率: {right_cnt / len(data):.2f}")
