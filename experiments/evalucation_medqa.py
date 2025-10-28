# from experiments.medqa_types import MedicalQuestionState, QuestionOption
import os
import sys
# 获取当前脚本所在目录（experiments/）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（mdt_medical_ai/，即 experimentsexperiments 的父目录）
project_root = os.path.dirname(current_script_dir)
# 将项目根目录添加到 Python 搜索路径
sys.path.append(project_root)
from dataclasses import dataclass
from datetime import datetime
import experiments.medqa_types as medqa_types
from typing import Dict, List, Optional
from src.core.data_models import RoleType, TreatmentOption, RoleOpinion
from src.core.data_models import PatientState
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface
import json
def read_jsonl(file_path: str, n: int = None) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if n is not None:
        lines = lines[:n]
    return [json.loads(line.strip()) for line in lines]

if __name__ == "__main__":

    path = os.path.join(project_root, "data/examples/medqa/data_clean/questions/Mainland/dev.jsonl")
    data = read_jsonl(path, 10)
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
        llm_config = LLMConfig(model_name=None, api_key=None, base_url=None)
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
        # 打印结果
        final_result = dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options)
        df = final_result["final_consensus"]["df"]
        best_treatment = df['mean'].idxmax()
        if medqa_types.QuestionOption(best_treatment).name == question_state.answer_idx:
            right_cnt += 1
    print(f"准确率: {right_cnt / len(data):.2f}")
       
    

    # 创建实例
    # question_state = medqa_types.MedicalQuestionState(
    #     patient_id="1",
    #     question=question,
    #     options=options,
    #     answer=answer,
    #     meta_info=meta_info,
    #     answer_idx=answer_idx,
    # )
    # question_options = list(medqa_types.QuestionOption)
    # llm_config = LLMConfig(model_name=None, api_key=None, base_url=None)
    # llm_interface = LLMInterface(config=llm_config)
    # rag_system = MedicalKnowledgeRAG()
    # dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
    # # 打印结果
    # final_result = dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options)
    # df = final_result["final_consensus"]["df"]
    # best_treatment = df['mean'].idxmax()
    # print(f"最优问题答案: {best_treatment}")
    # print(f"最优问题答案枚举值: {medqa_types.QuestionOption(best_treatment)}")
    # if medqa_types.QuestionOption(best_treatment).name == question_state.answer_idx:
    #     print("最优问题答案与正确答案一致")
    # else:
    #     print("最优问题答案与正确答案不一致")
    # print(question_state)
