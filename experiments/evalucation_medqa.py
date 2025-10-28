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


if __name__ == "__main__":
    # 示例数据
    question = (
        "男，50岁。吃海鲜后夜间突发左足第一跖趾关节剧烈疼痛1天。查体：关节局部红肿，"
    )
    options = {
        "A": "苯溴马隆",
        "B": "别嘌呤醇",
        "C": "抗生素",
        "D": "非甾体抗炎药",
        "E": "甲氟蝶呤",
    }
    # 第三步：初始化枚举类（必须在所有使用前执行）
    medqa_types.init_question_option(options)
    print("枚举成员列表：", list(medqa_types.QuestionOption))  # 应输出所有选项
    answer = "非甾体抗炎药"
    meta_info = "第一部分　历年真题"
    answer_idx = "D"

    # 创建实例
    question_state = medqa_types.MedicalQuestionState(
        patient_id="1",
        question=question,
        options=options,
        answer=answer,
        meta_info=meta_info,
        answer_idx=answer_idx,
    )
    question_options = list(medqa_types.QuestionOption)
    llm_config = LLMConfig(model_name=None, api_key=None, base_url=None)
    llm_interface = LLMInterface(config=llm_config)
    rag_system = MedicalKnowledgeRAG()
    dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
    # 打印结果
    dialogue_manager.conduct_mdt_discussion_medqa(question_state, question_options)
    print(question_state)
