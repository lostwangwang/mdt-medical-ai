import os
import sys
import logging
# 获取当前脚本所在目录（experiments/）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（mdt_medical_ai/，即 experimentsexperiments 的父目录）
project_root = os.path.dirname(current_script_dir)
sys.path.append(project_root)
from datetime import datetime
import experiments.medqa_types as medqa_types
from typing import Dict, List
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface
import json
from datasets import load_dataset
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s',
    filename=f'/mnt/e/project/LLM/mdt_medical_ai/data/result/pubmedqa_log/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode='a'  # 追加模式（默认）
)

def get_datas():
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return ds

if __name__ == "__main__":
    ds = get_datas()
    total = 5
    print("ds的个数", len(ds))
    right_cnt = 0
    for idx in range(total):
        print("第{}个问题".format(idx+1))
        case = ds['train'][idx]
        # print(case)
    #     # {"A": "是G+和G-菌的共有组分", "B": "为真核细胞所特有", "C": "占G+菌细胞壁干重的50%～80%", "D": "占G-菌细胞壁干重的5%～20%", "E": "G-菌的肽聚糖由聚糖骨架和四肽侧链组成"}
        options = {"A": "yes", "B": "no", "C": "maybe"}
    #     # options = {"A": "苯溴马隆", "B": "别嘌呤醇", "C": "抗生素", "D": "非甾体抗炎药", "E": "甲氟蝶呤"}
        context_text = ''
        print(case["context"]["contexts"])
        for ctx in case["context"]["contexts"]:
            context_text += ctx + "\n"
        question_state = medqa_types.MedicalQuestionState(
            patient_id=case['pubid'],
            question=case['question'],
            options=options,
            answer=case['final_decision'],
            meta_info=context_text,
            answer_idx=case['final_decision'],
        )
        medqa_types.init_question_option(options)
        print("枚举成员列表：", list(medqa_types.QuestionOption))
        question_options = list(medqa_types.QuestionOption)
        print(ds)
        question_options = list(medqa_types.QuestionOption)
        llm_config = LLMConfig(model_name=None, api_key=None, base_url=None)
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
        if medqa_types.QuestionOption(best_treatment).value == question_state.answer_idx:
            logging.info(f"第{idx}个问题的智能体给的答案: {best_treatment}，正确")
            right_cnt += 1
        else:
            logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}，错误")
        logging.info(f"第{idx}个问题的正确答案: {question_state.answer_idx}")
    logging.info(f"总体准确率: {right_cnt / total:.2f}")