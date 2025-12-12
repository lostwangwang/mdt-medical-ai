import ast
import logging
import os

from pandas import DataFrame, read_csv
from experiments.evaluation_ddxplus.convert_evidences import decode_evidence
from experiments.evaluation_symcat.evaluation_symcat import model_name, api_key
from src.tools.read_files import read_jsonl
from datetime import datetime
import experiments.medqa_types as medqa_types
from typing import Dict, List
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.llm_interface import LLMConfig, LLMInterface
from src.tools.list_to_options import create_letter_options
from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'./logs/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)


# 注解：df 是 DataFrame，columns 是字符串列表，返回值是 DataFrame
def convert_str_columns_to_lists(df: DataFrame, columns: List[str]) -> DataFrame:
    for col in columns:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df


if __name__ == "__main__":
    path = "/mnt/e/project/LLM/mdt_medical_ai/data/examples/ddxplus/release_test_patients.csv"
    data = read_csv(path)
    list_columns = ["DIFFERENTIAL_DIAGNOSIS", "EVIDENCES"]
    df = convert_str_columns_to_lists(data, list_columns)
    medical_objects = []
    df_sample = df.sample(n=50)
    for idx, row in df_sample.iterrows():
        medical_object = medqa_types.DDXPlusMedicalRecord(
            age=row["AGE"],
            differential_diagnosis=row["DIFFERENTIAL_DIAGNOSIS"],
            sex=row["SEX"],
            pathology=row["PATHOLOGY"],
            evidences=row["EVIDENCES"],
            initial_evidence=row["INITIAL_EVIDENCE"],
        )
        medical_objects.append(medical_object)
    # medical_objects = medical_objects[:1]  # 仅测试前10个病例
    right_cnt = 0
    for idx, item in enumerate(medical_objects, start=1):
        print(f"执行第{idx}个问题")
        print(f"item: {item}")
        init_diagnosis = decode_evidence(item.initial_evidence)
        print(f"init_diagnosis: {init_diagnosis}")
        formatted_init_diagnosis = f"{init_diagnosis}"
        evidences = [decode_evidence(evidence) for evidence in item.evidences]
        formatted_evidences = "\n".join(f"- {item}" for item in evidences)
        question = (
            f"A {item.age} year old {item.sex} patient with the initial evidence: {formatted_init_diagnosis} \n "
            f"And has the following evidences: {formatted_evidences}.\n"
            f"What is the most likely diagnosis?"
        )
        print(f"question: {question}")
        print(f"item.all_diagnosis: {item.all_diagnosis}")
        logging.info((f"第{idx}个问题的答案:{item.pathology}"))
        options = create_letter_options(item.all_diagnosis)
        print(options)
        reverse_options = {v: k for k, v in options.items()}
        question_state = medqa_types.MedicalQuestionState(
            patient_id=str(idx),
            question=question,
            options=options,
            answer=item.pathology,
            meta_info="",
            answer_idx=reverse_options[item.pathology],
        )
        medqa_types.init_question_option(options)
        print("枚举成员列表：", list(medqa_types.QuestionOption))
        question_options = list(medqa_types.QuestionOption)
        llm_config = LLMConfig(model_name=model_name, api_key=api_key, base_url=base_url)
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
        # 打印结果
        final_result = dialogue_manager.conduct_mdt_discussion_medqa_demo(
            question_state, question_options, dataset_name="ddxplus"
        )
        # final_df = final_result["final_consensus"]["df"]
        # logging.info(f"第{idx}个问题的共识矩阵: {final_df}")
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
        # best_treatment = final_df["mean"].idxmax()
        # logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}")
        # logging.info(f"第{idx}个问题的平均投票: {final_df['mean']}")
        # if medqa_types.QuestionOption(best_treatment).value == question_state.answer_idx:
        #     logging.info(f"第{idx}个问题的智能体给的答案: {best_treatment}，正确")
        #     right_cnt += 1
        # else:
        #     logging.info(f"第{idx}个问题的最佳治疗方案: {best_treatment}，错误")
        # logging.info(f"第{idx}个问题的正确答案: {question_state.answer_idx}")

    logging.info(f"总体准确率: {right_cnt / len(medical_objects):.2f}")
