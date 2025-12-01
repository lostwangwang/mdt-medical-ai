import logging
from datetime import datetime
import experiments.medqa_types as medqa_types
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.tools.read_files import read_jsonl
from src.utils.llm_interface import LLMConfig, LLMInterface
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量（默认找当前目录的 .env）
load_dotenv()  # 放在代码最前面，确保优先加载

# 之后就可以用 os.getenv() 读取了
model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'./logs/app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)

if __name__ == "__main__":
    path = "/mnt/e/project/LLM/mdt_medical_ai/data/examples/symcat/symcat_style_dataset.jsonl"

    data = read_jsonl(path, 50)
    print("读取的数据条数:", len(data))
    print("数据示例:", data[0])
    records = []
    for item in data:
        record = medqa_types.SymCatPatientRecord(
            patient_id=item.get("PATIENT"),
            age=item.get("Age"),
            gender=item.get("Gender"),
            race=item.get("Race"),
            ethnicity=item.get("Ethnicity"),
            num_symptoms=item.get("NUM_SYMPTOMS"),
            pathology=item.get("PATHOLOGY"),
            symptoms=item.get("Symptoms"),
            options=item.get("Options"),
        )
        records.append(record)
    right_cnt = 0
    for idx, rec in enumerate(records, start=1):
        logging.info(f"第{idx}个问题\n")
        print(f"患者ID: {rec.patient_id}, 症状: {rec.symptoms}")
        symptoms_str = "\n".join(
            f"- {symptom}({value})" for symptom, value in rec.symptoms.items()
        )
        question = (
            f"A {rec.age}-year-old {rec.gender} patient presents with the following findings: \n"
            f"Race: {rec.race}, Ethnicity: {rec.ethnicity} \n"
            f"With {rec.num_symptoms} symptoms: \n{symptoms_str}\n"
            f"Which option is most likely?"
        )
        print(f"生成的问题: {question}")
        reverse_options = {v: k for k, v in rec.options.items()}
        question_state = medqa_types.MedicalQuestionState(
            patient_id=rec.patient_id,
            question=question,
            options=rec.options,
            answer=rec.pathology,
            meta_info="",
            answer_idx=reverse_options[rec.pathology],
        )
        medqa_types.init_question_option(rec.options)
        print("枚举成员列表：", list(medqa_types.QuestionOption))
        question_options = list(medqa_types.QuestionOption)
        llm_config = LLMConfig(
            model_name=model_name, api_key=api_key, base_url=base_url
        )
        llm_interface = LLMInterface(config=llm_config)
        rag_system = MedicalKnowledgeRAG()
        dialogue_manager = MultiAgentDialogueManager(rag_system, llm_interface)
        # 打印结果
        final_result = dialogue_manager.conduct_mdt_discussion_medqa(
            question_state, question_options, dataset_name="symcat"
        )
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
    logging.info(f"总体准确率: {right_cnt / len(data):.2f}")
