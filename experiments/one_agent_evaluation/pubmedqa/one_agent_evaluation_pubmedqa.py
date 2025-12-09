from datasets import load_dataset
from openai import OpenAI
import logging
from datetime import datetime
import os

from dotenv import load_dotenv

load_dotenv()

model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("BASE_URL")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
    filename=f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',  # 日志文件路径
    filemode="a",  # 追加模式（默认）
)
logging.info("开始加载数据集...")
# 加载 PubMedQA 数据集
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

logging.info("数据集划分：%s", ds.keys())  # 输出如 dict_keys(['train'])
# ✅ 初始化阿里云 DashScope 客户端
client = OpenAI(
    api_key=api_key,  # 你的 Agent Token
    base_url=base_url,
)


def get_datas():
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return ds


def build_prompt(case) -> str:
    question = case["question"]
    context_text = case["context"]

    prompt = (
        "Please choose the most appropriate answer based on the following medical question description.\n\n"
        f"Question：{question}\n\n"
        f"Context：\n{context_text}\n\n"
        f'The answer should only be: "yes", "no", or "maybe".'
    )
    return prompt


def format_data(case):
    question = case["question"]
    context = ""
    for ctx in case["context"]["contexts"]:
        context += ctx + "\n"
    return {
        "question": question,
        "context": context,
        "final_decision": case["final_decision"],
    }


def ask_model(case: dict) -> dict:
    prompt = build_prompt(case)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical question-answering assistant. "
                        "Only output the final answer exactly as: yes, no, or maybe."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7
        )
        print(response)
        print(f"response model: {response.model}")
        logging.info("模型原始输出：%s", response)
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"[Error] {e}"

    return content


if __name__ == "__main__":
    ds = get_datas()
    right_count = 0
    for i in range(1):
        logging.info(f"==========第{i}题==============")
        case = format_data(ds["train"][i])
        prompt = build_prompt(case)
        logging.info(prompt)
        content = ask_model(case)
        print(f"content: {content}")
        print(f"final_decision: {case['final_decision']}")
        if content == case["final_decision"]:
            right_count += 1
    logging.info(f"准确率: {right_count / 50:.2%}")
