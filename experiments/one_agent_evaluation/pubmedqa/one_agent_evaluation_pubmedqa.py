from datasets import load_dataset
from openai import OpenAI
import logging
from datetime import datetime
import os

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
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 你的 Agent Token
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)


def get_datas(count: int = 100):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return ds


def build_prompt(case) -> str:
    question = case["question"]
    context_text = case["context"]

    prompt = (
        "请根据以下医疗问题描述，选择最合适的选项答案。\n\n"
        f"题目：{question}\n\n"
        f"摘要：\n{context_text}\n\n"
        '答案只需要回答: "yes", "no", or "maybe"'
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
            model="qwen3-max-preview",
            messages=[
                {
                    "role": "system",
                    "content": "你是医疗问题回答助手，请基于问题描述和上下文摘要，进行回答最终的问题。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        content = f"[Error] {e}"

    return content


if __name__ == "__main__":
    ds = get_datas(50)
    case = format_data(ds["train"][0])
    print(case)
    # right_count = 0
    # for i in range(50):
    #     logging.info(f"==========第{i}题==============")
    #     case = format_data(ds["train"][i])
    #     prompt = build_prompt(case)
    #     logging.info(prompt)
    #     content = ask_model(case)
    #     if content == case["final_decision"]:
    #         right_count += 1
    # logging.info(f"准确率: {right_count / 50:.2%}")
