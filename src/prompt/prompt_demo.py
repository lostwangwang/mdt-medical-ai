import os
from enum import Enum
from prompt_manager import PromptManager


# === 模拟角色枚举 ===
class RoleType(Enum):
    ONCOLOGIST = "肿瘤科医生"
    NURSE = "护士"


# === 模拟问题状态类 ===
class MedicalQuestionState:
    def __init__(self, patient_id, question, options, answer, meta_info, answer_idx):
        self.patient_id = patient_id
        self.question = question
        self.options = options
        self.answer = answer
        self.meta_info = meta_info
        self.answer_idx = answer_idx


# === 模拟输入数据 ===
question_state = MedicalQuestionState(
    patient_id=21645374,
    question="Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    options={"A": "yes", "B": "no", "C": "maybe"},
    answer="yes",
    meta_info="Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant produces perforations in its leaves through PCD.",
    answer_idx="yes",
)

# === 角色信息 ===
role_descriptions = {
    RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
    RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
}

role = RoleType.ONCOLOGIST

# === 初始化 PromptManager ===
# 注意：路径要改成你的 prompts 文件夹所在路径
prompt_dir = os.path.abspath("../../prompts")
pm = PromptManager(prompt_dir)

# === 构建 Prompt ===
final_prompt = pm.build_prompt_for_question(
    dataset_name="pubmedqa",                      # 对应你的 pubmedqa.yaml
    template_name="build_treatment_reasoning_prompt",  # YAML 中的模板名
    role=role,
    role_descriptions=role_descriptions,
    question_state=question_state,
)

# === 打印结果 ===
print("\n========== SYSTEM ==========")
print(final_prompt["system"])

print("\n========== PROMPT ==========")
print(final_prompt["prompt"])
