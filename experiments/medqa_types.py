from enum import Enum
from dataclasses import dataclass
from typing import Dict, Type, Optional


@dataclass
class MedicalQuestionState:
    """医疗问题状态摘要"""

    patient_id: str
    question: str
    options: Dict[str, str]  # 格式：{"A": "选项内容", "B": "..."}
    answer: str
    meta_info: str
    answer_idx: str  # 如 "A"


# 定义数据类（映射 CSV 列）
class SymCatPatientRecord:
    def __init__(
        self,
        patient_id,
        age,
        gender,
        race,
        ethnicity,
        num_symptoms,
        pathology,
        symptoms,
        options,
    ):
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.race = race
        self.ethnicity = ethnicity
        self.num_symptoms = num_symptoms
        self.pathology = pathology
        self.symptoms = self.parse_symptoms(symptoms)
        self.options = options

    def parse_symptoms(self, symptoms_str):
        """
        将症状字符串解析为字典: {'症状名': 数值}
        例如: "Difficulty breathing:35::::::35;Increased heart rate:34::::::34"
        -> {'Difficulty breathing': 35, 'Increased heart rate': 34}
        """
        symptom_dict = {}
        for item in symptoms_str.split(";"):
            if not item:
                continue
            parts = item.split(":")
            symptom_name = parts[0]
            try:
                value = int(parts[1])
            except ValueError:
                value = None
            symptom_dict[symptom_name] = value
        return symptom_dict


# 定义数据类（映射 CSV 列）
class DDXPlusMedicalRecord:
    def __init__(
        self, age, differential_diagnosis, sex, pathology, evidences, initial_evidence
    ):
        self.age = age  # 年龄
        self.differential_diagnosis = (
            differential_diagnosis  # 鉴别诊断列表（含概率） 相当于选项
        )
        self.sex = sex  # 性别
        self.pathology = pathology  # 确诊疾病 # 相当于答案
        self.evidences = evidences  # 所有证据 # 相当于上下文
        self.initial_evidence = initial_evidence  # 初始证据 # 初始证据
        self.all_diagnosis = self.get_top_all_diagnosis()  # 计算所有诊断

    # 可选：自定义打印格式
    def __str__(self):
        return f"MedicalRecord(age={self.age}, differential_diagnosis={self.differential_diagnosis}, sex={self.sex}, diagnosis={self.pathology}, pathology={self.pathology}, evidences={self.evidences}, initial_evidence={self.initial_evidence})"

    # 核心方法：计算全部诊断
    def get_top_all_diagnosis(self):
        if not self.differential_diagnosis:  # 处理空列表情况
            return []
        # 按概率降序排序，取前5个诊断名称
        sorted_diagnosis = sorted(
            self.differential_diagnosis, key=lambda x: x[1], reverse=True
        )
        return [item[0] for item in sorted_diagnosis]


class BaseOption(Enum):
    """选项基类，用于类型约束"""

    pass


# 全局变量：存储动态创建的枚举类（初始为None）
QuestionOption: Optional[Type[BaseOption]] = None


def init_question_option(options: Dict[str, str]) -> None:
    """
    初始化QuestionOption枚举类（必须在使用前调用）
    解决枚举动态创建和导入问题的核心函数
    """
    global QuestionOption
    # 关键：用Enum的构造函数直接创建，传入name、成员字典和基类
    QuestionOption = Enum(
        value="QuestionOption",  # 枚举类名
        names=options,  # 成员字典（直接传options，键为成员名，值为成员值）
        type=BaseOption,  # 继承BaseOption
    )
