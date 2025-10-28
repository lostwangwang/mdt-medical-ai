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
        value='QuestionOption',  # 枚举类名
        names=options,           # 成员字典（直接传options，键为成员名，值为成员值）
        type=BaseOption          # 继承BaseOption
    )