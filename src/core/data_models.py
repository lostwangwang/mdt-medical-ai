"""
核心数据模型定义
文件路径: src/core/data_models.py
作者: 团队共同维护
功能: 定义系统中使用的所有数据结构
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class RoleType(Enum):
    """医疗团队角色类型"""

    ONCOLOGIST = "oncologist"  # 肿瘤科医生
    RADIOLOGIST = "radiologist"  # 影像科医生
    NURSE = "nurse"  # 护士
    PSYCHOLOGIST = "psychologist"  # 心理师
    PATIENT_ADVOCATE = "patient_advocate"  # 患者代表


class TreatmentOption(Enum):
    """治疗方案选项"""

    SURGERY = "surgery"
    CHEMOTHERAPY = "chemotherapy"
    RADIOTHERAPY = "radiotherapy"
    IMMUNOTHERAPY = "immunotherapy"
    PALLIATIVE_CARE = "palliative_care"
    WATCHFUL_WAITING = "watchful_waiting"


@dataclass
class PatientState:
    """
    患者状态摘要:
    
    包含患者基本信息、诊断、分期、实验室结果、 vital 指标、症状、 comor病、心理状态、生活质量评分和时间戳。
    
    Attributes:
        patient_id (str): 患者唯一标识符。
        age (int): 患者年龄。
        diagnosis (str): 患者诊断信息。
        stage (str): 患者分期信息。
        lab_results (Dict[str, float]): 患者实验室测试结果，键为测试名称，值为测试结果。
        vital_signs (Dict[str, float]): 患者 vital 指标，键为指标名称，值为指标值。
        symptoms (List[str]): 患者症状描述列表。
        comorbidities (List[str]): 患者合并症描述列表。
        psychological_status (str): 患者心理状态描述。
        quality_of_life_score (float): 患者生活质量评分，0到100之间的浮点数。
        timestamp (datetime): 患者状态记录的时间戳。
    """

    patient_id: str
    age: int
    diagnosis: str
    stage: str
    lab_results: Dict[str, float]
    vital_signs: Dict[str, float]
    symptoms: List[str]
    comorbidities: List[str]
    psychological_status: str
    quality_of_life_score: float
    timestamp: datetime


@dataclass
class MedicalEvent:
    """医疗事件
    
    包含患者ID、事件时间、事件类型、事件名称、事件值和可选的备注。
    
    Attributes:
        patient_id (str): 患者唯一标识符。
        time (datetime): 事件发生时间。
        event_type (str): 事件类型，如"lab/vital/medication"。
        name (str): 事件名称，如"实验室测试结果"、" vital 指标"或"药物 administration"。
        value (float): 事件值，如测试结果、 vital 指标值或药物 administration 量。
        notes (Optional[str]): 可选的事件备注，用于记录额外信息。
    """

    patient_id: str
    time: datetime
    event_type: str  # "lab/vital/medication"
    name: str
    value: float
    notes: Optional[str] = None


@dataclass
class RoleOpinion:
    """
    角色意见
    
    包含角色的治疗偏好、推理、置信度和关注的问题。
    Attributes:
        role (RoleType): 角色类型
        treatment_preferences (Dict[TreatmentOption, float]): 治疗偏好，-1到1之间的值
        reasoning (str): 推理过程，说明为什么选择了这些治疗偏好
        confidence (float): 置信度，0到1之间的值
        concerns (List[str]): 关注的问题列表
    """
    
    role: RoleType
    treatment_preferences: Dict[TreatmentOption, float]  # -1 to +1
    reasoning: str
    confidence: float  # 0 to 1
    concerns: List[str]


@dataclass
class DialogueMessage:
    """
    对话消息
    
    包含角色的对话内容、时间戳、消息类型、引用角色、引用证据和治疗焦点。
    
    Attributes:
        role (RoleType): 角色类型
        content (str): 对话内容
        timestamp (datetime): 消息发送时间
        message_type (str): 消息类型，如"initial_opinion", "response", "rebuttal", "consensus"
        referenced_roles (List[RoleType]): 引用的其他角色列表
        evidence_cited (List[str]): 引用的证据列表
        treatment_focus (TreatmentOption): 当前治疗焦点
    """

    role: RoleType
    content: str
    timestamp: datetime
    message_type: str  # "initial_opinion", "response", "rebuttal", "consensus"
    referenced_roles: List[RoleType]
    evidence_cited: List[str]
    treatment_focus: TreatmentOption


@dataclass
class DialogueRound:
    """
    对话轮次
    
    包含轮次编号、消息列表、治疗焦点和共识状态。
    
    Attributes:
        round_number (int): 轮次编号
        messages (List[DialogueMessage]): 该轮次的所有消息列表
        focus_treatment (Optional[TreatmentOption]): 当前治疗焦点
        consensus_status (str): 共识状态，如"discussing", "converging", "concluded"
    """

    round_number: int
    messages: List[DialogueMessage]
    focus_treatment: Optional[TreatmentOption]
    consensus_status: str  # "discussing", "converging", "concluded"


@dataclass
class ConsensusResult:
    """
    共识结果
    
    包含共识矩阵、角色意见、聚合得分、冲突和达成共识的消息。
    
    Attributes:
        consensus_matrix (Any): 共识矩阵，通常是pandas.DataFrame
        role_opinions (Dict[RoleType, RoleOpinion]): 每个角色的意见
        aggregated_scores (Dict[TreatmentOption, float]): 每个治疗方案的聚合得分
        conflicts (List[Dict[str, Any]]): 记录冲突的消息列表
        agreements (List[Dict[str, Any]]): 记录达成共识的消息列表
        dialogue_summary (Optional[Dict[str, Any]]): 对话总结，如共识状态、轮次等
        timestamp (datetime): 记录时间
        convergence_achieved (bool): 是否达成共识
        total_rounds (int): 总轮次
    """

    consensus_matrix: Any  # pandas.DataFrame
    role_opinions: Dict[RoleType, RoleOpinion]
    aggregated_scores: Dict[TreatmentOption, float]
    conflicts: List[Dict[str, Any]]
    agreements: List[Dict[str, Any]]
    dialogue_summary: Optional[Dict[str, Any]]
    timestamp: datetime
    convergence_achieved: bool = False
    total_rounds: int = 0


@dataclass
class RLState:
    """
    强化学习状态
    
    包含患者特征、共识特征、治疗偏好和时间特征。
    
    Attributes:
        patient_features (List[float]): 患者特征向量
        consensus_features (List[float]): 共识特征向量
        treatment_preferences (List[float]): 治疗偏好向量
        temporal_features (List[float]): 时间特征向量
    """
    patient_features: List[float]
    consensus_features: List[float]
    treatment_preferences: List[float]
    temporal_features: List[float]


@dataclass
class RLAction:
    """
    强化学习动作
    
    包含推荐的治疗方案、置信度和解释。
    
    Attributes:
        treatment_recommendation (TreatmentOption): 推荐的治疗方案
        confidence_level (float): 置信度，0到1之间的值
        explanation (str): 动作的解释
    """

    treatment_recommendation: TreatmentOption
    confidence_level: float
    explanation: str


@dataclass
class RLReward:
    """
    强化学习奖励
    
    包含共识奖励、一致性奖励、冲突惩罚、患者适用性奖励和总奖励。
    
    Attributes:
        consensus_score (float): 共识奖励
        consistency_bonus (float): 一致性奖励
        conflict_penalty (float): 冲突惩罚
        patient_suitability (float): 患者适用性奖励
        total_reward (float): 总奖励
    """

    consensus_score: float
    consistency_bonus: float
    conflict_penalty: float
    patient_suitability: float
    total_reward: float


@dataclass
class MemoryState:
    """
    记忆状态（与杜军的Memory Controller接口）
    
    包含患者ID、个人记忆、群组记忆、时间序列和最后更新时间。
    
    Attributes:
        patient_id (str): 患者ID
        individual_memory (Dict[str, Any]): 个人记忆，如患者特征、治疗偏好等
        group_memory (Dict[str, Any]): 群组记忆，如共识特征、其他患者特征等
        temporal_sequence (List[Dict[str, Any]]): 时间序列，记录患者的历史交互
        last_updated (datetime): 最后更新时间
    """

    patient_id: str
    individual_memory: Dict[str, Any]
    group_memory: Dict[str, Any]
    temporal_sequence: List[Dict[str, Any]]
    last_updated: datetime
