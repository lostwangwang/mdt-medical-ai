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
    """患者状态摘要"""

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
    """医疗事件"""

    patient_id: str
    time: datetime
    event_type: str  # "lab/vital/medication"
    name: str
    value: float
    notes: Optional[str] = None


@dataclass
class RoleOpinion:
    """角色意见"""

    role: RoleType
    treatment_preferences: Dict[TreatmentOption, float]  # -1 to +1
    reasoning: str
    confidence: float  # 0 to 1
    concerns: List[str]


@dataclass
class DialogueMessage:
    """对话消息"""

    role: RoleType
    content: str
    timestamp: datetime
    message_type: str  # "initial_opinion", "response", "rebuttal", "consensus"
    referenced_roles: List[RoleType]
    evidence_cited: List[str]
    treatment_focus: TreatmentOption


@dataclass
class DialogueRound:
    """对话轮次"""

    round_number: int
    messages: List[DialogueMessage]
    focus_treatment: Optional[TreatmentOption]
    consensus_status: str  # "discussing", "converging", "concluded"


@dataclass
class ConsensusResult:
    """共识结果"""

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
    """强化学习状态"""

    patient_features: List[float]
    consensus_features: List[float]
    treatment_preferences: List[float]
    temporal_features: List[float]


@dataclass
class RLAction:
    """强化学习动作"""

    treatment_recommendation: TreatmentOption
    confidence_level: float
    explanation: str


@dataclass
class RLReward:
    """强化学习奖励"""

    consensus_score: float
    consistency_bonus: float
    conflict_penalty: float
    patient_suitability: float
    total_reward: float


@dataclass
class MemoryState:
    """记忆状态（与杜军的Memory Controller接口）"""

    patient_id: str
    individual_memory: Dict[str, Any]
    group_memory: Dict[str, Any]
    temporal_sequence: List[Dict[str, Any]]
    last_updated: datetime
