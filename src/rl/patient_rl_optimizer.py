"""
病人数据强化学习优化器
基于Memory Controller输出的病人数据进行治疗决策优化
作者: AI Assistant
功能: 实现Q-learning和PPO算法优化医疗决策
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientState:
    """病人状态表示"""
    # 基本信息
    age: float
    gender: str  # 'M' or 'F'
    
    # 实验室指标 (标准化后)
    anion_gap: float
    sodium: float
    potassium: float
    urea_nitrogen: float
    
    # 生命体征
    heart_rate: float
    sbp: float  # 收缩压
    dbp: float  # 舒张压
    spo2: float
    
    # 疾病严重程度评分
    chronic_disease_count: int
    medication_complexity: float
    
    # 临床摘要特征
    stability_trend: float  # 稳定性趋势 (-1到1)

@dataclass
class TreatmentAction:
    """治疗动作"""
    action_id: int
    action_name: str
    description: str
    risk_level: float  # 0-1, 风险等级

@dataclass
class RewardComponents:
    """奖励组件"""
    consensus_score: float      # 共识得分
    stability_score: float      # 稳定性指标
    safety_score: float         # 安全性评分
    effectiveness_score: float  # 有效性评分
    total_reward: float

class PatientRLEnvironment:
    """病人强化学习环境"""
    
    def __init__(self, patient_data_path: str = None):
        """初始化RL环境"""
        self.patient_data_path = patient_data_path
        self.current_patient_data = None
        self.current_state = None
        
        # 如果提供了患者数据路径，立即加载
        if patient_data_path:
            self.load_patient_data(patient_data_path)
        
        # 定义治疗动作空间
        self.actions = [
            TreatmentAction(0, "保守治疗", "监测观察，药物调整", 0.1),
            TreatmentAction(1, "药物强化", "增加药物剂量或种类", 0.3),
            TreatmentAction(2, "介入治疗", "微创介入手术", 0.5),
            TreatmentAction(3, "手术治疗", "外科手术干预", 0.7),
            TreatmentAction(4, "综合治疗", "多学科联合治疗", 0.4),
            TreatmentAction(5, "紧急处理", "急诊紧急干预", 0.9)
        ]
        
        # 奖励权重配置
        self.reward_weights = {
            "consensus_score": 0.3,
            "stability_score": 0.3,
            "safety_score": 0.2,
            "effectiveness_score": 0.2
        }
        
        # 历史记录
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # 环境空间定义
        self.state_size = 13  # 状态向量维度
        self.action_size = len(self.actions)  # 动作空间大小
        
        logger.info("病人RL环境初始化完成")
    
    def load_patient_data(self, patient_file: str) -> Dict[str, Any]:
        """加载病人数据"""
        try:
            with open(patient_file, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            self.current_patient_data = patient_data
            logger.info(f"成功加载病人数据: {patient_data.get('subject_id', 'Unknown')}")
            return patient_data
            
        except Exception as e:
            logger.error(f"加载病人数据失败: {e}")
            return None
    
    def extract_state_from_patient_data(self, patient_data: Dict[str, Any]) -> PatientState:
        """从病人数据提取状态向量"""
        try:
            # 基本信息
            age = patient_data.get('anchor_age', 65) / 100.0  # 标准化到0-1
            gender = 1.0 if patient_data.get('gender') == 'M' else 0.0
            
            # 实验室指标 (标准化)
            labs = patient_data.get('baseline_labs', {})
            anion_gap = self._normalize_lab_value(labs.get('anion_gap', 12), 8, 16)
            sodium = self._normalize_lab_value(labs.get('sodium', 140), 135, 145)
            potassium = self._normalize_lab_value(labs.get('potassium', 4.0), 3.5, 5.0)
            urea_nitrogen = self._normalize_lab_value(labs.get('urea_nitrogen', 15), 7, 25)
            
            # 生命体征
            vitals = patient_data.get('baseline_vitals', {})
            heart_rate = self._normalize_vital(vitals.get('heart_rate', 75), 60, 100)
            sbp = self._normalize_vital(vitals.get('sbp', 120), 90, 140)
            dbp = self._normalize_vital(vitals.get('dbp', 80), 60, 90)
            spo2 = self._normalize_vital(vitals.get('spo2', 98), 95, 100)
            
            # 疾病复杂度
            chronic_diseases = patient_data.get('chronic_diseases', [])
            chronic_disease_count = len(chronic_diseases) / 10.0  # 标准化
            
            medications = patient_data.get('discharge_medications', [])
            medication_complexity = len(medications) / 50.0  # 标准化
            
            # 稳定性趋势
            daily_summary = patient_data.get('daily_summaries', {})
            trend = daily_summary.get('trend', 'stable')
            stability_trend = {'stable': 0.5, 'improving': 1.0, 'declining': 0.0}.get(trend, 0.5)
            
            state = PatientState(
                age=age,
                gender=gender,
                anion_gap=anion_gap,
                sodium=sodium,
                potassium=potassium,
                urea_nitrogen=urea_nitrogen,
                heart_rate=heart_rate,
                sbp=sbp,
                dbp=dbp,
                spo2=spo2,
                chronic_disease_count=chronic_disease_count,
                medication_complexity=medication_complexity,
                stability_trend=stability_trend
            )
            
            return state
            
        except Exception as e:
            logger.error(f"状态提取失败: {e}")
            return None
    
    def _normalize_lab_value(self, value: float, min_normal: float, max_normal: float) -> float:
        """标准化实验室指标"""
        if value is None:
            return 0.5  # 缺失值用中位数
        
        # 将值映射到0-1范围，正常范围映射到0.3-0.7
        if value < min_normal:
            return max(0.0, 0.3 * (value / min_normal))
        elif value > max_normal:
            return min(1.0, 0.7 + 0.3 * ((value - max_normal) / max_normal))
        else:
            # 正常范围内线性映射到0.3-0.7
            return 0.3 + 0.4 * ((value - min_normal) / (max_normal - min_normal))
    
    def _normalize_vital(self, value: float, min_normal: float, max_normal: float) -> float:
        """标准化生命体征"""
        return self._normalize_lab_value(value, min_normal, max_normal)
    
    def state_to_vector(self, state: PatientState) -> np.ndarray:
        """将状态转换为向量"""
        return np.array([
            state.age,
            state.gender,
            state.anion_gap,
            state.sodium,
            state.potassium,
            state.urea_nitrogen,
            state.heart_rate,
            state.sbp,
            state.dbp,
            state.spo2,
            state.chronic_disease_count,
            state.medication_complexity,
            state.stability_trend
        ], dtype=np.float32)
    
    def calculate_reward(self, action: TreatmentAction, state: PatientState) -> RewardComponents:
        """计算奖励"""
        # 1. 共识得分 (基于专家经验规则)
        consensus_score = self._calculate_consensus_score(action, state)
        
        # 2. 稳定性指标
        stability_score = self._calculate_stability_score(action, state)
        
        # 3. 安全性评分
        safety_score = self._calculate_safety_score(action, state)
        
        # 4. 有效性评分
        effectiveness_score = self._calculate_effectiveness_score(action, state)
        
        # 计算总奖励
        total_reward = (
            self.reward_weights["consensus_score"] * consensus_score +
            self.reward_weights["stability_score"] * stability_score +
            self.reward_weights["safety_score"] * safety_score +
            self.reward_weights["effectiveness_score"] * effectiveness_score
        )
        
        return RewardComponents(
            consensus_score=consensus_score,
            stability_score=stability_score,
            safety_score=safety_score,
            effectiveness_score=effectiveness_score,
            total_reward=total_reward
        )
    
    def _calculate_consensus_score(self, action: TreatmentAction, state: PatientState) -> float:
        """计算共识得分"""
        # 基于病人状态和治疗动作的匹配度
        score = 0.5  # 基础分
        
        # 年龄因素
        if state.age > 0.8 and action.risk_level > 0.6:  # 高龄高风险
            score -= 0.3
        elif state.age < 0.5 and action.risk_level < 0.3:  # 年轻低风险
            score += 0.2
        
        # 稳定性因素
        if state.stability_trend > 0.7 and action.action_id == 0:  # 稳定时保守治疗
            score += 0.3
        elif state.stability_trend < 0.3 and action.action_id >= 3:  # 不稳定时积极治疗
            score += 0.2
        
        # 实验室指标异常
        abnormal_labs = sum([
            abs(state.anion_gap - 0.5) > 0.3,
            abs(state.sodium - 0.5) > 0.3,
            abs(state.potassium - 0.5) > 0.3
        ])
        
        if abnormal_labs >= 2 and action.action_id <= 1:  # 多项异常但治疗不足
            score -= 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _calculate_stability_score(self, action: TreatmentAction, state: PatientState) -> float:
        """计算稳定性指标"""
        # 基于当前稳定性和治疗强度的匹配
        base_stability = state.stability_trend
        
        # 治疗强度对稳定性的影响
        if action.risk_level > 0.5:  # 高风险治疗
            if base_stability < 0.3:  # 不稳定时需要积极治疗
                stability_improvement = 0.3
            else:  # 稳定时高风险治疗可能降低稳定性
                stability_improvement = -0.2
        else:  # 低风险治疗
            stability_improvement = 0.1  # 通常有助于稳定性
        
        final_stability = base_stability + stability_improvement
        return np.clip(final_stability, 0.0, 1.0)
    
    def _calculate_safety_score(self, action: TreatmentAction, state: PatientState) -> float:
        """计算安全性评分"""
        # 基础安全性 = 1 - 治疗风险
        base_safety = 1.0 - action.risk_level
        
        # 病人状态调整
        if state.age > 0.8:  # 高龄病人
            base_safety -= 0.1
        
        if state.chronic_disease_count > 0.5:  # 多种慢性病
            base_safety -= 0.1
        
        if state.medication_complexity > 0.6:  # 药物复杂
            base_safety -= 0.1
        
        return np.clip(base_safety, 0.0, 1.0)
    
    def _calculate_effectiveness_score(self, action: TreatmentAction, state: PatientState) -> float:
        """计算有效性评分"""
        # 基于病情严重程度和治疗强度的匹配
        severity_score = 1.0 - state.stability_trend  # 不稳定 = 严重
        
        # 治疗强度匹配度
        treatment_intensity = action.risk_level
        
        # 最佳匹配：严重病情需要强治疗，轻微病情需要轻治疗
        if severity_score > 0.7:  # 严重病情
            effectiveness = treatment_intensity  # 强治疗更有效
        elif severity_score < 0.3:  # 轻微病情
            effectiveness = 1.0 - treatment_intensity  # 轻治疗更有效
        else:  # 中等病情
            effectiveness = 1.0 - abs(severity_score - treatment_intensity)
        
        return np.clip(effectiveness, 0.0, 1.0)
    
    def reset(self, patient_file: str = None) -> np.ndarray:
        """重置环境"""
        if patient_file:
            self.load_patient_data(patient_file)
        
        if self.current_patient_data:
            self.current_state = self.extract_state_from_patient_data(self.current_patient_data)
            return self.state_to_vector(self.current_state)
        else:
            logger.warning("没有加载病人数据，使用默认状态")
            # 返回默认状态
            default_state = PatientState(
                age=0.65, gender=0.0, anion_gap=0.5, sodium=0.5, potassium=0.5,
                urea_nitrogen=0.5, heart_rate=0.5, sbp=0.5, dbp=0.5, spo2=0.5,
                chronic_disease_count=0.3, medication_complexity=0.4, stability_trend=0.5
            )
            self.current_state = default_state
            return self.state_to_vector(default_state)
    
    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一步"""
        if action_id >= len(self.actions):
            action_id = 0
        
        action = self.actions[action_id]
        reward_components = self.calculate_reward(action, self.current_state)
        
        # 记录历史
        self.state_history.append(self.state_to_vector(self.current_state))
        self.action_history.append(action_id)
        self.reward_history.append(reward_components.total_reward)
        
        # 生成下一状态 (简化：状态不变)
        next_state = self.state_to_vector(self.current_state)
        
        # 单步决策，每步都结束
        done = True
        
        info = {
            "action_name": action.action_name,
            "action_description": action.description,
            "reward_breakdown": {
                "consensus_score": reward_components.consensus_score,
                "stability_score": reward_components.stability_score,
                "safety_score": reward_components.safety_score,
                "effectiveness_score": reward_components.effectiveness_score
            },
            "patient_id": self.current_patient_data.get('subject_id', 'Unknown') if self.current_patient_data else 'Unknown'
        }
        
        return next_state, reward_components.total_reward, done, info
    
    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return len(self.actions)
    
    def get_state_space_size(self) -> int:
        """获取状态空间大小"""
        return 13  # 状态向量维度