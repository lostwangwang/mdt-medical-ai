"""
共识模块单元测试
文件路径: tests/test_consensus/test_consensus_matrix.py
作者: 姚刚
功能: 测试共识矩阵系统的各项功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

# 导入被测试的模块
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.core.data_models import (
    PatientState,
    RoleType,
    TreatmentOption,
    RoleOpinion,
    ConsensusResult,
)
from src.consensus.consensus_matrix import ConsensusMatrix
from src.consensus.role_agents import RoleAgent
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG


class TestConsensusMatrix:
    """共识矩阵测试类"""

    @pytest.fixture
    def sample_patient_state(self):
        """创建测试用患者状态"""
        return PatientState(
            patient_id="TEST_001",
            age=65,
            diagnosis="breast_cancer",
            stage="II",
            lab_results={"creatinine": 1.2, "hemoglobin": 11.5},
            vital_signs={"bp_systolic": 140, "heart_rate": 78},
            symptoms=["fatigue", "pain"],
            comorbidities=["diabetes", "hypertension"],
            psychological_status="anxious",
            quality_of_life_score=0.7,
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def consensus_matrix(self):
        """创建共识矩阵实例"""
        return ConsensusMatrix()

    @pytest.fixture
    def sample_role_opinions(self):
        """创建测试用角色意见"""
        opinions = {}

        # 肿瘤科医生：偏好手术和化疗
        opinions[RoleType.ONCOLOGIST] = RoleOpinion(
            role=RoleType.ONCOLOGIST,
            treatment_preferences={
                TreatmentOption.SURGERY: 0.8,
                TreatmentOption.CHEMOTHERAPY: 0.7,
                TreatmentOption.RADIOTHERAPY: 0.5,
                TreatmentOption.IMMUNOTHERAPY: 0.3,
                TreatmentOption.PALLIATIVE_CARE: -0.2,
                TreatmentOption.WATCHFUL_WAITING: -0.5,
            },
            reasoning="Surgery and chemotherapy offer best survival outcomes",
            confidence=0.9,
            concerns=["surgical_risks", "chemotherapy_toxicity"],
        )

        # 放射科医生：关注放射治疗
        opinions[RoleType.RADIOLOGIST] = RoleOpinion(
            role=RoleType.RADIOLOGIST,
            treatment_preferences={
                TreatmentOption.SURGERY: 0.0,
                TreatmentOption.CHEMOTHERAPY: 0.0,
                TreatmentOption.RADIOTHERAPY: 0.9,
                TreatmentOption.IMMUNOTHERAPY: 0.0,
                TreatmentOption.PALLIATIVE_CARE: 0.0,
                TreatmentOption.WATCHFUL_WAITING: 0.0,
            },
            reasoning="Radiotherapy is my specialty and offers targeted treatment",
            confidence=0.95,
            concerns=["radiation_side_effects"],
        )

        # 护士：关注可行性
        opinions[RoleType.NURSE] = RoleOpinion(
            role=RoleType.NURSE,
            treatment_preferences={
                TreatmentOption.SURGERY: 0.6,
                TreatmentOption.CHEMOTHERAPY: 0.4,
                TreatmentOption.RADIOTHERAPY: 0.7,
                TreatmentOption.IMMUNOTHERAPY: 0.5,
                TreatmentOption.PALLIATIVE_CARE: 0.8,
                TreatmentOption.WATCHFUL_WAITING: 0.3,
            },
            reasoning="Focus on patient care feasibility and comfort",
            confidence=0.8,
            concerns=["patient_compliance", "care_complexity"],
        )

        # 心理师：关注心理健康
        opinions[RoleType.PSYCHOLOGIST] = RoleOpinion(
            role=RoleType.PSYCHOLOGIST,
            treatment_preferences={
                TreatmentOption.SURGERY: 0.2,
                TreatmentOption.CHEMOTHERAPY: -0.1,
                TreatmentOption.RADIOTHERAPY: 0.4,
                TreatmentOption.IMMUNOTHERAPY: 0.3,
                TreatmentOption.PALLIATIVE_CARE: 0.6,
                TreatmentOption.WATCHFUL_WAITING: 0.5,
            },
            reasoning="Minimize psychological burden and maintain quality of life",
            confidence=0.7,
            concerns=["anxiety", "depression", "family_stress"],
        )

        # 患者代言人：关注患者权益
        opinions[RoleType.PATIENT_ADVOCATE] = RoleOpinion(
            role=RoleType.PATIENT_ADVOCATE,
            treatment_preferences={
                TreatmentOption.SURGERY: 0.0,
                TreatmentOption.CHEMOTHERAPY: 0.0,
                TreatmentOption.RADIOTHERAPY: 0.0,
                TreatmentOption.IMMUNOTHERAPY: 0.0,
                TreatmentOption.PALLIATIVE_CARE: 0.7,
                TreatmentOption.WATCHFUL_WAITING: 0.6,
            },
            reasoning="Focus on patient rights and quality of life",
            confidence=0.6,
            concerns=["patient_autonomy", "informed_consent"],
        )

        return opinions

    def test_consensus_matrix_initialization(self, consensus_matrix):
        """测试共识矩阵初始化"""
        assert isinstance(consensus_matrix, ConsensusMatrix)
        assert hasattr(consensus_matrix, "agents")
        assert hasattr(consensus_matrix, "rag_system")
        assert len(consensus_matrix.agents) == len(RoleType)

    def test_build_consensus_matrix(self, consensus_matrix, sample_role_opinions):
        """测试构建共识矩阵"""
        matrix = consensus_matrix._build_consensus_matrix(sample_role_opinions)

        # 检查矩阵维度
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix.index) == len(TreatmentOption)
        assert len(matrix.columns) == len(sample_role_opinions)

        # 检查数值范围
        assert matrix.min().min() >= -1.0
        assert matrix.max().max() <= 1.0

        # 检查特定值
        oncologist_surgery_score = matrix.loc["surgery", "oncologist"]
        assert oncologist_surgery_score == 0.8

    def test_aggregate_scores(self, consensus_matrix, sample_role_opinions):
        """测试评分聚合"""
        aggregated = consensus_matrix._aggregate_scores(sample_role_opinions)

        # 检查返回格式
        assert isinstance(aggregated, dict)
        assert len(aggregated) == len(TreatmentOption)

        # 检查所有治疗方案都有评分
        for treatment in TreatmentOption:
            assert treatment in aggregated
            assert isinstance(aggregated[treatment], float)
            assert -1.0 <= aggregated[treatment] <= 1.0

    def test_identify_conflicts(self, consensus_matrix, sample_role_opinions):
        """测试冲突识别"""
        conflicts = consensus_matrix._identify_conflicts(sample_role_opinions)

        # 检查返回格式
        assert isinstance(conflicts, list)

        # 检查冲突项格式
        for conflict in conflicts:
            assert isinstance(conflict, dict)
            assert "treatment" in conflict
            assert "variance" in conflict
            assert "min_score" in conflict
            assert "max_score" in conflict
            assert "conflicting_roles" in conflict

            # 检查方差大于阈值
            assert conflict["variance"] > 0.5

    def test_identify_agreements(self, consensus_matrix, sample_role_opinions):
        """测试一致意见识别"""
        agreements = consensus_matrix._identify_agreements(sample_role_opinions)

        # 检查返回格式
        assert isinstance(agreements, list)

        # 检查一致项格式
        for agreement in agreements:
            assert isinstance(agreement, dict)
            assert "treatment" in agreement
            assert "consensus_score" in agreement
            assert "agreement_strength" in agreement
            assert "unanimous" in agreement

            # 检查一致性强度
            assert 0.0 <= agreement["agreement_strength"] <= 1.0

    def test_generate_consensus_direct(self, consensus_matrix, sample_patient_state):
        """测试直接共识生成"""
        with patch.object(
            consensus_matrix, "_generate_direct_consensus"
        ) as mock_direct:
            mock_result = ConsensusResult(
                consensus_matrix=pd.DataFrame(),
                role_opinions={},
                aggregated_scores={TreatmentOption.SURGERY: 0.8},
                conflicts=[],
                agreements=[],
                dialogue_summary=None,
                timestamp=datetime.now(),
            )
            mock_direct.return_value = mock_result

            result = consensus_matrix.generate_consensus(
                sample_patient_state, use_dialogue=False
            )

            assert isinstance(result, ConsensusResult)
            mock_direct.assert_called_once()

    def test_analyze_consensus_patterns(self, consensus_matrix):
        """测试共识模式分析"""
        # 创建模拟共识结果
        mock_consensus_result = ConsensusResult(
            consensus_matrix=pd.DataFrame(),
            role_opinions={},
            aggregated_scores={
                TreatmentOption.SURGERY: 0.8,
                TreatmentOption.CHEMOTHERAPY: 0.6,
                TreatmentOption.RADIOTHERAPY: 0.4,
            },
            conflicts=[
                {
                    "treatment": TreatmentOption.SURGERY,
                    "variance": 0.6,
                    "conflicting_roles": ["nurse", "psychologist"],
                }
            ],
            agreements=[
                {
                    "treatment": TreatmentOption.CHEMOTHERAPY,
                    "consensus_score": 0.7,
                    "agreement_strength": 0.8,
                }
            ],
            dialogue_summary=None,
            timestamp=datetime.now(),
        )

        analysis = consensus_matrix.analyze_consensus_patterns(mock_consensus_result)

        # 检查分析结果格式
        assert isinstance(analysis, dict)
        assert "overall_consensus_level" in analysis
        assert "role_influence_analysis" in analysis
        assert "treatment_polarization" in analysis
        assert "decision_complexity" in analysis
        assert "recommendation_strength" in analysis

        # 检查数值范围
        assert 0.0 <= analysis["overall_consensus_level"] <= 1.0

    def test_export_consensus_report(self, consensus_matrix):
        """测试共识报告导出"""
        # 创建模拟共识结果
        mock_consensus_result = ConsensusResult(
            consensus_matrix=pd.DataFrame(
                {"oncologist": [0.8, 0.6], "nurse": [0.5, 0.7]},
                index=["surgery", "chemotherapy"],
            ),
            role_opinions={},
            aggregated_scores={
                TreatmentOption.SURGERY: 0.8,
                TreatmentOption.CHEMOTHERAPY: 0.6,
            },
            conflicts=[],
            agreements=[],
            dialogue_summary=None,
            timestamp=datetime.now(),
        )

        report = consensus_matrix.export_consensus_report(mock_consensus_result)

        # 检查报告内容
        assert isinstance(report, str)
        assert "Medical Team Consensus Report" in report
        assert "surgery" in report
        assert "chemotherapy" in report
        assert "0.800" in report  # 检查数值格式


class TestRoleAgent:
    """角色智能体测试类"""

    @pytest.fixture
    def oncologist_agent(self):
        """创建肿瘤科医生智能体"""
        return RoleAgent(RoleType.ONCOLOGIST)

    @pytest.fixture
    def nurse_agent(self):
        """创建护士智能体"""
        return RoleAgent(RoleType.NURSE)

    @pytest.fixture
    def sample_patient_state(self):
        """创建测试用患者状态"""
        return PatientState(
            patient_id="TEST_AGENT",
            age=65,
            diagnosis="breast_cancer",
            stage="II",
            lab_results={"creatinine": 1.2, "hemoglobin": 11.5},
            vital_signs={"bp_systolic": 140, "heart_rate": 78},
            symptoms=["fatigue"],
            comorbidities=["diabetes"],
            psychological_status="stable",
            quality_of_life_score=0.7,
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def sample_knowledge(self):
        """创建测试用知识"""
        return {
            "guidelines": ["NCCN recommends surgery for stage II breast cancer"],
            "success_rates": {"5_year": 0.85},
            "contraindications": [],
            "similar_cases": [],
        }

    def test_role_agent_initialization(self, oncologist_agent):
        """测试角色智能体初始化"""
        assert oncologist_agent.role == RoleType.ONCOLOGIST
        assert isinstance(oncologist_agent.specialization, dict)
        assert "primary_concerns" in oncologist_agent.specialization
        assert "weight_factors" in oncologist_agent.specialization
        assert oncologist_agent.dialogue_history == []
        assert oncologist_agent.current_stance == {}

    def test_get_specialization(self, oncologist_agent, nurse_agent):
        """测试获取专业特征"""
        onco_spec = oncologist_agent.specialization
        nurse_spec = nurse_agent.specialization

        # 检查肿瘤科医生特征
        assert "survival" in onco_spec["primary_concerns"]
        assert "treatment_efficacy" in onco_spec["primary_concerns"]

        # 检查护士特征
        assert "feasibility" in nurse_spec["primary_concerns"]
        assert "patient_compliance" in nurse_spec["primary_concerns"]

        # 检查不同角色有不同的特征
        assert onco_spec != nurse_spec

    def test_generate_initial_opinion(
        self, oncologist_agent, sample_patient_state, sample_knowledge
    ):
        """测试生成初始意见"""
        opinion = oncologist_agent.generate_initial_opinion(
            sample_patient_state, sample_knowledge
        )

        # 检查返回格式
        assert isinstance(opinion, RoleOpinion)
        assert opinion.role == RoleType.ONCOLOGIST
        assert isinstance(opinion.treatment_preferences, dict)
        assert isinstance(opinion.reasoning, str)
        assert isinstance(opinion.confidence, float)
        assert isinstance(opinion.concerns, list)

        # 检查数值范围
        assert 0.0 <= opinion.confidence <= 1.0
        for score in opinion.treatment_preferences.values():
            assert -1.0 <= score <= 1.0

        # 检查肿瘤科医生倾向于手术和化疗
        assert opinion.treatment_preferences[TreatmentOption.SURGERY] > 0
        assert opinion.treatment_preferences[TreatmentOption.CHEMOTHERAPY] > 0

    def test_calculate_treatment_preferences(
        self, oncologist_agent, sample_patient_state, sample_knowledge
    ):
        """测试计算治疗偏好"""
        preferences = oncologist_agent._calculate_treatment_preferences(
            sample_patient_state, sample_knowledge
        )

        # 检查返回格式
        assert isinstance(preferences, dict)
        assert len(preferences) == len(TreatmentOption)

        # 检查所有治疗方案都有评分
        for treatment in TreatmentOption:
            assert treatment in preferences
            assert -1.0 <= preferences[treatment] <= 1.0

    def test_evaluate_treatment_for_role(
        self, oncologist_agent, nurse_agent, sample_patient_state
    ):
        """测试角色特异性治疗评估"""
        # 肿瘤科医生评估
        onco_surgery_score = oncologist_agent._evaluate_treatment_for_role(
            TreatmentOption.SURGERY, sample_patient_state
        )

        # 护士评估
        nurse_surgery_score = nurse_agent._evaluate_treatment_for_role(
            TreatmentOption.SURGERY, sample_patient_state
        )

        # 检查评分范围
        assert -1.0 <= onco_surgery_score <= 1.0
        assert -1.0 <= nurse_surgery_score <= 1.0

        # 肿瘤科医生通常对手术评分更高
        assert onco_surgery_score > nurse_surgery_score

    def test_adjust_score_by_patient_state(
        self, oncologist_agent, sample_patient_state
    ):
        """测试根据患者状态调整评分"""
        base_score = 0.8

        # 测试年龄调整
        elderly_patient = PatientState(
            patient_id="ELDERLY",
            age=85,  # 高龄
            diagnosis="breast_cancer",
            stage="II",
            lab_results={},
            vital_signs={},
            symptoms=[],
            comorbidities=[],
            psychological_status="stable",
            quality_of_life_score=0.5,
            timestamp=datetime.now(),
        )

        adjusted_score = oncologist_agent._adjust_score_by_patient_state(
            base_score, TreatmentOption.SURGERY, elderly_patient
        )

        # 高龄患者的手术评分应该降低
        assert adjusted_score < base_score

    def test_calculate_confidence(self, oncologist_agent, sample_patient_state):
        """测试置信度计算"""
        confidence = oncologist_agent._calculate_confidence(sample_patient_state)

        # 检查置信度范围
        assert 0.0 <= confidence <= 1.0

        # 测试数据完整性对置信度的影响
        incomplete_patient = PatientState(
            patient_id="INCOMPLETE",
            age=65,
            diagnosis="breast_cancer",
            stage="II",
            lab_results={},  # 缺少实验室数据
            vital_signs={},
            symptoms=[],
            comorbidities=[],
            psychological_status="stable",
            quality_of_life_score=0.7,
            timestamp=datetime.now(),
        )

        incomplete_confidence = oncologist_agent._calculate_confidence(
            incomplete_patient
        )
        assert incomplete_confidence < confidence

    def test_identify_concerns(self, oncologist_agent, sample_patient_state):
        """测试识别关注点"""
        concerns = oncologist_agent._identify_concerns(sample_patient_state)

        # 检查返回格式
        assert isinstance(concerns, list)
        assert len(concerns) <= 3  # 限制最多3个关注点

        # 所有关注点都应该是字符串
        for concern in concerns:
            assert isinstance(concern, str)

    def test_update_stance_based_on_dialogue(self, oncologist_agent):
        """测试基于对话更新立场"""
        # 模拟对话消息
        from src.core.data_models import DialogueMessage

        dialogue_context = [
            DialogueMessage(
                role=RoleType.NURSE,
                content="I recommend chemotherapy for this patient",
                timestamp=datetime.now(),
                message_type="response",
                referenced_roles=[],
                evidence_cited=[],
                treatment_focus=TreatmentOption.CHEMOTHERAPY,
            ),
            DialogueMessage(
                role=RoleType.PSYCHOLOGIST,
                content="I also recommend chemotherapy",
                timestamp=datetime.now(),
                message_type="response",
                referenced_roles=[],
                evidence_cited=[],
                treatment_focus=TreatmentOption.CHEMOTHERAPY,
            ),
        ]

        # 设置初始立场
        oncologist_agent.current_stance = {TreatmentOption.CHEMOTHERAPY: 0.5}

        # 更新立场
        oncologist_agent.update_stance_based_on_dialogue(dialogue_context)

        # 检查立场是否有所调整
        updated_stance = oncologist_agent.current_stance[TreatmentOption.CHEMOTHERAPY]
        assert updated_stance >= 0.5  # 应该不降低


class TestDialogueManager:
    """对话管理器测试类"""

    @pytest.fixture
    def rag_system(self):
        """创建RAG系统"""
        return MedicalKnowledgeRAG()

    @pytest.fixture
    def dialogue_manager(self, rag_system):
        """创建对话管理器"""
        return MultiAgentDialogueManager(rag_system)

    @pytest.fixture
    def sample_patient_state(self):
        """创建测试用患者状态"""
        return PatientState(
            patient_id="TEST_DIALOGUE",
            age=65,
            diagnosis="breast_cancer",
            stage="II",
            lab_results={"creatinine": 1.2},
            vital_signs={"bp_systolic": 140},
            symptoms=["fatigue"],
            comorbidities=["diabetes"],
            psychological_status="stable",
            quality_of_life_score=0.7,
            timestamp=datetime.now(),
        )

    def test_dialogue_manager_initialization(self, dialogue_manager):
        """测试对话管理器初始化"""
        assert isinstance(dialogue_manager, MultiAgentDialogueManager)
        assert len(dialogue_manager.agents) == len(RoleType)
        assert dialogue_manager.current_round == 0
        assert dialogue_manager.max_rounds == 5
        assert dialogue_manager.dialogue_rounds == []

    def test_initialize_discussion(self, dialogue_manager, sample_patient_state):
        """测试初始化讨论"""
        dialogue_manager._initialize_discussion(sample_patient_state)

        # 检查是否创建了初始轮次
        assert len(dialogue_manager.dialogue_rounds) == 1
        initial_round = dialogue_manager.dialogue_rounds[0]

        # 检查初始消息数量
        assert len(initial_round.messages) == len(RoleType)

        # 检查每个角色都有发言
        message_roles = {msg.role for msg in initial_round.messages}
        assert len(message_roles) == len(RoleType)

    def test_get_recommendation_phrase(self, dialogue_manager):
        """测试推荐措辞"""
        test_cases = [
            (0.8, "strongly recommend"),
            (0.5, "recommend"),
            (0.0, "have mixed feelings about"),
            (-0.5, "have concerns about"),
            (-0.8, "strongly advise against"),
        ]

        for score, expected_phrase in test_cases:
            phrase = dialogue_manager._get_recommendation_phrase(score)
            assert phrase == expected_phrase

    def test_select_focus_treatment(self, dialogue_manager):
        """测试选择焦点治疗方案"""
        # 空对话轮次时的默认选择
        focus = dialogue_manager._select_focus_treatment()
        assert focus == TreatmentOption.SURGERY  # 默认值

    def test_check_convergence(self, dialogue_manager):
        """测试收敛检查"""
        # 初始状态应该未收敛
        assert not dialogue_manager._check_convergence()

        # 模拟多轮对话后的状态
        for agent in dialogue_manager.agents.values():
            agent.current_stance = {
                TreatmentOption.SURGERY: 0.8,  # 强烈立场
                TreatmentOption.CHEMOTHERAPY: 0.7,
            }

        # 创建模拟对话轮次 - 需要至少2个轮次才能收敛
        from src.core.data_models import DialogueRound

        dialogue_manager.dialogue_rounds = [
            DialogueRound(0, [], None, "discussing"),
            DialogueRound(1, [], None, "discussing")
        ]

        # 现在应该收敛
        assert dialogue_manager._check_convergence()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
