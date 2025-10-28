"""
多智能体对话管理器
文件路径: src/consensus/dialogue_manager.py
作者: 姚刚
功能: 管理医疗团队多智能体间的对话协商过程
"""

from os import name
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


from ..core.data_models import (
    RoleType,
    TreatmentOption,
    PatientState,
    DialogueMessage,
    DialogueRound,
    ConsensusResult,
    RoleOpinion,
)
from .role_agents import RoleAgent
from ..knowledge.rag_system import MedicalKnowledgeRAG
from ..utils.llm_interface import LLMInterface, LLMConfig
from .calcuate_consensus import CalculateConsensus
from experiments.medqa_types import MedicalQuestionState, QuestionOption

logger = logging.getLogger(__name__)


class MultiAgentDialogueManager:
    """多智能体对话管理器"""

    def __init__(
        self,
        rag_system: MedicalKnowledgeRAG,
        llm_interface: Optional[LLMInterface] = None,
    ):
        self.agents = {
            role: RoleAgent(role, llm_interface=llm_interface) for role in RoleType
        }
        self.consensus_calculator = CalculateConsensus()
        self.rag_system = rag_system
        self.dialogue_rounds = []
        self.current_round = 0
        self.max_rounds = 5
        self.convergence_threshold = 0.8

    def conduct_mdt_discussion_medqa(
        self,
        question_state: MedicalQuestionState,
        question_options: List[QuestionOption],
    ) -> ConsensusResult:
        """
        进行MDT讨论

        讨论流程:
        1. 初始化讨论 - 各角色基于RAG检索的医学知识生成初始意见
        2. 多轮对话协商 - 角色间就治疗方案进行结构化讨论
        3. 立场更新 - 基于其他角色观点调整自己的立场
        4. 争议聚焦 - 针对分歧较大的治疗方案深入讨论
        5. 共识达成 - 生成最终的治疗建议和共识结果
        """
        print("Question options in dialogue manager:", question_options)
        logger.info(f"Starting MDT discussion for question {question_state}")

        # 初始化对话 - 各角色生成基于证据的初始意见
        opinions_list = self._initialize_discussion_medqa(
            question_state, question_options
        )
        logger.info(f"Initial opinions: {opinions_list}")
        # 将意见列表转换为字典，方便按角色快速访问
        opinions_dict = {opinion.role: opinion for opinion in opinions_list}
        logger.info(f"Initial opinions dict: {opinions_dict}")
        # 进行多轮对话协商
        while (
            self.current_round < self.max_rounds
            and not self._check_discussion_convergence(opinions_dict)
        ):
            self.current_round += 1
            logger.info(f"Starting dialogue round {self.current_round}")

            # 进行一轮结构化对话
            current_round = self._conduct_dialogue_round_medqa(
                question_state, question_options, opinions_dict
            )
            logger.info(f"dialogue round {self.current_round}: {current_round}")
            self.dialogue_rounds.append(current_round)
            logger.info(f"previous opinions dict: {opinions_dict}")
            # 基于对话内容更新各角色立场
            new_opinions_dict = self._update_agent_opinions(
                patient_state, current_round, opinions_dict, treatment_options
            )
            logger.info(f"Updated opinions dict: {new_opinions_dict}")
        logger.info(f"the last round: {self.current_round}")
        logger.info(f"final opinions dict: {new_opinions_dict}")
        logger.info("达成共识!!!")
        logger.info("\n==生成共识结果开始：==")
        # 生成最终共识结果
        final_result = self._generate_final_consensus(patient_state)

        logger.info(f"MDT discussion completed after {self.current_round} rounds")
        logger.info(f"Final consensus result: {final_result}")
        return final_result
    
    def _conduct_dialogue_round_medqa(
        self, question_state: MedicalQuestionState,
        question_options: List[QuestionOption],
        opinions_dict: Dict[RoleType, RoleOpinion],
    ) -> DialogueRound:
        """进行一轮结构化对话"""
        current_round = DialogueRound(
            round_number=self.current_round,
            messages=[],
            focus_treatment=self._select_focus_treatment_medqa(question_options, opinions_dict),
            consensus_status="discussing",
        )

        current_round = self._conduct_sequential_presentation_medqa(
            current_round, question_state, question_options, opinions_dict
        )
        logger.info(
            f"Generated messages for round {self.current_round}: {current_round}"
        )
        return current_round
        pass
    def _select_focus_treatment_medqa(
        self, question_options: List[QuestionOption],
        opinions_dict: Dict[RoleType, RoleOpinion],
    ) -> TreatmentOption:
        """根据当前意见选择聚焦治疗方案"""
        # 简单实现：选择当前意见中最高分的治疗方案
        last_round = self.dialogue_rounds[-1]
        treatment_mentions = {}
        for message in last_round.messages:
            treatment = message.treatment_focus
            if treatment not in treatment_mentions:
                treatment_mentions[treatment] = 0
            treatment_mentions[treatment] += 1
        max_mentions = max(treatment_mentions.values())
        focus_treatment = next(
            treatment
            for treatment, mentions in treatment_mentions.items()
            if mentions == max_mentions
        )
        return focus_treatment

    def _initialize_discussion_medqa(
        self, question_state: MedicalQuestionState,
        question_options: List[QuestionOption],
    ) -> None:
        # 生成各角色的初始意见
        initial_round = DialogueRound(
            round_number=0,
            messages=[],
            focus_treatment=None,
            consensus_status="discussing",
        )

        # 遍历每个角色智能体，生成其初始意见并构建首轮对话消息
        opinions_list = []
        for role, agent in self.agents.items():
            # 构建初始意见
            opinion = agent.generate_initial_opinion_medqa(
                question_state, question_options
            )
            logger.info(f"Generated initial opinion for {role}: {opinion}")
            opinions_list.append(opinion)

            # 生成初始发言
            initial_message = self._create_initial_message_medqa(
                agent, opinion, question_state, question_options
            )

            logger.info(
                f"Generated initial message for {role}: {initial_message.content}"
            )

            initial_round.messages.append(initial_message)

        self.dialogue_rounds.append(initial_round)
        logger.info(
            f"Initialized discussion with {len(initial_round.messages)} initial opinions"
        )
        return opinions_list

    def _create_initial_message_medqa(
        self, agent: RoleAgent, opinion: RoleOpinion, question_state: MedicalQuestionState, question_options: List[QuestionOption]
    ) -> DialogueMessage:
        """创建初始消息"""
        # 智能选择治疗方案（仅仅是最高分）
        focus_question = self._select_focus_question_for_role_medqa(
            agent, opinion, question_state, question_options
        )

        # 尝试使用LLM生成个性化初始消息
        content = self._generate_llm_initial_message_meqa(
            agent, opinion, question_state, focus_question, question_options
        )
        logger.info(f"测试Generated initial message for {agent.role}: {content}")

        return DialogueMessage(
            role=agent.role,
            content=content,
            timestamp=datetime.now(),
            message_type="initial_opinion",
            treatment_focus=focus_question,
        )

    def _generate_llm_initial_message_meqa(
        self, agent: RoleAgent, opinion: RoleOpinion, question_state: MedicalQuestionState, focus_treatment: QuestionOption, question_options: List[QuestionOption]
    ):
        """使用LLM生成个性化初始消息"""
        if not hasattr(agent, "llm_interface") or not agent.llm_interface:
            return ""

        try:
            # 使用专门的治疗推理方法生成基础推理
            if hasattr(agent.llm_interface, "generate_focus_treatment_reasoning_meqa"):
                reasoning = agent.llm_interface.generate_focus_treatment_reasoning_meqa(
                    question_state=question_state,
                    role=agent.role,
                    opinion=opinion,
                    treatment_option=focus_treatment,
                    question_options=question_options,
                )
                logger.info(
                    f"生成初始化发言, 治疗选项: {focus_treatment}, 推理: {reasoning}"
                )
                if reasoning and len(reasoning.strip()) > 0:
                    # 基于推理生成MDT发言格式的消息
                    content = self._format_reasoning_as_mdt_message(
                        agent, reasoning, focus_treatment, opinion
                    )
                    logger.debug(
                        f"Generated LLM initial message for {agent.role.value}: {content}"
                    )
                    return content

        except Exception as e:
            logger.warning(
                f"LLM initial message generation failed for {agent.role}: {e}"
            )

        return ""

    def _select_focus_question_for_role_medqa(
        self, agent: RoleAgent, opinion: RoleOpinion, question_state: MedicalQuestionState, question_options: List[QuestionOption]
    ) -> QuestionOption:
        """选择当前角色关注的偏好最高的问题选项"""
        prefs = opinion.treatment_preferences  # 格式：{"A": 0.9, "B": -0.2, ...}
        
        # 1. 过滤出有效选项（确保选项在偏好字典中存在）
        valid_options = [
            option for option in question_options 
            # option.name 是选项标识（如"A"/"B"），需与prefs的键匹配
            if option.name in prefs  
            # 同时确保该选项在问题的原始选项中（双重校验，可选）
            and option.name in question_state.options  
        ]
        
        if not valid_options:
            raise ValueError("没有找到有效的选项偏好映射")
        
        # 2. 找到最高得分
        max_score = max(prefs[option.name] for option in valid_options)
        
        # 3. 筛选出所有得分等于最高分的选项（处理并列情况）
        top_options = [
            option for option in valid_options 
            if prefs[option.name] == max_score
        ]
        
        # 4. 如果有多个并列最高分，返回第一个；否则返回唯一的最高分选项
        focus_question = top_options[0]
        
        logger.info(f"最高得分: {max_score}")
        logger.info(f"选中的选项: {focus_question.name} ({question_state.options[focus_question.name]})")  # 打印选项标识和内容
        return focus_question


    def conduct_mdt_discussion(
        self, patient_state: PatientState, treatment_options: List[TreatmentOption]
    ) -> ConsensusResult:
        """
        进行MDT讨论

        讨论流程:
        1. 初始化讨论 - 各角色基于RAG检索的医学知识生成初始意见
        2. 多轮对话协商 - 角色间就治疗方案进行结构化讨论
        3. 立场更新 - 基于其他角色观点调整自己的立场
        4. 争议聚焦 - 针对分歧较大的治疗方案深入讨论
        5. 共识达成 - 生成最终的治疗建议和共识结果
        """
        logger.info(f"Starting MDT discussion for patient {patient_state.patient_id}")

        # 初始化对话 - 各角色生成基于证据的初始意见
        opinions_list = self._initialize_discussion(patient_state, treatment_options)
        logger.info(f"Initial opinions: {opinions_list}")
        # 将意见列表转换为字典，方便按角色快速访问
        opinions_dict = {opinion.role: opinion for opinion in opinions_list}
        logger.info(f"Initial opinions dict: {opinions_dict}")
        # 进行多轮对话协商
        while (
            self.current_round < self.max_rounds
            and not self._check_discussion_convergence(opinions_dict)
        ):
            self.current_round += 1
            logger.info(f"Starting dialogue round {self.current_round}")

            # 进行一轮结构化对话
            current_round = self._conduct_dialogue_round(patient_state, opinions_dict)
            logger.info(f"dialogue round {self.current_round}: {current_round}")
            self.dialogue_rounds.append(current_round)
            logger.info(f"previous opinions dict: {opinions_dict}")
            # 基于对话内容更新各角色立场
            new_opinions_dict = self._update_agent_opinions(
                patient_state, current_round, opinions_dict, treatment_options
            )
            logger.info(f"Updated opinions dict: {new_opinions_dict}")
        logger.info(f"the last round: {self.current_round}")
        logger.info(f"final opinions dict: {new_opinions_dict}")
        logger.info("达成共识!!!")
        logger.info("\n==生成共识结果开始：==")
        # 生成最终共识结果
        final_result = self._generate_final_consensus(patient_state)

        logger.info(f"MDT discussion completed after {self.current_round} rounds")
        logger.info(f"Final consensus result: {final_result}")
        return final_result

    def _check_discussion_convergence(
        self, opinions_dict: Dict[RoleType, RoleOpinion]
    ) -> bool:
        """
        检查讨论是否收敛
        """
        if self.current_round <= 2:
            return False
        self.consensus_calculator.build_weighted_matrix(opinions_dict)
        self.consensus_calculator.summarize()
        self.consensus_calculator.compute_kendalls_w()
        df, W, p_value, consensus = self.consensus_calculator.summarize()
        return consensus

    def _update_agent_opinions(
        self,
        patient_state: PatientState,
        current_round: DialogueRound,
        opinions_dict: Dict[RoleType, RoleOpinion],
        treatment_options: List[TreatmentOption],
    ) -> Dict[RoleType, RoleOpinion]:
        """
        更新各角色立场
        - 基于当前轮对话内容，更新每个角色的治疗意见
        - 以及每个角色的治疗偏好、置信度、治疗意见
        - 考虑其他角色的观点，调整自己的立场
        """
        new_opinions_dict: Dict[RoleType, RoleOpinion] = {}

        # 大模型分析当前的治疗意见
        for role, agent in self.agents.items():
            # 根据当前的对话对话内容以及其他角色的对话进行分析,更新角色的治疗偏好和治疗意见
            # current_dialogue = current_round.messages[-1]
            previous_opinion = opinions_dict[role.value]
            logger.info(f"Previous opinion for {role.value}: {previous_opinion}")
            new_opintion = agent._update_agent_opinions_and_preferences(
                patient_state, current_round, previous_opinion, treatment_options
            )
            new_opinions_dict[role.value] = new_opintion

        logger.info(f"Updated opinions dict: {new_opinions_dict}")
        return new_opinions_dict

    def _initialize_discussion(
        self, patient_state: PatientState, treatment_options: List[TreatmentOption]
    ) -> None:
        """
        初始化讨论
        - RAG知识检索 ：从医学知识库中检索与患者状态相关的初始评估知识
        - 角色意见生成 ：每个医疗角色（肿瘤医生、护士、心理医生等）基于检索到的知识生成初始治疗意见
        - 智能治疗方案选择 ：不是简单选择最高分方案，而是考虑：
            - 角色专业特长偏好
            - 患者复杂度（年龄、生活质量、心理状态）
            - 治疗方案可行性评估
        """
        # 获取相关医学知识 这里怎么检索? 从RAG系统中检索与患者状态相关的初始评估知识,要这里检索?我这里还没搞好，我先不专注这里
        initial_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "initial_assessment"
        )

        # 生成各角色的初始意见
        # 这里怎么生成? 调用每个智能体的generate_initial_opinion方法，传入患者状态和初始知识
        initial_round = DialogueRound(
            round_number=0,
            messages=[],
            focus_treatment=None,
            consensus_status="discussing",
        )
        # 遍历每个角色智能体，生成其初始意见并构建首轮对话消息
        opinions_list = []
        for role, agent in self.agents.items():
            # 构建初始意见
            opinion = agent.generate_initial_opinion(
                patient_state, initial_knowledge, treatment_options
            )
            logger.info(f"Generated initial opinion for {role}: {opinion}")
            opinions_list.append(opinion)

            # 生成初始发言
            initial_message = self._create_initial_message(
                agent, opinion, patient_state, treatment_options
            )

            logger.info(
                f"Generated initial message for {role}: {initial_message.content}"
            )

            initial_round.messages.append(initial_message)

        self.dialogue_rounds.append(initial_round)
        logger.info(
            f"Initialized discussion with {len(initial_round.messages)} initial opinions"
        )
        return opinions_list

    def _create_initial_message(
        self,
        agent: RoleAgent,
        opinion: RoleOpinion,
        patient_state: PatientState,
        treatment_options: List[TreatmentOption],
    ) -> DialogueMessage:
        """创建初始消息"""
        # 智能选择治疗方案（仅仅是最高分）
        focus_treatment = self._select_focus_treatment_for_role(
            agent, opinion, patient_state, treatment_options
        )
        focus_treatment = TreatmentOption(focus_treatment)
        logger.info(f"focus_treament: {focus_treatment}")

        # 尝试使用LLM生成个性化初始消息
        content = self._generate_llm_initial_message(
            agent, opinion, patient_state, focus_treatment, treatment_options
        )
        logger.info(f"测试Generated initial message for {agent.role}: {content}")
        # 如果LLM生成失败，降级到模板化方法
        if not content:
            content = self._generate_template_initial_message(
                agent, opinion, focus_treatment
            )

        return DialogueMessage(
            role=agent.role,
            content=content,
            timestamp=datetime.now(),
            message_type="initial_opinion",
            treatment_focus=focus_treatment,
        )

    def _select_focus_treatment_for_role(
        self,
        agent: RoleAgent,
        opinion,
        patient_state: PatientState,
        treatment_options: List[TreatmentOption],
    ) -> TreatmentOption:
        """为特定角色智能选择焦点治疗方案"""
        prefs = opinion.treatment_preferences

        # 1. 过滤掉明显不合适的治疗方案（评分过低）
        viable_treatments = {
            treatment: score
            for treatment, score in prefs.items()
            if score > -0.5  # 排除强烈反对的方案
        }

        if not viable_treatments:
            # 如果所有方案都被强烈反对，选择最不反对的
            return max(prefs.items(), key=lambda x: x[1])[0]

        # 2. 考虑角色专业特长
        role_preferred_treatments = self._get_role_preferred_treatments(agent.role)

        # 3. 智能选择策略
        return self._apply_intelligent_selection_strategy(
            viable_treatments, role_preferred_treatments, patient_state, agent.role
        )

    def _get_role_preferred_treatments(self, role: RoleType) -> List[TreatmentOption]:
        """获取角色偏好的治疗方案类型"""
        role_preferences = {
            RoleType.ONCOLOGIST: [
                TreatmentOption.CHEMOTHERAPY,
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.SURGERY,
                TreatmentOption.IMMUNOTHERAPY,
            ],
            RoleType.NURSE: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.CHEMOTHERAPY,  # 护理角度关注
                TreatmentOption.RADIOTHERAPY,
            ],
            RoleType.PSYCHOLOGIST: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY,  # 相对温和
            ],
            RoleType.RADIOLOGIST: [
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.SURGERY,
                TreatmentOption.CHEMOTHERAPY,
            ],
            RoleType.PATIENT_ADVOCATE: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY,
            ],
            RoleType.NUTRITIONIST: [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.IMMUNOTHERAPY,
                TreatmentOption.CHEMOTHERAPY,
            ],
            RoleType.REHABILITATION_THERAPIST: [
                TreatmentOption.SURGERY,
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.PALLIATIVE_CARE,
            ],
        }
        return role_preferences.get(role, list(TreatmentOption))

    def _apply_intelligent_selection_strategy(
        self,
        viable_treatments: Dict[TreatmentOption, float],
        role_preferred_treatments: List[TreatmentOption],
        patient_state: PatientState,
        role: RoleType,
    ) -> TreatmentOption:
        """应用智能选择策略"""

        # 策略1: 如果有明显的高分治疗方案（>0.7），直接选择
        high_score_treatments = {t: s for t, s in viable_treatments.items() if s > 0.7}
        if high_score_treatments:
            return max(high_score_treatments.items(), key=lambda x: x[1])[0]

        # 策略2: 在角色偏好的治疗方案中选择评分最高的
        role_viable_treatments = {
            t: s for t, s in viable_treatments.items() if t in role_preferred_treatments
        }
        if role_viable_treatments:
            return max(role_viable_treatments.items(), key=lambda x: x[1])[0]

        # 策略3: 考虑患者状态的特殊情况
        if patient_state.quality_of_life_score < 0.3:
            # 生活质量很差，优先考虑姑息治疗
            if TreatmentOption.PALLIATIVE_CARE in viable_treatments:
                return TreatmentOption.PALLIATIVE_CARE

        if patient_state.age > 80:
            # 高龄患者，优先考虑温和治疗
            gentle_treatments = [
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
                TreatmentOption.IMMUNOTHERAPY,
            ]
            for treatment in gentle_treatments:
                if treatment in viable_treatments:
                    return treatment

        # 策略4: 如果评分接近，选择争议性较小的方案
        max_score = max(viable_treatments.values())
        close_treatments = {
            t: s for t, s in viable_treatments.items() if abs(s - max_score) < 0.2
        }

        if len(close_treatments) > 1:
            # 选择通常争议较小的治疗方案
            preference_order = [
                TreatmentOption.IMMUNOTHERAPY,
                TreatmentOption.RADIOTHERAPY,
                TreatmentOption.CHEMOTHERAPY,
                TreatmentOption.SURGERY,
                TreatmentOption.PALLIATIVE_CARE,
                TreatmentOption.WATCHFUL_WAITING,
            ]

            for preferred in preference_order:
                if preferred in close_treatments:
                    return preferred

        # 策略5: 默认选择评分最高的
        return max(viable_treatments.items(), key=lambda x: x[1])[0]

    def _generate_llm_initial_message(
        self,
        agent: RoleAgent,
        opinion: RoleOpinion,
        patient_state: PatientState,
        focus_treatment: TreatmentOption,
        treatment_options: List[TreatmentOption],
    ) -> str:
        """使用LLM生成个性化初始消息"""
        if not hasattr(agent, "llm_interface") or not agent.llm_interface:
            return ""

        try:
            # 使用专门的治疗推理方法生成基础推理
            if hasattr(agent.llm_interface, "generate_focus_treatment_reasoning"):
                reasoning = agent.llm_interface.generate_focus_treatment_reasoning(
                    patient_state=patient_state,
                    role=agent.role,
                    opinion=opinion,
                    treatment_option=focus_treatment,
                    knowledge_context={},
                    treatment_options=treatment_options,
                )
                logger.info(
                    f"生成初始化发言, 治疗选项: {focus_treatment}, 推理: {reasoning}"
                )
                if reasoning and len(reasoning.strip()) > 0:
                    # 基于推理生成MDT发言格式的消息
                    content = self._format_reasoning_as_mdt_message(
                        agent, reasoning, focus_treatment, opinion
                    )
                    logger.debug(
                        f"Generated LLM initial message for {agent.role.value}: {content}"
                    )
                    return content

        except Exception as e:
            logger.warning(
                f"LLM initial message generation failed for {agent.role}: {e}"
            )

        return ""

    def _format_reasoning_as_mdt_message(
        self,
        agent: RoleAgent,
        reasoning: str,
        focus_treatment: TreatmentOption,
        opinion,
    ) -> str:
        """将LLM推理格式化为MDT发言消息"""

        # 获取推荐强度
        treatment_score = opinion.treatment_preferences.get(focus_treatment, 0.0)
        logger.debug(f"treatment_score: {treatment_score}")
        recommendation_phrase = self._get_recommendation_phrase(treatment_score)
        logger.debug(f"recommendation_phrase: {recommendation_phrase}")
        # 构建MDT发言格式
        content = f"作为{agent.role.value}，我{recommendation_phrase}{focus_treatment.value}。\n\n"
        content += f"我的专业分析：{reasoning}"

        # 添加关注事项
        if opinion.concerns:
            content += f"\n\n需要特别关注的问题：{', '.join(opinion.concerns[:3])}"

        return content

    def _generate_template_initial_message(
        self, agent: RoleAgent, opinion, focus_treatment: TreatmentOption
    ) -> str:
        """生成模板化初始消息（降级方案）"""
        content = f"As the {agent.role.value}, I {self._get_recommendation_phrase(opinion.treatment_preferences[focus_treatment.value])} {focus_treatment.value}. "
        content += f"My reasoning: {opinion.reasoning}"

        if opinion.concerns:
            content += (
                f" However, I have concerns about: {', '.join(opinion.concerns[:2])}"
            )

        return content

    def _get_recommendation_phrase(self, score: float) -> str:
        """获取推荐措辞"""
        if score > 0.7:
            return "strongly recommend "
        elif score > 0.3:
            return "recommend "
        elif score > -0.3:
            return "have mixed feelings about "
        elif score > -0.7:
            return "have concerns about "
        else:
            return "strongly advise against "

    def _conduct_dialogue_round(
        self, patient_state: PatientState, opinions_dict: Dict[RoleType, RoleOpinion]
    ) -> DialogueRound:
        """进行一轮对话 - 增强版本，支持更自然的对话流程"""
        current_round = DialogueRound(
            round_number=self.current_round,
            messages=[],
            focus_treatment=self._select_focus_treatment(),
            consensus_status="discussing",
        )

        logger.info(
            f"Round {self.current_round} focusing on: {current_round.focus_treatment.value}"
        )

        # 获取该轮的相关知识
        round_knowledge = self.rag_system.retrieve_relevant_knowledge(
            patient_state, "treatment_discussion", current_round.focus_treatment
        )

        # 智能确定发言顺序和对话策略
        # 立场我觉得没啥用，之后可以去掉
        speaking_order, dialogue_strategy = self._determine_intelligent_speaking_order(
            current_round.focus_treatment, patient_state
        )

        logger.info(
            f"Dialogue strategy: {dialogue_strategy}, Speaking order: {[r.value for r in speaking_order]}"
        )

        current_round = self._conduct_sequential_presentation(
            current_round, patient_state, round_knowledge, speaking_order, opinions_dict
        )
        logger.info(
            f"Generated messages for round {self.current_round}: {current_round}"
        )
        return current_round

    def _determine_intelligent_speaking_order(
        self, focus_treatment: TreatmentOption, patient_state: PatientState
    ) -> tuple[List[RoleType], str]:
        """发言顺序和对话策略"""

        # 分析患者复杂度
        patient_complexity = self._assess_patient_complexity(patient_state)
        logger.info(f"Patient complexity: {patient_complexity:.2f}")

        strategy = "sequential_presentation"
        # 按传统的专业重要性排序
        speaking_order = self._get_traditional_speaking_order(focus_treatment)

        return speaking_order, strategy

    def _calculate_stance_variance(self, treatment: TreatmentOption) -> float:
        """计算立场方差，衡量争议程度"""
        stances = []
        for agent in self.agents.values():
            stance = agent.current_stance.get(treatment, 0)
            stances.append(stance)

        if len(stances) < 2:
            return 0.0

        import numpy as np

        return float(np.var(stances))

    def _assess_patient_complexity(self, patient_state: PatientState) -> float:
        """评估患者复杂度"""
        complexity_score = 0.0

        # 年龄因素
        if patient_state.age > 75:
            complexity_score += 0.3
        elif patient_state.age < 40:
            complexity_score += 0.2

        # 合并症
        complexity_score += min(0.4, len(patient_state.comorbidities) * 0.1)

        # 生活质量
        if patient_state.quality_of_life_score < 0.5:
            complexity_score += 0.2

        # 心理状态
        if patient_state.psychological_status in ["severe_anxiety", "depression"]:
            complexity_score += 0.1

        return min(1.0, complexity_score)

    def _get_polarized_speaking_order(
        self, treatment: TreatmentOption
    ) -> List[RoleType]:
        """获取极化发言顺序（用于争议性讨论）"""
        stance_roles = []
        for role, agent in self.agents.items():
            stance = agent.current_stance.get(treatment, 0)
            stance_roles.append((role, abs(stance)))

        # 按立场强度排序，最极端的先发言
        stance_roles.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            f"Polarized speaking order: {[role.value for role, _ in stance_roles]}"
        )
        return [role for role, _ in stance_roles]

    def _get_expertise_based_order(
        self, treatment: TreatmentOption, patient_state: PatientState
    ) -> List[RoleType]:
        """基于专业相关性的发言顺序"""
        # 根据治疗类型和患者情况确定专业相关性
        expertise_weights = {
            TreatmentOption.SURGERY: {
                RoleType.ONCOLOGIST: 0.9,
                RoleType.RADIOLOGIST: 0.8,
                RoleType.NURSE: 0.7,
                RoleType.NUTRITIONIST: 0.6,
                RoleType.REHABILITATION_THERAPIST: 0.7,
                RoleType.PSYCHOLOGIST: 0.4,
                RoleType.PATIENT_ADVOCATE: 0.5,
            },
            TreatmentOption.CHEMOTHERAPY: {
                RoleType.ONCOLOGIST: 0.9,
                RoleType.NURSE: 0.8,
                RoleType.NUTRITIONIST: 0.7,
                RoleType.PSYCHOLOGIST: 0.6,
                RoleType.RADIOLOGIST: 0.4,
                RoleType.PATIENT_ADVOCATE: 0.6,
                RoleType.REHABILITATION_THERAPIST: 0.4,
            },
            TreatmentOption.RADIOTHERAPY: {
                RoleType.RADIOLOGIST: 0.9,
                RoleType.ONCOLOGIST: 0.8,
                RoleType.NURSE: 0.7,
                RoleType.NUTRITIONIST: 0.6,
                RoleType.PSYCHOLOGIST: 0.5,
                RoleType.PATIENT_ADVOCATE: 0.5,
                RoleType.REHABILITATION_THERAPIST: 0.4,
            },
            TreatmentOption.IMMUNOTHERAPY: {
                RoleType.ONCOLOGIST: 0.9,
                RoleType.NURSE: 0.8,
                RoleType.NUTRITIONIST: 0.5,
                RoleType.PSYCHOLOGIST: 0.5,
                RoleType.PATIENT_ADVOCATE: 0.6,
                RoleType.RADIOLOGIST: 0.4,
                RoleType.REHABILITATION_THERAPIST: 0.3,
            },
            TreatmentOption.PALLIATIVE_CARE: {
                RoleType.NURSE: 0.9,
                RoleType.PATIENT_ADVOCATE: 0.9,
                RoleType.PSYCHOLOGIST: 0.8,
                RoleType.NUTRITIONIST: 0.7,
                RoleType.REHABILITATION_THERAPIST: 0.6,
                RoleType.ONCOLOGIST: 0.6,
                RoleType.RADIOLOGIST: 0.3,
            },
            TreatmentOption.WATCHFUL_WAITING: {
                RoleType.PATIENT_ADVOCATE: 0.9,
                RoleType.PSYCHOLOGIST: 0.7,
                RoleType.NURSE: 0.7,
                RoleType.NUTRITIONIST: 0.6,
                RoleType.ONCOLOGIST: 0.5,
                RoleType.RADIOLOGIST: 0.3,
                RoleType.REHABILITATION_THERAPIST: 0.4,
            },
        }

        weights = expertise_weights.get(treatment, {})

        # 根据患者复杂度与具体症状动态调整
        if len(patient_state.comorbidities) > 2:
            weights[RoleType.NURSE] = weights.get(RoleType.NURSE, 0.5) + 0.2

        # 心理状态（支持中英文关键词匹配）
        symptoms_lower = {s.lower() for s in (patient_state.symptoms or [])}
        psych_status_text = patient_state.psychological_status or ""
        psych_status_lower = psych_status_text.lower()

        psych_flags_en = {"anxious", "depressed", "severe_anxiety", "depression"}
        psych_flags_zh = {"焦虑", "抑郁", "重度焦虑", "抑郁症"}
        if any(flag in psych_status_lower for flag in psych_flags_en) or any(
            flag in psych_status_text for flag in psych_flags_zh
        ):
            weights[RoleType.PSYCHOLOGIST] = (
                weights.get(RoleType.PSYCHOLOGIST, 0.5) + 0.2
            )

        # 营养相关风险（体重下降/恶病质/营养不良，支持中英文）
        nutrition_flags_en = {"weight_loss", "cachexia", "malnutrition"}
        nutrition_flags_zh = {"体重下降", "恶病质", "营养不良"}
        if symptoms_lower.intersection(nutrition_flags_en) or any(
            flag in (patient_state.symptoms or []) for flag in nutrition_flags_zh
        ):
            weights[RoleType.NUTRITIONIST] = (
                weights.get(RoleType.NUTRITIONIST, 0.5) + 0.2
            )

        # 功能受限/术后康复需求（支持中英文）
        rehab_flags_en = {"mobility_issue", "weakness", "postoperative"}
        rehab_flags_zh = {"运动障碍", "虚弱", "术后"}
        if symptoms_lower.intersection(rehab_flags_en) or any(
            flag in (patient_state.symptoms or []) for flag in rehab_flags_zh
        ):
            weights[RoleType.REHABILITATION_THERAPIST] = (
                weights.get(RoleType.REHABILITATION_THERAPIST, 0.4) + 0.2
            )

        # 按权重排序；若为空，回退到传统顺序
        sorted_roles = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ordered = [role for role, _ in sorted_roles if role in self.agents]
        if not ordered:
            return self._get_traditional_speaking_order(treatment)
        return ordered

    def _get_traditional_speaking_order(
        self, treatment: TreatmentOption
    ) -> List[RoleType]:
        """传统的发言顺序"""
        traditional_order = [
            RoleType.ONCOLOGIST,
            RoleType.RADIOLOGIST,
            RoleType.NURSE,
            RoleType.PSYCHOLOGIST,
            RoleType.PATIENT_ADVOCATE,
            RoleType.NUTRITIONIST,
            RoleType.REHABILITATION_THERAPIST,
        ]
        return [role for role in traditional_order if role in self.agents]

    def _conduct_focused_debate(
        self,
        round_data: DialogueRound,
        patient_state: PatientState,
        knowledge: Dict,
        speaking_order: List[RoleType],
    ) -> DialogueRound:
        """进行聚焦辩论式对话"""
        logger.info("Conducting focused debate format")

        # 第一轮：各方表明立场
        for role in speaking_order:
            agent = self.agents[role]
            all_previous_messages = self._get_all_previous_messages(round_data)

            response = agent.generate_dialogue_response(
                patient_state,
                knowledge,
                all_previous_messages,
                round_data.focus_treatment,
            )
            round_data.messages.append(response)

        # 第二轮：针对性回应（如果有明显分歧）
        if len(speaking_order) > 2:
            opposing_pairs = self._identify_opposing_pairs(
                speaking_order, round_data.focus_treatment
            )
            for role1, role2 in opposing_pairs[:2]:  # 最多2对辩论
                # role1 回应 role2
                agent1 = self.agents[role1]
                all_previous_messages = self._get_all_previous_messages(round_data)

                response = agent1.generate_dialogue_response(
                    patient_state,
                    knowledge,
                    all_previous_messages,
                    round_data.focus_treatment,
                )
                round_data.messages.append(response)

        return round_data

    def _conduct_collaborative_discussion(
        self,
        round_data: DialogueRound,
        patient_state: PatientState,
        knowledge: Dict,
        speaking_order: List[RoleType],
    ) -> DialogueRound:
        """进行协作式讨论"""
        logger.info("Conducting collaborative discussion format")

        # 协作式讨论：每个角色都有机会回应前面的观点
        for i, role in enumerate(speaking_order):
            agent = self.agents[role]
            all_previous_messages = self._get_all_previous_messages(round_data)

            response = agent.generate_dialogue_response(
                patient_state,
                knowledge,
                all_previous_messages,
                round_data.focus_treatment,
            )
            round_data.messages.append(response)

            # 如果不是最后一个发言者，给其他人机会简短回应
            if i < len(speaking_order) - 1 and len(round_data.messages) > 2:
                # 随机选择一个之前发言的角色进行简短回应
                import random

                previous_speakers = speaking_order[:i]
                if previous_speakers and random.random() < 0.3:  # 30%概率有人回应
                    responder_role = random.choice(previous_speakers)
                    responder_agent = self.agents[responder_role]

                    brief_response = responder_agent.generate_dialogue_response(
                        patient_state,
                        knowledge,
                        all_previous_messages,
                        round_data.focus_treatment,
                    )
                    round_data.messages.append(brief_response)

        return round_data
    
    def _conduct_sequential_presentation_medqa(
        self,
        round_data: DialogueRound,
        patient_state: PatientState,
        opinions_dict: Dict[RoleType, RoleOpinion],
    ) -> DialogueRound:
        """进行顺序陈述式对话"""

        # 传统的顺序发言
        for role in speaking_order:
            agent = self.agents[role]
            last_round_messages = self._get_last_round_previous_messages()
            logger.info(f"Last round messages: {last_round_messages}")
            all_previous_messages = self._get_all_previous_messages(round_data)
            logger.info(f"All previous messages: {all_previous_messages}")
            response = agent.generate_dialogue_response(
                patient_state,
                knowledge,
                all_previous_messages,
                round_data.focus_treatment,
                opinions_dict,
                last_round_messages,
            )
            logger.info(f"{role.value} responded: {response.content}...")
            round_data.messages.append(response)

            logger.debug(f"{role.value} responded: {response.content}...")

        return round_data

    def _conduct_sequential_presentation(
        self,
        round_data: DialogueRound,
        patient_state: PatientState,
        knowledge: Dict,
        speaking_order: List[RoleType],
        opinions_dict: Dict[RoleType, RoleOpinion],
    ) -> DialogueRound:
        """进行顺序陈述式对话"""
        logger.info("Conducting sequential presentation format")

        # 传统的顺序发言
        for role in speaking_order:
            agent = self.agents[role]
            last_round_messages = self._get_last_round_previous_messages()
            logger.info(f"Last round messages: {last_round_messages}")
            all_previous_messages = self._get_all_previous_messages(round_data)
            logger.info(f"All previous messages: {all_previous_messages}")
            response = agent.generate_dialogue_response(
                patient_state,
                knowledge,
                all_previous_messages,
                round_data.focus_treatment,
                opinions_dict,
                last_round_messages,
            )
            logger.info(f"{role.value} responded: {response.content}...")
            round_data.messages.append(response)

            logger.debug(f"{role.value} responded: {response.content}...")

        return round_data

    def _get_all_previous_messages(self, current_round: DialogueRound) -> List:
        """获取所有之前的消息"""
        all_previous_messages = []
        for round in self.dialogue_rounds:
            all_previous_messages.extend(round.messages)
        all_previous_messages.extend(current_round.messages)
        return all_previous_messages

    def _get_last_round_previous_messages(self) -> List:
        """获取上一轮的所有消息"""
        if self.dialogue_rounds:
            last_round = self.dialogue_rounds[-1]
            return last_round.messages
        return []

    def _identify_opposing_pairs(
        self, speaking_order: List[RoleType], treatment: TreatmentOption
    ) -> List[tuple]:
        """识别对立的角色对"""
        opposing_pairs = []

        for i, role1 in enumerate(speaking_order):
            stance1 = self.agents[role1].current_stance.get(treatment, 0)

            for role2 in speaking_order[i + 1 :]:
                stance2 = self.agents[role2].current_stance.get(treatment, 0)

                # 如果立场相反（一个支持一个反对）
                if (stance1 > 0.3 and stance2 < -0.3) or (
                    stance1 < -0.3 and stance2 > 0.3
                ):
                    opposing_pairs.append((role1, role2))

        return opposing_pairs

    def _select_focus_treatment(self) -> TreatmentOption:
        """选择焦点治疗方案"""
        if not self.dialogue_rounds:
            return TreatmentOption.SURGERY  # 默认开始话题

        # 分析前一轮的争议点
        last_round = self.dialogue_rounds[-1]
        treatment_mentions = {}

        for message in last_round.messages:
            treatment = message.treatment_focus
            if treatment not in treatment_mentions:
                treatment_mentions[treatment] = 0
            treatment_mentions[treatment] += 1

        # 选择提及最多或争议最大的治疗方案
        if treatment_mentions:
            logger.info(f"选择焦点治疗方案：treatment_mentions: {treatment_mentions}")
            return max(treatment_mentions.items(), key=lambda x: x[1])[0]

        # 轮换讨论不同治疗方案
        treatments = list(TreatmentOption)
        return treatments[self.current_round % len(treatments)]

    def _determine_speaking_order(
        self, focus_treatment: TreatmentOption
    ) -> List[RoleType]:
        """确定发言顺序"""
        # 基于角色对治疗方案的相关性确定顺序
        relevance_scores = {}

        for role, agent in self.agents.items():
            current_stance = agent.current_stance.get(focus_treatment, 0)
            relevance_scores[role] = abs(current_stance)  # 立场越强烈越早发言

        # 按相关性排序
        sorted_roles = sorted(
            relevance_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [role for role, _ in sorted_roles]

    def _update_agent_stances(self, round_data: DialogueRound) -> None:
        """更新智能体立场"""
        for role, agent in self.agents.items():
            agent.update_stance_based_on_dialogue(round_data.messages)

    def _check_convergence(self) -> bool:
        """检查是否收敛（聚焦上一轮的焦点治疗）"""
        if len(self.dialogue_rounds) < 2:
            return False

        last_round = self.dialogue_rounds[-1]
        focus_treatment = last_round.focus_treatment
        stance_threshold = 0.6

        if focus_treatment:
            stable_roles = []
            role_stances = {}
            for role, agent in self.agents.items():
                score = agent.current_stance.get(focus_treatment, 0.0)
                role_stances[role.value] = score
                if abs(score) >= stance_threshold:
                    stable_roles.append(role.value)

            convergence_ratio = len(stable_roles) / len(self.agents)
            logger.debug(
                f"Convergence(check on {focus_treatment.value}): {convergence_ratio:.2f} "
                f"(stable_roles: {stable_roles}, role_stances: {role_stances}, "
                f"threshold: {self.convergence_threshold})"
            )
            return convergence_ratio >= self.convergence_threshold

        # 无焦点时回退到原逻辑
        stable_agents = 0
        for role, agent in self.agents.items():
            current_stances = agent.current_stance
            strong_stances = [
                abs(score) > stance_threshold for score in current_stances.values()
            ]
            if (
                len(strong_stances) > 0
                and sum(strong_stances) / len(strong_stances) > 0.7
            ):
                stable_agents += 1

        convergence_ratio = stable_agents / len(self.agents)
        logger.debug(
            f"Convergence check: {convergence_ratio:.2f} (threshold: {self.convergence_threshold})"
        )
        return convergence_ratio >= self.convergence_threshold

    def _focus_on_contentious_treatments(self, patient_state: PatientState) -> None:
        """聚焦争议治疗方案"""
        # 识别争议最大的治疗方案
        treatment_variances = {}

        for treatment in TreatmentOption:
            scores = []
            for agent in self.agents.values():
                score = agent.current_stance.get(treatment, 0)
                scores.append(score)

            if len(scores) > 1:
                treatment_variances[treatment] = np.var(scores)

        # 如果方差过大，在下一轮重点讨论
        if treatment_variances:
            most_contentious = max(treatment_variances.items(), key=lambda x: x[1])
            if most_contentious[1] > 0.5:  # 高争议阈值
                logger.info(
                    f"High contention detected for: {most_contentious[0].value} (variance: {most_contentious[1]:.3f})"
                )

    def _generate_final_consensus(self, patient_state: PatientState) -> ConsensusResult:
        """生成最终共识结果"""
        from .consensus_matrix import ConsensusMatrix

        # 收集最终立场
        final_opinions = {}
        for role, agent in self.agents.items():
            final_opinion = agent.generate_initial_opinion(patient_state, {})
            # 使用对话后的立场覆盖初始立场
            final_opinion.treatment_preferences = agent.current_stance.copy()
            final_opinion.reasoning = (
                f"Final position after {self.current_round} rounds of discussion"
            )
            final_opinions[role] = final_opinion

        # 使用共识矩阵系统处理最终结果
        consensus_system = ConsensusMatrix()
        consensus_system.agents = self.agents  # 使用更新后的智能体

        # 生成详细的对话摘要
        dialogue_summary = self._generate_dialogue_summary()

        # 构建最终共识结果
        import pandas as pd

        # 构建共识矩阵
        treatments = list(TreatmentOption)
        roles = list(RoleType)

        matrix_data = np.zeros((len(treatments), len(roles)))

        for i, treatment in enumerate(treatments):
            for j, role in enumerate(roles):
                if role in final_opinions:
                    score = final_opinions[role].treatment_preferences.get(
                        treatment, 0.0
                    )
                    matrix_data[i, j] = score

        consensus_matrix = pd.DataFrame(
            matrix_data,
            index=[t.value for t in treatments],
            columns=[r.value for r in roles],
        )

        # 计算聚合评分
        aggregated_scores = {}
        for treatment in TreatmentOption:
            scores = []
            weights = []

            for role, opinion in final_opinions.items():
                score = opinion.treatment_preferences.get(treatment, 0.0)
                weight = opinion.confidence
                scores.append(score)
                weights.append(weight)

            if scores:
                weighted_score = np.average(scores, weights=weights)
                aggregated_scores[treatment] = weighted_score

        # 识别冲突与一致性
        conflicts = self._identify_final_conflicts(final_opinions)
        agreements = self._identify_final_agreements(final_opinions)

        return ConsensusResult(
            consensus_matrix=consensus_matrix,
            role_opinions=final_opinions,
            aggregated_scores=aggregated_scores,
            conflicts=conflicts,
            agreements=agreements,
            dialogue_summary=dialogue_summary,
            timestamp=datetime.now(),
            convergence_achieved=self._check_convergence(),
            total_rounds=self.current_round,
        )

    def _generate_dialogue_summary(self) -> Dict[str, Any]:
        """生成对话摘要"""
        summary = {
            "total_messages": sum(
                len(round.messages) for round in self.dialogue_rounds
            ),
            "key_topics": [],
            "major_agreements": [],
            "persistent_disagreements": [],
        }

        # 统计话题频率
        topic_counts = {}
        all_evidence = set()

        for round in self.dialogue_rounds:
            for message in round.messages:
                treatment = message.treatment_focus
                if treatment:
                    topic_counts[treatment] = topic_counts.get(treatment, 0) + 1

        # 提取关键信息
        summary["key_topics"] = sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        return summary

    def _identify_final_conflicts(
        self, final_opinions: Dict[RoleType, Any]
    ) -> List[Dict[str, Any]]:
        """识别最终冲突"""
        conflicts = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in final_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                if variance > 0.5:  # 高方差表示冲突
                    conflicts.append(
                        {
                            "treatment": treatment,
                            "variance": variance,
                            "min_score": min(scores),
                            "max_score": max(scores),
                            "conflicting_roles": self._find_conflicting_roles(
                                treatment, final_opinions
                            ),
                        }
                    )

        return conflicts

    def _identify_final_agreements(
        self, final_opinions: Dict[RoleType, Any]
    ) -> List[Dict[str, Any]]:
        """识别最终一致意见"""
        agreements = []

        for treatment in TreatmentOption:
            scores = [
                opinion.treatment_preferences.get(treatment, 0.0)
                for opinion in final_opinions.values()
            ]

            if len(scores) > 1:
                variance = np.var(scores)
                mean_score = np.mean(scores)

                if variance < 0.2 and abs(mean_score) > 0.3:  # 低方差且有明确倾向
                    agreements.append(
                        {
                            "treatment": treatment,
                            "consensus_score": mean_score,
                            "agreement_strength": 1.0 - variance,
                            "unanimous": all(s > 0.5 for s in scores)
                            or all(s < -0.5 for s in scores),
                        }
                    )

        return agreements

    def _find_conflicting_roles(
        self, treatment: TreatmentOption, final_opinions: Dict[RoleType, Any]
    ) -> List[str]:
        """找出在特定治疗上有冲突的角色"""
        scores_by_role = {
            role.value: opinion.treatment_preferences.get(treatment, 0.0)
            for role, opinion in final_opinions.items()
        }

        mean_score = np.mean(list(scores_by_role.values()))
        conflicting_roles = [
            role
            for role, score in scores_by_role.items()
            if abs(score - mean_score) > 0.5
        ]

        return conflicting_roles

    def get_dialogue_transcript(self) -> str:
        """获取完整对话记录"""
        transcript = f"=== MDT Discussion Transcript ===\n"
        transcript += f"Total Rounds: {len(self.dialogue_rounds)}\n\n"

        for round in self.dialogue_rounds:
            transcript += f"--- Round {round.round_number} ---\n"
            if round.focus_treatment:
                transcript += f"Focus: {round.focus_treatment.value}\n"

            for message in round.messages:
                transcript += f"\n{message.role.value}: {message.content}\n"

            transcript += "\n"

        return transcript


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dialogue_manager = MultiAgentDialogueManager(
        max_rounds=10, convergence_threshold=0.5
    )
    opintions_dict = {
        "oncologist": RoleOpinion(
            role="oncologist",
            treatment_preferences={
                "surgery": 0.9,
                "chemotherapy": 0.7,
                "radiotherapy": 0.8,
                "immunotherapy": 0.3,
                "palliative_care": -0.5,
                "watchful_waiting": -0.8,
            },
            reasoning="手术为根治基石，多学科支持可降低风险，综合治疗提升生存率",
            confidence=0.95,
            concerns=["围术期心脏事件", "化疗耐受性", "血糖波动"],
        ),
        "radiologist": RoleOpinion(
            role="radiologist",
            treatment_preferences={
                "surgery": 0.9,
                "chemotherapy": 0.7,
                "radiotherapy": 0.8,
                "immunotherapy": 0.3,
                "palliative_care": -0.5,
                "watchful_waiting": -0.8,
            },
            reasoning="影像学支持根治性手术，多学科协同可降低围术期风险，术后放疗强化局部控制。",
            confidence=0.95,
            concerns=["围术期心血管事件", "放疗耐受性", "血糖波动影响愈合"],
        ),
        "nurse": RoleOpinion(
            role="nurse",
            treatment_preferences={
                "surgery": 0.9,
                "chemotherapy": 0.6,
                "radiotherapy": 0.7,
                "immunotherapy": 0.4,
                "palliative_care": -0.3,
                "watchful_waiting": -0.6,
            },
            reasoning="患者状态稳定，多学科支持下手术风险可控，术后恢复预期良好。",
            confidence=0.95,
            concerns=["围术期心功能波动", "血糖波动风险", "术后感染可能"],
        ),
        "psychologist": RoleOpinion(
            role="psychologist",
            treatment_preferences={
                "surgery": 0.85,
                "chemotherapy": 0.65,
                "radiotherapy": 0.55,
                "immunotherapy": 0.3,
                "palliative_care": -0.2,
                "watchful_waiting": -0.4,
            },
            reasoning="患者心理状态良好，手术可增强掌控感，多学科支持降低心理负担",
            confidence=0.9,
            concerns=["术后心理适应", "治疗依从性波动", "康复信心波动"],
        ),
        "patient_advocate": RoleOpinion(
            role="patient_advocate",
            treatment_preferences={
                "surgery": 0.95,
                "chemotherapy": 0.65,
                "radiotherapy": 0.75,
                "immunotherapy": 0.3,
                "palliative_care": -0.1,
                "watchful_waiting": -0.6,
            },
            reasoning="多学科支持手术，患者状态稳定，围术期管理可控，根治性治疗优先",
            confidence=0.9,
            concerns=["围术期并发症", "术后恢复挑战", "合并症叠加风险"],
        ),
        "nutritionist": RoleOpinion(
            role="nutritionist",
            treatment_preferences={
                "surgery": 0.8,
                "chemotherapy": 0.6,
                "radiotherapy": 0.5,
                "immunotherapy": 0.4,
                "palliative_care": -0.3,
                "watchful_waiting": -0.6,
            },
            reasoning="术前营养优化可提升手术耐受，术后肠内营养支持促进恢复，整体获益显著",
            confidence=0.9,
            concerns=["术后吸收障碍", "化疗食欲下降", "合并症恶化"],
        ),
        "rehabilitation_therapist": RoleOpinion(
            role="rehabilitation_therapist",
            treatment_preferences={
                "surgery": 0.9,
                "chemotherapy": 0.6,
                "radiotherapy": 0.5,
                "immunotherapy": 0.3,
                "palliative_care": -0.2,
                "watchful_waiting": -0.6,
            },
            reasoning="手术获益明确，多学科支持下风险可控，术前康复可提升耐受力",
            confidence=0.9,
            concerns=["术后功能障碍", "治疗依从性", "康复延迟"],
        ),
    }

    check_convergence = dialogue_manager._check_discussion_convergence(
        opintions_dict, TreatmentOption
    )
    print(check_convergence)
