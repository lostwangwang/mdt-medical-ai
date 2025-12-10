"""
大语言模型接口模块
文件路径: src/utils/llm_interface.py
作者: 姚刚
功能: 提供统一的LLM调用接口，支持多种模型和任务
输入:
- patient_state: 患者状态对象，包含患者基本信息、历史记录等
- role: 角色类型，用于指定不同的任务（如MDT、患者、医生等）
- treatment_option: 治疗选项对象，包含治疗方案的详细信息
- knowledge_context: 知识库上下文，用于提供额外的背景知识
输出:
- 治疗推理文本，包含对患者状态的分析和对治疗选项的建议

"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import openai
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

from ..core.data_models import (
    PatientState,
    TreatmentOption,
    RoleType,
    RoleOpinion,
    DialogueRound, QuestionOpinion, RoleRegistry,
)
import experiments.medqa_types as medqa_types

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置"""

    model_name: str = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 30

    def __post_init__(self):
        """后初始化，设置默认值"""
        if self.model_name is None:
            self.model_name = os.getenv("MODEL_NAME")
        if self.api_key is None:
            self.api_key = os.getenv("QWEN_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("BASE_URL")


class LLMInterface:
    """大语言模型接口"""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._setup_client()

    def _setup_client(self):
        """设置LLM客户端，支持阿里云百炼和OpenAI"""
        try:
            # 初始化变量，避免作用域问题
            api_key = self.config.api_key
            base_url = self.config.base_url

            # 检查API密钥是否有效
            if (
                    not api_key
                    or api_key.strip() == ""
                    or api_key == "test-api-key-for-testing"
            ):
                logger.info(
                    "No valid API key provided, LLM client will not be initialized. Using template fallback."
                )
                self.client = None
                return

            # 根据base_url判断是否为阿里云百炼
            if base_url and "dashscope" in base_url:
                logger.info("Detected Alibaba Cloud DashScope (百炼) configuration")
            elif base_url:
                logger.info(f"Using custom base URL: {base_url}")
            else:
                logger.info("Using OpenAI default endpoint")

            # 初始化客户端（兼容OpenAI格式的接口）
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            logger.info(
                f"LLM client initialized successfully with model: {self.config.model_name}"
            )

        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            logger.info("Falling back to template-based responses")
            self.client = None

    def generate_update_agent_opinions_reasoning_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            current_round: DialogueRound,
            previous_opinion: Union[RoleOpinion, QuestionOpinion],
            question_options: List[medqa_types.QuestionOption],
            mdt_leader_summary: str = None,
            dataset_name: str = None,
    ):
        """生成更新角色有关医学问题的推理"""

        prompt = self._build_update_agent_opinions_reasoning_prompt_medqa(
            question_state, role, current_round, previous_opinion, question_options, mdt_leader_summary, dataset_name
        )
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是一个医学多学科团队（MDT, Multidisciplinary Team）智能体成员，"
                                       f"当前身份是 **{role.value}**。描述: {role.description}, 权重: {role.weight} "
                                       f"你的任务是在本轮对医疗问题进行‘立场再评估’（soft update）。"
                                       f"你将根据以下信息更新自己的观点：上一轮观点、领导者总结以及本轮新增证据。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    question_state, role, question_options
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

    def _build_mdt_leader_content_prompt(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            dialogue_round: DialogueRound
    ):
        dialogus_messages = dialogue_round.messages
        agents_messages = "\n\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in dialogus_messages
        ])
        prompt = f"""
        你是多学科医疗团队（MDT）的负责人（Leader）。
        你的任务是根据各智能体的发言内容，对当前讨论进行总结，并提供下一轮讨论的指导方向。

        输入信息：
        - 题目：{question_state.question}
        - 选项：{[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
        - 智能体发言内容：{agents_messages}

        请完成以下任务：
        1. 当前轮总结（200–250字）：
           - 提炼各选项的支持趋势，指出不同智能体在选项上的主要分歧和共识。
           - 强调关键证据或理由，帮助团队理解哪些观点得到广泛支持，哪些存在争议。
        2. 下一轮讨论指导：
           - 指出分歧明显或证据不足的选项，建议重点讨论。
           - 提示哪些论点或证据需要补充或澄清。
           - 可采用条目形式列出具体行动或讨论方向。

        要求：
        - 用中文输出，简明、清晰。
        """

        return prompt

    def _build_final_mdt_leader_summary_prompt(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            dialogue_round: DialogueRound,
            consensus_dict: Dict[str, Any],
            opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]] = None,
    ):
        df, group_icc, consensus = consensus_dict["df"], consensus_dict["group_icc"], consensus_dict["consensus"]
        means = df["mean"]
        mean_scores = [f"{option.name}: {means[option.value]}" for option in question_options]
        variances = [f"{option.name}: {df['std'][option.value]}" for option in question_options]
        agents_messages = "\n\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in dialogue_round.messages
        ])
        agents_opinions_str = "\n\n"

        # 格式化各角色的聚合意见
        for role, current_opinion in opinions_dict.items():
            agents_opinions_str += self.format_opinion_for_prompt(current_opinion, role)
            agents_opinions_str += "\n\n"

        prompt = f"""
        你是多学科医疗团队（MDT）的负责人（Leader）。你的任务是根据智能体的聚合意见，对题目进行最终总结，并给出结论。

        输入信息：
        - 题目（QUESTION）：{question_state.question}
        - 选项（OPTIONS）：{[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
        - 各选项的平均分（mean_scores）：{mean_scores}
        - 各选项的方差（variances）：{variances}
        - 智能体发言内容：{agents_messages}
        - 各智能体的聚合意见：{agents_opinions_str}

        任务要求：
        - 总结每个选项的支持程度（参考平均分）和意见分布（参考方差）。
        - 综合各智能体意见、证据和分析，给出最可能的正确选项或最终结论。
        - 提供决策依据，说明为何选择该选项并排除其他选项。
        - 输出 JSON，格式如下：
        {{
            "label": "{{最终选项标签}}",
            "content": "{{选项内容}}",
            "decision_reasoning": "{{决策推理，100~150字}}"
        }}
        """
        return prompt

    def _build_llm_recruit_agents_medqa_prompt(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
    ):
        prompt = f"""
        You are the MDT Leader (Multidisciplinary Team Leader) in a medical intelligent reasoning system.

        Your task is to recruit the most relevant **medical expert roles** (not real people) for this question, focusing on **knowledge-based reasoning** rather than clinical practice judgment.

        【Input Information】
        - Question: {question_state.question}
        - Options: {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}

        【Requirements】
        1. Recruit 3–5 **medical expert roles**, not real individuals.
           - Prefer knowledge-oriented roles such as:
             "Pharmacologist", "Biostatistician", "Clinical Trial Specialist", "Toxicologist", 
             "Epidemiologist", "Medical Informatics Specialist"
           - DO NOT generate personal names, fictional doctors, or real-world people.

        2. Each role must represent a distinct medical subfield to ensure heterogeneity.

        3. Assign a weight (0–1) to each expert role based on relevance to answering the question:
           - Core: 0.7–1.0
           - Secondary: 0.3–0.6
           - Weakly related: 0–0.3

        4. Output strictly in JSON, no extra explanation.

        【Output Format】
        {{
          "recruited_experts": [
            {{
              "name": "<RoleTypeEnglish>", 
              "value": "<RoleTypeChinese>",
              "description": "<A concise role description (10–25 words)>",
              "weight": <0~1>
            }}
          ]
        }}

        【Important Instructions】
        - Only output role types, not personal identities.
        - Do NOT use or fabricate any real or fictional human names.
        - Focus on **knowledge-driven reasoning** rather than clinical intuition.
        - Ensure JSON is strictly valid.
        """
        return prompt

    def llm_recurt_agents_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption]
    ):
        """
        MDT LEADER负责通过分析问题的难易程度，并且为每个问题招募对应的角色
        :param question_state: 问题信息
        :param question_options: 问题选项
        :return: 返回
        """
        prompt = self._build_llm_recruit_agents_medqa_prompt(question_state, question_options)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the MDT Leader (Multidisciplinary Team Leader) in a medical reasoning system. "
                        "Recruit medical expert roles using knowledge-based, data-driven reasoning."
                        "Only output a strict JSON; do not include explanations or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        return response.choices[0].message.content.strip()

    def llm_generate_final_mdt_leader_summary(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            dialogue_round: DialogueRound,
            consensus_dict: Dict[str, Any] = None,
            opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]] = None,
    ):
        """
        mdt_leader生成最终方案
        :param question_state:
        :param question_options:
        :param dialogue_round:
        :param consensus_dict:
        :param opinions_dict:
        :return:
        """
        prompt = self._build_final_mdt_leader_summary_prompt(
            question_state, question_options, dialogue_round, consensus_dict, opinions_dict
        )
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"你是多学科会诊（MDT, Multidisciplinary Team）的一名成员, 当前身份为一位专业的MDT_LEADER，"
                               f"你是多学科医疗团队（MDT）的负责人（Leader）。你的任务是根据智能体的聚合意见，对题目进行总结，并给出最终方案",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def llm_generate_mdt_leader_content(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            dialogue_round: DialogueRound,
    ):
        prompt = self._build_mdt_leader_content_prompt(question_state, question_options, dialogue_round)
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是多学科医疗团队（MDT）的负责人（Leader）。"
                        "你的任务是基于现有智能体发言总结讨论，提炼各选项支持趋势、分歧与共识，"
                        "强调关键证据，并提供下一轮讨论的指导。"
                        "总结要简明、清晰，不引入未参与的角色或外部专业视角。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def generate_update_agent_opinions_reasoning(
            self,
            patient_state: PatientState,
            role: RoleType,
            current_round: DialogueRound,
            previous_opinion: RoleOpinion,
            treatment_options: List[TreatmentOption],
    ) -> str:
        """生成更新角色意见的推理"""
        prompt = self._build_update_agent_opinions_reasoning_prompt(
            patient_state, role, current_round, previous_opinion, treatment_options
        )
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是多学科会诊（MDT, Multidisciplinary Team）的一名成员,当前身份为一位专业的{role.value}，请基于患者信息、角色专业性、对话上下文、上一轮对话和当前对话，更新角色意见、治疗偏好、置信度",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    patient_state, role, treatment_options
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

    def build_consensus_feedback(self, df: DataFrame, W: float, cur_round: int,
                                 focus_question: medqa_types.QuestionOption):
        """
        根据当前意见生成共识反馈提示
        """
        # 计算每个选项的平均偏好
        best_treatment = df["mean"].idxmax()
        print(f"best_treatement:{best_treatment}")
        option = medqa_types.QuestionOption(best_treatment)
        feedback = (
            f"当前为第{cur_round}轮讨论轮"
            f"当前团队一致性指标 Kendall's W = {W:.2f}。\n"
            f"目前团队整体倾向于选项: {option.name}: {option.value}(平均偏好最高)\n"
            f"请结合此趋势，重新评估你的立场。\n"
            f"如果你的意见与多数不一致，请说明原因；"
            f"如果没有强证据支撑差异，请考虑适度靠拢以促进团队共识。"
        )
        return feedback

    def _build_update_agent_opinions_reasoning_prompt_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            current_round: DialogueRound,
            previous_opinion: Union[RoleOpinion, QuestionOpinion],
            question_options: List[medqa_types.QuestionOption],
            mdt_leader_summary: str,
            dataset_name: str = None,
    ) -> str:
        cur_round = current_round.round_number
        previous_opinion_str = self.format_opinion_for_prompt(previous_opinion, role.value)
        new_evidence = [msg.content for msg in current_round.messages if msg.role == role]
        print(f"[DEBUG]current_round:{current_round}")
        if dataset_name in ["medqa", "pubmedqa", "symcat", "ddxplus", "medbullets"]:
            prompt = f"""
            你是一名医疗多学科团队（MDT）的成员。
            你的当前角色是：**{role.value}**。描述: {role.description}, 权重: {role.weight}
            
            你的任务是在本轮根据提供的信息，对医疗题目进行“立场再评估”（soft update）。
            
            你将会收到：
            1. 当前轮次 (current_round)
            2. 医学问题（question）
            3. 问题选项（options）
            4. 你上一轮的观点（previous_opinion）
            5. MDT 领导者的上一轮总结（leader_summary）
            6. 本轮新增证据或讨论内容（new_evidence）
            -----------------------------------
            【你的核心任务】
            请基于以上信息进行“软更新”，你必须遵守以下原则：
            
            - 参考 previous_opinion 作为“偏好趋势”，但 **不直接累加或微调分数**
            - leader_summary 的方向性 > previous_opinion
            - 本轮 new_evidence 的重要性最高
            - 你本轮输出的分数是一个“重新评估后的新分数”，但会受前一轮讨论影响（即 soft update，而非累加）
            
            -----------------------------------
            【评分规则】
            - 每个选项独立评分
            - 分数范围：-1.0 到 1.0
              - 越高 = 越支持该选项是正确答案  
              - 越低 = 越反对该选项  
              - 0 = 中性／证据不足  
            
            -----------------------------------
            【输出格式要求】
            你必须只输出以下 JSON（不要包含任何额外文字）：
            
            {{
                "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
                "reasoning": "",
                "evidences": []
            }}
            
            字段解释：
            - scores：你对每个选项的重新评估分数  
            - reasoning：80–120 字的中文解释，说明你的判断逻辑  
            - evidences：2–3 条关键证据点，每条 ≤20 字  
            
            -----------------------------------
            【可使用的信息】
            1. 当前轮次：{cur_round}
            医学问题：{question_state.question}
            问题选项：{[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            上一轮观点（previous_opinion）：
            {previous_opinion_str}
            领导者总结：{mdt_leader_summary}
            本轮智能体新增证据：{new_evidence}
            
            请根据以上内容完成你的本轮 soft update，并输出 JSON。

            """
        return prompt

    def _build_update_agent_opinions_reasoning_prompt(
            self,
            patient_state: PatientState,
            role: RoleType,
            current_round: DialogueRound,
            previous_opinion: RoleOpinion,
            treatment_options: List[TreatmentOption],
    ) -> str:
        """构建更新角色意见的推理提示"""

        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量",
            RoleType.NUTRITIONIST: "营养师，关注患者营养状况和营养支持治疗",
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善",
        }
        dialogue_text = "\n".join(
            [
                f"{msg.role.value}当前的观点是: {msg.content}"
                for msg in current_round.messages
            ]
        )
        logger.info("dialogue_text: %s", dialogue_text)
        prompt = f"""
你是一名医疗多学科团队（MDT, Multidisciplinary Team）的成员，当前身份为 **{role.value}**。  
请根据以下患者信息和团队对话内容，对每个治疗方案进行综合评估并重新打分。

==============================
【患者信息】
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 症状: {', '.join(patient_state.symptoms)}
- 合并症: {', '.join(patient_state.comorbidities)}
- 实验室结果: {json.dumps(patient_state.lab_results, ensure_ascii=False, indent=2)}
- 生命体征: {json.dumps(patient_state.vital_signs, ensure_ascii=False, indent=2)}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

==============================
【角色历史意见】
- 角色身份: {role_descriptions.get(role, role.value)}
- 上轮推理: {previous_opinion.reasoning}
- 上轮治疗偏好: {json.dumps(previous_opinion.treatment_preferences, ensure_ascii=False, indent=2)}
- 上轮核心关注点: {json.dumps(previous_opinion.concerns, ensure_ascii=False, indent=2)}
- 上轮完整意见: {json.dumps(previous_opinion.__dict__, ensure_ascii=False, indent=2)}

==============================
【多学科团队对话记录】
{dialogue_text}

==============================
【当前角色偏好】
{role.value} 当前的治疗偏好: {json.dumps(previous_opinion.treatment_preferences, ensure_ascii=False, indent=2)}

==============================
【治疗选项列表】
{[option.value for option in treatment_options]}

==============================
【任务要求】
请从 **{role.value}** 专业角度出发，综合分析：
- 患者整体病情及特征；
- 本角色历史意见与关注点；
- 其他学科成员最新发言；
- 本轮讨论的共识与分歧；

为每个治疗选项进行重新评估。


输出结果需满足 **RoleOpinion 类** 的结构：
1. `treatment_preferences`：字典，键为选项索引（如 "A", "B"），值为 -1~1 偏好分；  
2. `reasoning`：字符串（≤80字），说明总体打分逻辑；  
3. `confidence`：浮点数，0~1，表示对当前判断的信心；  
4. `concerns`：列表，包含 2~3 条 ≤20 字的关键担忧。

==============================
【输出要求】
- 严格返回 JSON，不得包含解释或额外文字；  
- 所有治疗选项必须完整包含在 `treatment_preferences` 中；  
- 键名、字段名、数据类型必须完全符合要求；  
- 输出可直接用于 RoleOpinion 实例化。

==============================
【输出示例】
{{
    "role": "{role.value}",
    "treatment_preferences": {{"A": 0.8, "B": 0.5, "C": -0.3}},
    "reasoning": "患者病灶可切除，手术预后优于保守治疗",
    "confidence": 0.85,
    "concerns": ["术后并发症风险", "患者耐受性", "术后康复周期"]
}}
注意：`treatment_preferences` 的键必须使用选项索引（如 "A"、"B"），不要使用选项全文。
"""

        logger.info("更新立场的prompt: %s", prompt)
        return prompt

    def generate_treatment_reasoning_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            question_options: List[medqa_types.QuestionOption] = None,
            dataset_name: str = None
    ) -> str:
        """
        生成角色的初始意见
        :param question_state: 问题描述
        :param role: 角色信息
        :param question_options: 问题选项
        :param dataset_name: 数据集名称
        :return: 返回response
        """

        prompt = self._build_treatment_reasoning_prompt_medqa(
            question_state, role, question_options, dataset_name
        )

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"你是医学多学科团队（MDT）成员，身份：{role.value}，角色描述：{role.description}，权重：{role.weight}。"
                                f"你的任务：根据题目和选项评估每个选项对最可能正确答案的支持程度。"
                                f"严格输出 JSON，不要输出其他文字。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    question_state, role, question_options
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_reasoning(
                question_state, role, question_options
            )

    def generate_all_treatment_reasoning_meqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            opinion: QuestionOpinion,
            question_options: List[medqa_types.QuestionOption] = None,
            dataset_name: str = None
    ) -> str:
        """
        每个智能体基于初始意见生成对话
        :param question_state:
        :param role:
        :param opinion:
        :param question_options:
        :param dataset_name:
        :return:
        """

        prompt = self._build_all_treatment_reasoning_prompt_meqa(
            question_state,
            role,
            opinion,
            question_options,
            dataset_name
        )

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是多学科医疗团队（MDT）的成员，当前身份为 **{role.value}**。描述: {role.description}, 权重: {role.weight}"
                                       f"你的任务是根据你之前生成的初始意见，为每个问题选项生成 **初始陈述**。"
                                       f"分析每个选项，解释其可能正确或不正确的原因，并参考你的初始意见中的评分、推理和证据。"
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                logger.debug(f"[DEBUGING第一轮发言]LLM response debug 当前关注solving: {role.value}: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    question_state, role
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_reasoning(
                question_state, role
            )

    def _build_all_treatment_reasoning_prompt_meqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            opinion: QuestionOpinion,
            question_options: List[medqa_types.QuestionOption] = None,
            dataset_name: str = None
    ) -> str:
        if dataset_name in ["medqa", "pubmedqa", "symcat", "ddxplus", "medbullets"]:
            prompt = f"""
            你是一个多学科医疗团队（MDT）的一名成员，当前身份是 **{role.value}**。
            描述: {role.description}, 权重: {role.weight}

            你的任务是基于你之前生成的**初步意见（initial opinion）**，为每个选项生成清晰的**初步陈述（initial statement）**。

            请严格遵守以下要求：

            ==============================
            输入信息

            * 问题：{question_state.question}
            * 选项：{[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            * 初步意见（不得修改）：

              * 各选项评分：{opinion.scores}
              * 推理：{opinion.reasoning}
              * 证据：{opinion.evidences}

            ==============================
            任务要求

            1. 你必须分析**每个选项**，说明为什么可能正确或不正确。
            2. 你的分析必须严格基于初步意见中的评分、推理和证据，不得引入与初步意见冲突的新观点。
            3. **不得重新评分，不得改变倾向**。
            4. 在结尾必须明确表达你的最终推荐选项，例如：
               “因此，根据我的分析，我认为最可能正确的选项是 X。”
            5. 输出应简洁流畅，150–200字，适合 MDT 讨论环境。
            6. 不需要 JSON，仅自然语言说明。

            ==============================
            输出示例

            “选项A …；选项B …；选项C …；……  
            因此，我认为最可能正确的是 C。”

            ==============================
            现在请生成你的初步陈述：
            """
        return prompt

    def _build_treatment_reasoning_prompt_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            role: Union[RoleType, RoleRegistry],
            question_options: List[medqa_types.QuestionOption] = None,
            dataset_name: str = None
    ) -> str:
        """构建MedQA场景下的治疗推理提示词"""
        # role_name = role.name
        role_value = role.value
        role_desc = role.description
        role_weight = role.weight
        if dataset_name in ["medqa", "pubmedqa", "symcat", "ddxplus", "medbullets"]:
            prompt = f"""
            你是医学多学科团队（MDT）系统的一名角色：
                - 身份：{role_value}
                - 角色描述：{role_desc}
                - 权重：{role_weight}
                
                你的任务是对题目进行专业分析、推理与量化评分。
                
                ==============================
                题目：
                {question_state.question}
                
                选项：
                {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
                ==============================
                
                请严格按照以下流程思考（必须按顺序执行）：
                
                1. **关键证据提取**
                   - 列出题目中与判断相关的关键线索（症状、机制、实验、药理、影像、逻辑关系等）
                
                2. **推理过程（非常关键）**
                   - 基于医学知识与证据链，推理哪一个选项最合理
                   - 在推理末尾，用一句话明确写出：
                     **“Answer = <A/B/C/D/E>”**
                
                3. **量化评分（-1.0 ~ 1.0）**
                   - 你的评分必须符合以下规则：
                     - Answer 对应的选项必须获得最高分
                     - 该选项的分数必须比次优选项高 ≥ 0.3
                     - 分数必须能反映你在推理中的证据强度
                     - 禁止所有选项分数过于接近
                   - 所有分数必须与 “Answer = X” 保持一致
                
                4. **反思（Reflection）**
                   - 检查 scores 是否与推理一致
                   - 如不一致，必须修正 scores
                   - 简要说明是否进行了自我修正
                
                5. **严格 JSON 输出（禁止额外文字）**：
                {{
                  "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
                  "reasoning": "<推理过程（包含 Answer = X）>",
                  "evidences": ["<证据1>", "<证据2>"],
                  "final_choice": "<A/B/C/D/E>",
                  "reflection": "<反思说明>"
                }}
            """
            # prompt = f"""
            # 你是医学多学科团队（MDT, Multidisciplinary Team）的一名智能体成员，
            # 身份：{role_value}，角色描述：{role_desc}，角色权重：{role_weight}。
            # 你需要根据题目做出医学推理，并对每个选项进行量化评分。
            #
            # ==============================
            # 题目：
            # {question_state.question}
            #
            # 选项：
            # {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            # ==============================
            #
            # 你的任务包括：
            #
            # 1. **识别关键证据**
            #    提取题目中支持或反驳各选项的关键证据（症状、实验室检查、影像、机制、数据等）。
            #
            # 2. **推理正确答案**
            #    基于医学知识和证据链，推理哪个选项最可能正确，并给出理由。
            #
            # 3. **严格量化评分（-1.0 ~ 1.0）**
            #    必须严格按照以下区间打分：
            #    - 最合理、最可能正确 → 0.9~1.0
            #    - 次优、部分合理 → 0.4~0.6
            #    - 不太合理、证据弱 → -0.4~0.2
            #    - 明显错误、与证据相反 → -0.8~-0.5
            #    - 完全错误、方向相反 → -1.0
            #    【约束】：
            #    - 最合理选项分数必须比次优选项高至少 0.3
            #    - 禁止所有选项给相近分数
            #    - 分数必须能反映你的推理过程
            #
            # 4. **反思（Reflection）**
            #    - 对你刚才生成的评分和证据强度进行复核
            #    - 检查是否与证据一致，是否有分数区间违规，是否最合理选项高于次优选项至少 0.3
            #    - 如发现问题，调整评分和证据强度
            #    - 输出反思结论的简短说明
            #
            # 5. **输出格式（严格 JSON，不加任何文字）**
            #
            # 请严格输出如下格式（JSON）：
            # {{
            #   "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
            #   "reasoning": "<简短推理说明>",
            #   "evidences": ["<证据1>", "<证据2>"],
            #   "reflection": "<简短反思说明>"
            # }}
            #
            # 评分流程必须遵循：
            # 列证据 → 推理出最可能正确答案 → 根据推理赋分 → 自我反思 → 输出 JSON。
            #
            # """

            # prompt = f"""
            # 你是医学多学科团队（MDT, Multidisciplinary Team）的一名智能体成员，身份：{role_value}，角色描述：{role_desc}，角色权重：{role_weight}。
            #
            # 题目：
            # {question_state.question}
            #
            # 选项：
            # {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            #
            # 任务：
            # 1. **识别关键证据**：根据题目内容（症状、检查数据、实验结果等）找出支持或反驳各选项的证据。
            # 2. **推理正确答案**：结合你的专业知识，基于证据进行推理，得出最可能的正确答案。
            # 3. **打分（-1.0 到 1.0）**：
            #    - 正确答案应接近 0.8~1.0
            #    - 明显错误选项应接近 -0.8~-1.0
            #    - 其他选项根据证据强度和合理性打中间分
            # 4. **证据强度**：根据证据可靠性和逻辑清晰度输出 0~1 的数值，用于加权聚合。
            # 5. **输出要求**：
            #    - 输出严格 JSON 格式
            #    - 包含字段：
            #      ```json
            #      {{
            #        "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
            #        "reasoning": "<简短推理说明>",
            #        "evidence_strength": 0.0,
            #        "evidences": ["<证据1>", "<证据2>"]
            #      }}
            #      ```
            #    - 不要输出其他文字或说明
            # 6. **打分逻辑链条**：
            #    - 先列出证据 → 再推理出正确答案 → 然后根据推理结果打分 → 输出证据强度
            # """
            # prompt = f"""
            # 你是医学多学科团队（MDT）成员，身份：{role_value}，角色描述：{role_desc}, 角色权重：{role_weight}。
            #
            # 题目：{question_state.question}
            # 选项：{[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            #
            # 任务：
            # 1. 推理出最可能的正确答案。
            # 2. 对每个选项打分（-1.0 至 1.0），表示该选项对正确答案的支持程度：
            #    - 正确答案接近 1.0
            #    - 明显错误选项接近 -1.0
            # 3. 简短说明评分理由（reasoning）。
            # 4. 提供关键证据（evidences）。
            # 5. 输出严格 JSON 格式，示例：
            # {{
            #   "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
            #   "reasoning": "<简短推理说明>",
            #   "evidence_strength": 0.0,
            #   "evidences": ["<证据1>", "<证据2>"]
            # }}
            # 6. 不要输出额外文字，严格按 JSON 格式返回。
            # """
            # return prompt
            # prompt = f"""
            # 你是多学科医疗团队（MDT）成员，当前身份为 **{role_value}**，角色描述：{role_desc}，权重：{role_weight}。
            # 你的任务是根据题目的内容、症状类型、推理结构和涉及的医学领域，推理出最可能的正确答案，并为每个选项分配评分，表示该选项支持正确答案的程度。
            #
            # ==============================
            # **题目（QUESTION）：**
            # {question_state.question}
            #
            # **选项（OPTIONS）：**
            # {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            #
            # ==============================
            # **指导原则（GUIDELINES）：**
            #
            # 1. **题目分析：**
            #    - 仔细阅读题目并理解关键信息：患者的症状、病史、检查结果（如实验室数据、影像学检查）等。
            #    - 仔细了解题目的问法和要求。
            #
            # 2. **推理正确答案：**
            #    - 根据题目内容和你的专业知识，推理出最可能的正确答案。你需要综合考虑患者的临床表现、实验室检查、诊断标准等，找到最符合的诊断或治疗选择。
            #
            # 3. **评估每个选项：**
            #    - 一旦推理出最可能的正确答案，就需要评估每个选项：
            #      - 选项是否支持正确答案？
            #      - 选项与题目内容的关系如何？
            #      - 选项是否提供了充足的证据来支持其为正确答案？
            #
            # 4. **评分范围：** -1.0 至 1.0
            #    - 分数越高表示该选项更支持正确答案，分数越低表示该选项不太可能是正确答案。
            #
            # 5. **输出格式：**
            #    请返回一个严格的 JSON 格式，包含以下内容：
            #    ```json
            #    {{
            #         "scores": {{"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}},
            #        "reasoning": "<解释评分的推理过程>",
            #        "evidence_strength": 0.0,
            #        "evidences": [
            #            "<证据1>",
            #            "<证据2>",
            #            "<证据3>"
            #        ]
            #    }}
            # """
        return prompt

    def generate_treatment_reasoning(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            knowledge_context: Dict[str, Any] = None,
            treatment_options: List[TreatmentOption] = None,
    ) -> str:
        """生成治疗推理"""

        prompt = self._build_treatment_reasoning_prompt(
            patient_state, role, treatment_option, knowledge_context, treatment_options
        )

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是多学科会诊（MDT, Multidisciplinary Team）的一名成员,当前身份是一位专业的{role.value}，请基于患者信息和角色专业性提供治疗推理,并对每个治疗选项的置信度进行打分",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    patient_state, role, treatment_option
                )
        except Exception as e:
            logger.error(f"1111 LLM generation failed: {e}")
            return self._generate_template_reasoning(
                patient_state, role, treatment_option
            )

    def generate_focus_treatment_reasoning(
            self,
            patient_state: PatientState,
            role: RoleType,
            opinion: RoleOpinion,
            treatment_option: TreatmentOption,
            treatment_options: List[TreatmentOption] = None,
    ) -> str:
        """生成聚焦治疗选项的推理"""

        prompt = self._build_focus_treatment_reasoning_prompt(
            patient_state,
            role,
            opinion,
            treatment_option,
            knowledge_context,
            treatment_options,
        )

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是多学科会诊（MDT, Multidisciplinary Team）的一名成员，你是一位专业的{role.value}，请基于患者信息和角色专业性提供治疗推理,并对每个治疗选项的置信度进行打分",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(
                    patient_state, role, treatment_option
                )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_reasoning(
                patient_state, role, treatment_option
            )

    def generate_dialogue_response_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            role: Union[RoleType, RoleRegistry],
            current_opinion: Union[RoleOpinion, QuestionOpinion] = None,
            dialogue_history: List[Dict] = None,
            mdt_leader_summary: str = None,
            dataset_name: str = None
    ) -> str:
        """
        基于新立场进行一次多轮对话
        :param question_state:
        :param question_options:
        :param role:
        :param current_opinion:
        :param dialogue_history:
        :param mdt_leader_summary:
        :param dataset_name:
        :return:
        """

        prompt = self._build_dialogue_response_prompt_medqa(
            question_state,
            question_options,
            role,
            current_opinion,
            dialogue_history,
            mdt_leader_summary,
            dataset_name
        )
        try:
            print(f"DEBUG: self.client = {self.client}")
            if self.client:
                print("DEBUG: 使用LLM客户端生成对话")
                # 使用更高的temperature增加多样性
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"你是多学科医疗团队（MDT）的一名成员，当前身份为 **{role.value}**。描述: {role.description}, 权重: {role.weight}"
                                "你的任务是在医学知识问答场景中，以简洁、自然、有个人特色的方式参与推理型讨论，"
                                "并根据 MDT_LEADER 的总结与讨论方向生成你的观点。你需要根据你的医学专业视角，"
                                "结合上一轮的观点和 MDT_LEADER 的指导意见，提供清晰、严谨的医学推理。"
                                "请注意，不要给出最终答案，而是帮助团队逐步形成共识。"
                            )
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                print("DEBUG查看错误: LLM响应原始内容:", response)
                logger.info(f"新方案_MDT_LEADER生成response:{response}")
                response_text = response.choices[0].message.content.strip()
                logger.info(f"DEBUG: LLM响应 response_text: {response_text}")
                return response_text
            else:
                print("DEBUG: 没有LLM客户端，使用模板回退")
                # 如果没有LLM，使用模板化回退
                try:
                    print("DEBUG: 尝试调用 _generate_template_dialogue_fallback")
                    result = self._generate_template_dialogue_fallback(
                        question_state, role, treatment_option, discussion_context
                    )
                    print(f"DEBUG: 模板方法返回: {result}")
                    return result
                except AttributeError as ae:
                    print(f"🚨🚨🚨 DEBUG: AttributeError caught: {ae}")
                    print(f"🚨🚨🚨 DEBUG: Returning hardcoded fallback")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
                except Exception as e:
                    print(f"🚨🚨🚨 DEBUG: Other exception: {type(e).__name__}: {e}")
                    print(f"🚨🚨🚨 DEBUG: Returning hardcoded fallback")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
                except Exception as ee:
                    print(f"DEBUG: 其他异常: {ee}")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
        except Exception as e:
            print(f"DEBUG: 异常发生: {e}")
            logger.error(f"Dialogue response generation failed: {e}")
            return self._generate_template_dialogue_fallback(
                patient_state, role, treatment_option, discussion_context
            )

    def generate_dialogue_response(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            discussion_context: str,
            knowledge_context: Dict[str, Any] = None,
            current_stance: RoleOpinion = None,
            dialogue_history: List[Dict] = None,
    ) -> str:
        """生成自然的多轮对话回应 - 减少模板化"""

        prompt = self._build_dialogue_response_prompt(
            patient_state,
            role,
            treatment_option,
            discussion_context,
            knowledge_context,
            current_stance,
            dialogue_history,
        )

        try:
            print(f"DEBUG: self.client = {self.client}")
            if self.client:
                print("DEBUG: 使用LLM客户端生成对话")
                # 使用更高的temperature增加多样性
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": f"你是多学科会诊（MDT, Multidisciplinary Team）的一名成员，你是一位专业的{self._get_role_system_prompt(role)}，请和其他智能体进行讨论，并保持一致的立场，可能需要讨论多轮。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=min(self.config.temperature + 0.2, 1.0),  # 增加随机性
                    max_tokens=self.config.max_tokens,
                    presence_penalty=0.3,  # 减少重复
                    frequency_penalty=0.3,  # 增加词汇多样性
                )
                logger.info(f"生成response:{response}")
                response_text = response.choices[0].message.content.strip()
                logger.info(f"DEBUG: LLM响应 response_text: {response_text}")
                return response_text
            else:
                print("DEBUG: 没有LLM客户端，使用模板回退")
                # 如果没有LLM，使用模板化回退
                try:
                    print("DEBUG: 尝试调用 _generate_template_dialogue_fallback")
                    result = self._generate_template_dialogue_fallback(
                        patient_state, role, treatment_option, discussion_context
                    )
                    print(f"DEBUG: 模板方法返回: {result}")
                    return result
                except AttributeError as ae:
                    print(f"🚨🚨🚨 DEBUG: AttributeError caught: {ae}")
                    print(f"🚨🚨🚨 DEBUG: Returning hardcoded fallback")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
                except Exception as e:
                    print(f"🚨🚨🚨 DEBUG: Other exception: {type(e).__name__}: {e}")
                    print(f"🚨🚨🚨 DEBUG: Returning hardcoded fallback")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
                except Exception as ee:
                    print(f"DEBUG: 其他异常: {ee}")
                    return f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}），作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
        except Exception as e:
            print(f"DEBUG: 异常发生: {e}")
            logger.error(f"Dialogue response generation failed: {e}")
            return self._generate_template_dialogue_fallback(
                patient_state, role, treatment_option, discussion_context
            )

    def _build_focus_treatment_reasoning_prompt(
            self,
            patient_state: PatientState,
            role: RoleType,
            opinion: RoleOpinion,
            treatment_option: TreatmentOption,
            knowledge_context: Dict[str, Any] = None,
            treatment_options: List[TreatmentOption] = None,
    ) -> str:
        """构建聚焦治疗选项的推理提示词"""

        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量",
            RoleType.NUTRITIONIST: "营养师，关注患者营养状况和营养支持治疗",
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善",
        }

        prompt = f"""
你是一名医疗多学科团队（MDT, Multidisciplinary Team）的成员，负责从{role.value}专业角度给出本轮综合治疗建议。  
请根据以下信息，结合本轮对话内容，对每个治疗方案进行重新评估。
患者信息：
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 症状: {', '.join(patient_state.symptoms)}
- 合并症: {', '.join(patient_state.comorbidities)}
- 实验室结果: {json.dumps(patient_state.lab_results, ensure_ascii=False, indent=2)}
- vital_signs: {json.dumps(patient_state.vital_signs, ensure_ascii=False, indent=2)}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

角色身份: {role_descriptions.get(role, role.value)}

治疗选项: {[option.value for option in treatment_options]}

请从{role.value}的专业角度，为该患者的{treatment_option.value}治疗提供详细的推理分析，包括：
1. 治疗选项偏好值大于0的帮我分析支持原因，治疗选项偏好值小于0的帮我分析反对原因
2. 可能的风险和注意事项
3. 与患者具体情况的匹配度

请用专业但易懂的语言回答，控制在200字以内。
"""

        if knowledge_context:
            prompt += f"\n\n相关医学知识：\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}"

        return prompt

    def _build_treatment_reasoning_prompt(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            knowledge_context: Dict[str, Any] = None,
            treatment_options: List[TreatmentOption] = None,
    ) -> str:
        """构建治疗推理提示词"""

        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量",
            RoleType.NUTRITIONIST: "营养师，关注患者营养状况和营养支持治疗",
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善",
        }

        prompt = f"""
你是一名医疗多学科团队（MDT, Multidisciplinary Team）的成员，负责从 {role.value} 专业角度给出提供治疗推理。  

==============================
【患者基本信息】
- 患者ID: {patient_state.patient_id}
- 主要诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 主要症状: {', '.join(patient_state.symptoms)}
- 合并症: {', '.join(patient_state.comorbidities)}
- 实验室结果: {json.dumps(patient_state.lab_results, ensure_ascii=False, indent=2)}
- 生命体征: {json.dumps(patient_state.vital_signs, ensure_ascii=False, indent=2)}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

==============================
【角色信息】
- 当前角色: {role.value}
- 角色定义: {role_descriptions.get(role, role.value)}

==============================
【治疗选项列表】
{[option.value for option in treatment_options]}


输出需满足以下格式：

==============================
【输出要求】
- 输出为严格 JSON（不要包含解释或额外文字）；
- 字段说明如下：
  1. `treatment_preferences`: 字典，键为治疗选项索引（如 "A"、"B"），值为偏好分（-1~1 浮点数）；
  2. `reasoning`: 字符串，≤80字，说明总体打分逻辑；
  3. `confidence`: 浮点数，0~1，代表本角色对当前判断的信心；
  4. `concerns`: 列表，包含 2~3 条 ≤20 字的关键担忧。

==============================
【输出示例】
{{
    "treatment_preferences": {{"A": 0.8, "B": 0.4, "C": -0.3}},
    "reasoning": "患者病灶可切除，手术预后优于保守治疗",
    "confidence": 0.85,
    "concerns": ["术后并发症风险", "患者耐受性", "术后康复周期"]
}}
"""

        if knowledge_context:
            prompt += f"\n\n相关医学知识：\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}"

        return prompt

    def _generate_template_reasoning(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
    ) -> str:
        """生成模板化推理（降级方案）"""

        # 基础患者信息
        age_factor = "年龄较大" if patient_state.age > 65 else "年龄适中"
        stage_severity = (
            "早期"
            if "I" in patient_state.stage
            else (
                "中晚期"
                if "II" in patient_state.stage or "III" in patient_state.stage
                else "晚期"
            )
        )

        templates = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: f"""
基于患者{patient_state.diagnosis}（{patient_state.stage}期）的临床特征分析：
1. 肿瘤分期评估：{stage_severity}肿瘤，手术切除可获得良好的局部控制效果
2. 患者年龄因素：{age_factor}（{patient_state.age}岁），需评估手术耐受性和预期获益
3. 病理学考虑：根据肿瘤大小、位置和侵犯范围，手术是标准一线治疗选择
4. 预后评估：完整切除可显著改善长期生存率，建议积极手术治疗
5. 风险效益比：手术获益明显大于风险，符合循证医学证据
                """.strip(),
                TreatmentOption.CHEMOTHERAPY: f"""
针对{patient_state.diagnosis}（{patient_state.stage}期）的化疗适应症分析：
1. 分期特点：由于{stage_severity}疾病的特征，化疗可有效控制全身微转移病灶
2. 年龄考量：考虑到{age_factor}患者的生理特点，需选择合适的化疗方案和剂量调整
3. 治疗目标：因为系统性治疗的优势，可显著降低复发风险，提高无病生存期
4. 方案选择：基于患者耐受性评估，建议采用标准化疗方案并个体化调整
5. 疗效预期：由于临床研究数据支持，化疗可显著改善患者预后和生活质量
                """.strip(),
                TreatmentOption.RADIOTHERAPY: f"""
放射治疗在{patient_state.diagnosis}（{patient_state.stage}期）中的应用价值：
1. 适应症评估：{stage_severity}病变，放疗可提供精准的局部区域控制
2. 技术选择：现代放疗技术可最大化肿瘤剂量，最小化正常组织损伤
3. 联合治疗：与手术或化疗联合可获得协同效应
4. 年龄因素：{age_factor}患者通常能较好耐受分次放疗
5. 预期效果：可显著降低局部复发率，改善生活质量
                """.strip(),
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: f"""
手术护理评估和计划制定：
1. 术前准备：患者{age_factor}，需加强术前宣教和心理支持
2. 风险评估：评估患者手术耐受性，制定个性化护理计划
3. 术后监护：密切观察生命体征，预防并发症发生
4. 康复指导：制定渐进式康复计划，促进患者早期恢复
5. 家属教育：指导家属参与护理，提供持续支持
                """.strip(),
                TreatmentOption.CHEMOTHERAPY: f"""
化疗护理管理和安全监护：
1. 用药安全：严格执行化疗药物配置和给药流程
2. 副作用监测：密切观察恶心呕吐、骨髓抑制等不良反应
3. 感染预防：{age_factor}患者免疫
4. 营养支持：评估营养状况，制定个性化营养干预方案
5. 心理护理：提供情感支持，帮助患者建立治疗信心
                """.strip(),
                TreatmentOption.RADIOTHERAPY: f"""
放疗期间护理要点和注意事项：
1. 皮肤护理：指导患者正确的皮肤保护方法，预防放射性皮炎
2. 体位固定：确保每次治疗体位准确，提高放疗精度
3. 副作用管理：监测和处理放疗相关不良反应
4. 生活指导：{age_factor}患者需要更多的生活护理支持
5. 随访教育：制定放疗后的长期随访和自我管理计划
                """.strip(),
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.SURGERY: f"""
手术相关心理干预和支持策略：
1. 术前焦虑：{age_factor}患者对手术的恐惧和担忧需要专业疏导
2. 应对机制：评估患者现有的心理应对资源和支持系统
3. 认知重构：帮助患者建立对手术治疗的正确认知
4. 家庭支持：指导家属如何提供有效的情感支持
5. 术后适应：制定术后心理康复计划，促进心理健康恢复
                """.strip(),
                TreatmentOption.CHEMOTHERAPY: f"""
化疗期间心理健康维护和干预：
1. 情绪管理：帮助患者应对化疗带来的情绪波动和抑郁倾向
2. 治疗依从性：通过心理支持提高患者的治疗配合度
3. 生活质量：关注{age_factor}患者的生活质量和社会功能
4. 希望重建：帮助患者维持积极的治疗态度和社会希望
5. 压力缓解：教授有效的压力管理和放松技巧
                """.strip(),
                TreatmentOption.RADIOTHERAPY: f"""
放疗期间心理支持和干预措施：
1. 治疗适应：帮助患者适应长期的放疗治疗过程
2. 身体形象：处理放疗可能带来的身体形象改变问题
3. 社会支持：{age_factor}患者更需要社会支持网络的维护
4. 恐惧管理：缓解对放射治疗的恐惧和误解
5. 生活规划：协助患者制定治疗期间的生活安排和目标
                """.strip(),
            },
        }

        role_templates = templates.get(role, {})
        default_reasoning = f"""
作为{role.value}，对于{patient_state.diagnosis}（{patient_state.stage}期）患者的{treatment_option.value}治疗建议：
考虑到患者{age_factor}（{patient_state.age}岁）的具体情况，{treatment_option.value}治疗具有重要的临床价值。
需要综合评估患者的整体状况，制定个性化的治疗方案，确保治疗效果的同时最大化患者的生活质量。
建议在多学科团队协作下，为患者提供最优的医疗服务。
        """.strip()

        return role_templates.get(treatment_option, default_reasoning)

    def _generate_template_treatment_plan(
            self, patient_state: PatientState
    ) -> Dict[str, Any]:
        """生成模板化治疗方案"""

        return {
            "primary_treatment": f"针对{patient_state.diagnosis}的标准治疗方案",
            "supportive_care": "症状管理和营养支持",
            "timeline": "治疗周期约3-6个月",
            "expected_outcomes": "预期良好的治疗反应",
            "side_effects": "常见副作用的预防和管理",
            "follow_up": "定期复查和评估",
            "generated_by": "template",
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_template_timeline_events(
            self, patient_state: PatientState, days_ahead: int
    ) -> List[Dict[str, Any]]:
        """生成模板化时间线事件"""

        events = []
        for day in range(1, min(days_ahead + 1, 31)):
            if day % 7 == 0:  # 每周检查
                events.append(
                    {
                        "day": day,
                        "event_type": "检查",
                        "description": "常规血液检查和体征监测",
                        "severity": 2,
                        "requires_intervention": False,
                    }
                )

            if day % 14 == 0:  # 双周治疗
                events.append(
                    {
                        "day": day,
                        "event_type": "治疗",
                        "description": "按计划进行治疗",
                        "severity": 3,
                        "requires_intervention": True,
                    }
                )

        return events

    def _generate_template_dialogue_fallback_NEW(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            discussion_context: str,
    ) -> str:
        """生成上下文相关的模板化对话回应"""
        print("🔥🔥🔥 ENTERING NEW FALLBACK METHOD 🔥🔥🔥")
        print(
            f"🔥🔥🔥 Parameters: role={role.value}, treatment={treatment_option.value}, context={discussion_context}"
        )
        result = f"🔥🔥🔥 NEW FALLBACK METHOD CALLED! Role: {role.value}, Treatment: {treatment_option.value}, Context: {discussion_context} 🔥🔥🔥"
        print(f"🔥🔥🔥 Returning: {result}")
        return result

    def _generate_template_dialogue_fallback(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            discussion_context: str,
    ) -> str:
        """生成上下文相关的模板化对话回应"""
        return self._generate_template_dialogue_fallback_NEW(
            patient_state, role, treatment_option, discussion_context
        )

    def format_opinion_for_prompt(self, current_opinion: Union[RoleOpinion, QuestionOpinion], role_name: str):
        """
        将单个智能体的 opinion 转换为适合 MDT Prompt 输入的完整自然语言结构。

        current_opinion: dict, 包含 keys: scores, reasoning, evidences, evidence_strength
        role_name: str, 当前角色名
        返回: str，可直接放入 prompt
        """
        # 选项评分压缩成一行
        scores_str = ", ".join([f"{opt}={score}" for opt, score in current_opinion.scores.items()])

        # 核心观点取全部 reasoning
        core_reasoning = current_opinion.reasoning.strip()

        # 关键证据列出全部
        evidences = current_opinion.evidences
        evidences_str = "\n- ".join(evidences)
        if evidences_str:
            evidences_str = "- " + evidences_str  # 加上列表标记

        # 最终字符串
        formatted = (
            f"{role_name}观点:\n"
            f"选项评分: {scores_str}\n"
            f"核心观点: {core_reasoning}\n"
            f"关键证据:\n{evidences_str}\n"
        )
        return formatted

    def _build_dialogue_response_prompt_medqa(
            self,
            question_state: medqa_types.MedicalQuestionState,
            question_options: List[medqa_types.QuestionOption],
            role: Union[RoleType, RoleRegistry],
            current_opinion: Union[RoleOpinion, QuestionOpinion],
            dialogue_history: List[Dict[str, str]],
            mdt_leader_summary: str = None,
            dataset_name: str = None
    ):

        # # 构建立场信息
        stance_info = self.format_opinion_for_prompt(current_opinion, role.value)

        if dataset_name in ["medqa", "pubmedqa", "symcat", "ddxplus", "medbullets"]:
            prompt = f"""
            你是多学科会诊（MDT）的一名成员，当前身份为 **{role.value}**。描述: {role.description}, 权重: {role.weight} "
            你的任务是在医学知识问答场景中，以简洁、自然、有个人特色的方式参与推理型讨论。
            
            请依据以下信息生成你的本轮观点：
            1. 医疗问题：{question_state.question}
            2. 选项: {[f"{option.name}: {question_state.options[option.name]}" for option in question_options]}
            3. 本轮 MDT_LEADER 的总结与讨论方向：{mdt_leader_summary}
            4. 你上一轮的观点（若有）：{stance_info or "无"}
            
            ==============================
            【发言要求】
            请严格控制在 **2–3 句** 内，包含以下要素：
            
            ① **承接上一轮**  
            - 若你上一轮有观点，请先用半句话说明你“保持/调整/反思”了哪些看法  
              （示例：“相比上一轮，我仍认为… / 我会稍作调整，因为…”）
            
            ② **回应 MDT_LEADER 的方向**  
            - 明确呼应 Leader 的总结或所强调的风险点、证据点或争议点  
              （示例：“结合 Leader 指出的 X，我认为…”）
            
            ③ **给出你的专业风格观点**  
            - 医学推理必须清晰、严谨，但不模板化  
            - 整体目标是帮助群体逐步形成共识
            
            ==============================
            【输出格式】
            - 仅输出自然语言文本，不要 JSON、不要列点和编号。  
            - 句子要顺畅、非模板化，能被视为 MDT 讨论中的真实对话。
            """
        return prompt

    def _build_dialogue_response_prompt(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            discussion_context: str,
            knowledge_context: Dict[str, Any] = None,
            current_stance: RoleOpinion = None,
            dialogue_history: List[Dict] = None,
    ) -> str:
        """构建对话回应提示词 - 强调自然性和个性化"""

        # 构建对话历史上下文
        history_context = ""
        if dialogue_history:
            recent_exchanges = dialogue_history
            history_context = "\n上一轮对话:\n"
            for i, exchange in enumerate(recent_exchanges):
                history_context += f"上一轮{i + 1}: {exchange.get('role', 'Unknown')} - {exchange.get('content', '')}...\n"

        logger.info(f"上一轮非自己的对话: {history_context}")

        # 构建立场信息
        stance_info = ""

        if current_stance:
            stance_value = current_stance.treatment_preferences.get(
                treatment_option.value, 0
            )
            if stance_value > 0.7:
                stance_info = "你对该治疗方案持积极态度"
            elif stance_value > 0:
                stance_info = "你对该治疗方案持谨慎支持态度"
            elif stance_value < -0.5:
                stance_info = "你对该治疗方案有较大担忧"
            else:
                stance_info = "你对该治疗方案持中性态度"
        logger.info(f"{role.value}当前立场Stance info: {stance_info}")
        prompt = f"""
你是一名医疗多学科团队（MDT, Multidisciplinary Team）的成员，当前身份为 **{role.value}**。  
请针对以下患者情况和讨论内容，给出自然、专业且角色特色鲜明的回应：

==============================
【患者信息】
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 症状: {', '.join(patient_state.symptoms)}
- 合并症: {', '.join(patient_state.comorbidities)}
- 实验室结果: {json.dumps(patient_state.lab_results, ensure_ascii=False, indent=2)}
- 生命体征: {json.dumps(patient_state.vital_signs, ensure_ascii=False, indent=2)}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

==============================
【讨论方案】
- 当前讨论的治疗方案: {treatment_option.value}

==============================
【讨论背景】
- 当前讨论背景: {discussion_context}
- 当前立场信息: {stance_info}
- 历史讨论内容: {history_context}

==============================
【任务要求】
请从 **{role.value}** 专业角度出发生成回应：
1. 回应自然流畅，避免模板化表达；
2. 体现你的专业角色特点和判断逻辑；
3. 考虑之前对话内容，保持连贯性；
4. 表达带有个人色彩，不千篇一律；
5. 长度控制在 2~3 句话，简洁有力；
6. 如有不同意见，礼貌且坚定地表达。

==============================
【输出要求】
- 仅返回文字回应，不得包含 JSON 或额外标记；
- 回应应可直接用于多智能体 MDT 对话系统。
"""

        return prompt

    def _get_role_system_prompt(self, role: RoleType) -> str:
        """获取角色特定的系统提示词"""

        role_prompts = {
            RoleType.ONCOLOGIST: "你是一位经验丰富的肿瘤科医生，专注于癌症治疗的疗效和安全性。你的回应应该基于循证医学，同时考虑患者的整体状况。说话风格专业但易懂，偶尔会引用临床经验。",
            RoleType.NURSE: "你是一位资深的肿瘤科护士，关注患者的日常护理和生活质量。你的回应应该实用、贴心，关注治疗的可行性和患者的舒适度。说话风格温和关怀，经常从护理角度思考问题。",
            RoleType.PSYCHOLOGIST: "你是一位临床心理学家，专注于癌症患者的心理健康。你的回应应该考虑患者的心理承受能力和情感需求。说话风格温暖支持，善于从心理角度分析问题。",
            RoleType.RADIOLOGIST: "你是一位放射科医生，专精于医学影像和放射治疗。你的回应应该基于影像学证据和放射治疗的技术特点。说话风格精确客观，经常引用影像学发现。",
            RoleType.PATIENT_ADVOCATE: "你是一位患者权益代表，致力于维护患者的最佳利益。你的回应应该平衡医疗建议和患者的价值观、偏好。说话风格坚定但富有同理心，经常站在患者角度思考。",
        }

        return role_prompts.get(
            role, f"你是一位{role.value}，请基于你的专业背景提供建议。"
        )

    def _parse_treatment_plan_response(self, response: str) -> Dict[str, Any]:
        """解析治疗方案响应"""
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()

            plan = json.loads(json_str)
            plan["generated_by"] = "llm"
            plan["timestamp"] = datetime.now().isoformat()
            return plan
        except:
            # 解析失败时返回文本格式
            return {
                "content": response,
                "generated_by": "llm_text",
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_template_dialogue_fallback(
            self,
            patient_state: PatientState,
            role: RoleType,
            treatment_option: TreatmentOption,
            discussion_context: str,
    ) -> str:
        """生成模板化对话回退响应"""

        # 基础角色模板
        role_templates = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: "从肿瘤学角度分析，手术治疗对该患者具有重要意义。",
                TreatmentOption.CHEMOTHERAPY: "化疗方案需要根据患者的具体病理特征进行个体化设计。",
                TreatmentOption.RADIOTHERAPY: "放射治疗在该患者的综合治疗方案中可发挥关键作用。",
                TreatmentOption.IMMUNOTHERAPY: "免疫治疗为该患者提供了新的治疗机会和希望。",
                TreatmentOption.PALLIATIVE_CARE: "姑息治疗能够有效改善患者的生活质量。",
                TreatmentOption.WATCHFUL_WAITING: "密切观察策略在当前阶段是合理的选择。",
            },
            RoleType.SURGEON: {
                TreatmentOption.SURGERY: "从外科角度评估，患者的手术适应症和风险需要综合考虑。",
                TreatmentOption.CHEMOTHERAPY: "新辅助化疗或辅助化疗的时机选择对手术效果很重要。",
                TreatmentOption.RADIOTHERAPY: "放疗与手术的配合时机需要多学科团队讨论决定。",
            },
            RoleType.RADIOLOGIST: {
                TreatmentOption.RADIOTHERAPY: "基于影像学评估，放疗的靶区设计和剂量分布需要精确规划。",
                TreatmentOption.SURGERY: "影像学检查为手术方案的制定提供了重要的解剖学参考。",
            },
            RoleType.NURSE: {
                TreatmentOption.CHEMOTHERAPY: "化疗期间的护理管理和不良反应监测是治疗成功的关键。",
                TreatmentOption.SURGERY: "围手术期护理对患者的康复具有重要意义。",
                TreatmentOption.PALLIATIVE_CARE: "姑息护理能够显著提升患者的舒适度和生活质量。",
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.CHEMOTHERAPY: "化疗期间的心理支持有助于患者更好地配合治疗。",
                TreatmentOption.SURGERY: "术前心理准备和术后心理康复同样重要。",
                TreatmentOption.PALLIATIVE_CARE: "心理关怀在姑息治疗中发挥着不可替代的作用。",
            },
        }

        # 获取角色特定模板
        role_specific = role_templates.get(role, {})
        base_template = role_specific.get(
            treatment_option,
            f"作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。",
        )

        # 添加患者特定信息
        patient_context = f"考虑到患者{patient_state.age}岁，诊断为{patient_state.diagnosis}（{patient_state.stage}期），"

        # 如果有讨论上下文，添加相关回应
        context_response = ""
        if discussion_context and len(discussion_context.strip()) > 0:
            if "安全性" in discussion_context or "风险" in discussion_context:
                context_response = "关于安全性方面的考虑，"
            elif "有效性" in discussion_context or "效果" in discussion_context:
                context_response = "从治疗效果的角度来看，"
            elif "费用" in discussion_context or "经济" in discussion_context:
                context_response = "在的经济方面，"

        return f"{patient_context}{context_response}{base_template}"
