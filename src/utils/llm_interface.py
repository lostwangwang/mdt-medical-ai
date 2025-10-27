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

import openai
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

from ..core.data_models import PatientState, TreatmentOption, RoleType, RoleOpinion, DialogueMessage

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
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
            if not api_key or api_key.strip() == "" or api_key == "test-api-key-for-testing":
                logger.info("No valid API key provided, LLM client will not be initialized. Using template fallback.")
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
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info(f"LLM client initialized successfully with model: {self.config.model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            logger.info("Falling back to template-based responses")
            self.client = None
    
    def generate_update_agent_opinions_reasoning(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        agent_dialogue: DialogueMessage,
        opinions_list: List[RoleOpinion]
    ) -> str:
        """生成更新角色意见的推理"""
        prompt = self._build_update_agent_opinions_reasoning_prompt(
            patient_state, role, agent_dialogue, opinions_list
        )
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位专业的{role.value}，请基于患者信息、角色专业性、对话上下文、上一轮对话和当前对话，更新角色意见、治疗偏好、置信度"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(patient_state, role, treatment_option)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")


    def _build_update_agent_opinions_reasoning_prompt(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        agent_dialogue: DialogueMessage,
        opinions_list: List[RoleOpinion]
    ) -> str:
        """构建更新角色意见的推理提示"""
         
        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量",
            RoleType.NUTRITIONIST: "营养师，关注患者营养状况和营养支持治疗",
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善"
        }
        
        prompt = f"""
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

当前对话的观点: {agent_dialogue.content}

{role.value}当前的治疗偏好: 
治疗选项列表：{[option.value for option in treatment_options]}

请从{role.value}专业角度，综合角色当前对话的观点，为每个治疗选项进行重新评估（结果需适配RoleOpinion类）：
1. treatment_preferences：字典，键为治疗选项（如"surgery"），值为-1~1的偏好度分；
2. reasoning：字符串，≤80字，说明打分理由；
3. confidence：0~1的浮点数，自身判断的可靠性；
4. concerns：列表，含2-3个字符串，每项≤20字，核心担忧。

**输出要求**：
- 仅返回JSON，不包含任何额外文本；
- 字段名严格匹配上述名称，类型符合要求；
- 治疗选项必须完整包含列表中的所有项。

示例输出：
{{
    "treatment_preferences": {{"surgery": 0.8, "chemotherapy": 0.5, ...}},
    "reasoning": "根据患者情况，积极治疗更优，手术获益明确",
    "confidence": 0.8,
    "concerns": ["手术并发症风险", "化疗耐受性"]
}}
"""
    
        
        return prompt


    def generate_treatment_reasoning(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None,
        treatment_options: List[TreatmentOption] = None
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
                        {"role": "system", "content": f"你是一位专业的{role.value}，请基于患者信息和角色专业性提供治疗推理,并对每个治疗选项的置信度进行打分"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(patient_state, role, treatment_option)
        except Exception as e:
            logger.error(f"1111 LLM generation failed: {e}")
            return self._generate_template_reasoning(patient_state, role, treatment_option)
    

    def generate_focus_treatment_reasoning(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        opinion: RoleOpinion,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None,
        treatment_options: List[TreatmentOption] = None
    ) -> str:
        """生成聚焦治疗选项的推理"""
        
        prompt = self._build_focus_treatment_reasoning_prompt(
            patient_state, role, opinion, treatment_option, knowledge_context, treatment_options
        )
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位专业的{role.value}，请基于患者信息和角色专业性提供治疗推理,并对每个治疗选项的置信度进行打分"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                logger.debug(f"LLM response debug: {response}")
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(patient_state, role, treatment_option)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_reasoning(patient_state, role, treatment_option)

    
    def generate_dialogue_response(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        discussion_context: str,
        knowledge_context: Dict[str, Any] = None,
        current_stance: RoleOpinion = None,
        dialogue_history: List[Dict] = None
    ) -> str:
        """生成自然的多轮对话回应 - 减少模板化"""
        
        prompt = self._build_dialogue_response_prompt(
            patient_state, role, treatment_option, discussion_context, 
            knowledge_context, current_stance, dialogue_history
        )
        
        try:
            print(f"DEBUG: self.client = {self.client}")
            if self.client:
                print("DEBUG: 使用LLM客户端生成对话")
                # 使用更高的temperature增加多样性
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": f"{self._get_role_system_prompt(role)}，请和其他智能体进行讨论，并保持一致的立场，可能需要讨论多轮。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=min(self.config.temperature + 0.2, 1.0),  # 增加随机性
                    max_tokens=self.config.max_tokens,
                    presence_penalty=0.3,  # 减少重复
                    frequency_penalty=0.3   # 增加词汇多样性
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
    
    def generate_professional_reasoning(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None
    ) -> str:
        """生成专业推理 - 角色特异性强化版"""
        
        prompt = self._build_professional_reasoning_prompt(
            patient_state, role, treatment_option, knowledge_context
        )
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": self._get_professional_system_prompt(role)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                return response.choices[0].message.content.strip()
            else:
                return self._generate_template_reasoning(patient_state, role, treatment_option)
        except Exception as e:
            logger.error(f"Professional reasoning generation failed: {e}")
            return self._generate_template_reasoning(patient_state, role, treatment_option)
    
    def _build_focus_treatment_reasoning_prompt(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        opinion: RoleOpinion,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None,
        treatment_options: List[TreatmentOption] = None
    ) -> str:
        """构建聚焦治疗选项的推理提示词"""
        
        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量",
            RoleType.NUTRITIONIST: "营养师，关注患者营养状况和营养支持治疗",
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善"
        }
        
        prompt = f"""
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
            RoleType.REHABILITATION_THERAPIST: "康复治疗师，关注患者功能恢复和生活质量改善"
        }
        
        prompt = f"""
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

治疗选项列表：{[option.value for option in treatment_options]}

请从{role.value}专业角度，为每个治疗选项完成以下任务（结果需适配RoleOpinion类）：
1. treatment_preferences：字典，键为治疗选项（如"surgery"），值为-1~1的偏好度分；
2. reasoning：字符串，≤80字，说明打分理由；
3. confidence：0~1的浮点数，自身判断的可靠性；
4. concerns：列表，含2-3个字符串，每项≤20字，核心担忧。

**输出要求**：
- 仅返回JSON，不包含任何额外文本；
- 字段名严格匹配上述名称，类型符合要求；
- 治疗选项必须完整包含列表中的所有项。

示例输出：
{{
    "treatment_preferences": {{"surgery": 0.8, "chemotherapy": 0.5, ...}},
    "reasoning": "根据患者情况，积极治疗更优，手术获益明确",
    "confidence": 0.8,
    "concerns": ["手术并发症风险", "化疗耐受性"]
}}
"""
        
        if knowledge_context:
            prompt += f"\n\n相关医学知识：\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}"
        
        return prompt
    
    def _build_treatment_plan_prompt(
        self,
        patient_state: PatientState,
        memory_context: Dict[str, Any],
        knowledge_context: Dict[str, Any] = None
    ) -> str:
        """构建治疗方案提示词"""
        
        prompt = f"""
患者基本信息：
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

患者历史记录：
{json.dumps(memory_context, ensure_ascii=False, indent=2)}

请制定一个综合的治疗方案，包括：
1. 主要治疗方案（手术、化疗、放疗等）
2. 辅助治疗措施
3. 预期疗效和时间安排
4. 可能的副作用和应对措施
5. 随访计划

请以JSON格式返回，包含以下字段：
- primary_treatment: 主要治疗方案
- supportive_care: 支持治疗
- timeline: 治疗时间线
- expected_outcomes: 预期结果
- side_effects: 副作用管理
- follow_up: 随访计划
"""
        
        if knowledge_context:
            prompt += f"\n\n参考医学指南：\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}"
        
        return prompt
    
    def _build_timeline_events_prompt(
        self,
        patient_state: PatientState,
        memory_context: Dict[str, Any],
        days_ahead: int
    ) -> str:
        """构建时间线事件提示词"""
        
        prompt = f"""
患者信息：
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 当前状态: {json.dumps(memory_context, ensure_ascii=False, indent=2)}

请为该患者生成未来{days_ahead}天的医疗事件时间线，包括：
1. 检查和检验安排
2. 治疗进展
3. 可能的症状变化
4. 医疗干预

请以JSON数组格式返回，每个事件包含：
- day: 第几天
- event_type: 事件类型（检查、治疗、症状等）
- description: 事件描述
- severity: 严重程度（1-5）
- requires_intervention: 是否需要医疗干预
"""
        
        return prompt
    
    def _generate_template_reasoning(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        treatment_option: TreatmentOption
    ) -> str:
        """生成模板化推理（降级方案）"""
        
        # 基础患者信息
        age_factor = "年龄较大" if patient_state.age > 65 else "年龄适中"
        stage_severity = "早期" if "I" in patient_state.stage else "中晚期" if "II" in patient_state.stage or "III" in patient_state.stage else "晚期"
        
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
                """.strip()
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
                """.strip()
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
                """.strip()
            }
        }
        
        role_templates = templates.get(role, {})
        default_reasoning = f"""
作为{role.value}，对于{patient_state.diagnosis}（{patient_state.stage}期）患者的{treatment_option.value}治疗建议：
考虑到患者{age_factor}（{patient_state.age}岁）的具体情况，{treatment_option.value}治疗具有重要的临床价值。
需要综合评估患者的整体状况，制定个性化的治疗方案，确保治疗效果的同时最大化患者的生活质量。
建议在多学科团队协作下，为患者提供最优的医疗服务。
        """.strip()
        
        return role_templates.get(treatment_option, default_reasoning)
    
    def _generate_template_treatment_plan(self, patient_state: PatientState) -> Dict[str, Any]:
        """生成模板化治疗方案"""
        
        return {
            "primary_treatment": f"针对{patient_state.diagnosis}的标准治疗方案",
            "supportive_care": "症状管理和营养支持",
            "timeline": "治疗周期约3-6个月",
            "expected_outcomes": "预期良好的治疗反应",
            "side_effects": "常见副作用的预防和管理",
            "follow_up": "定期复查和评估",
            "generated_by": "template",
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_template_timeline_events(
        self, 
        patient_state: PatientState, 
        days_ahead: int
    ) -> List[Dict[str, Any]]:
        """生成模板化时间线事件"""
        
        events = []
        for day in range(1, min(days_ahead + 1, 31)):
            if day % 7 == 0:  # 每周检查
                events.append({
                    "day": day,
                    "event_type": "检查",
                    "description": "常规血液检查和体征监测",
                    "severity": 2,
                    "requires_intervention": False
                })
            
            if day % 14 == 0:  # 双周治疗
                events.append({
                    "day": day,
                    "event_type": "治疗",
                    "description": "按计划进行治疗",
                    "severity": 3,
                    "requires_intervention": True
                })
        
        return events
    
    def _generate_template_dialogue_fallback_NEW(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        discussion_context: str
    ) -> str:
        """生成上下文相关的模板化对话回应"""
        print("🔥🔥🔥 ENTERING NEW FALLBACK METHOD 🔥🔥🔥")
        print(f"🔥🔥🔥 Parameters: role={role.value}, treatment={treatment_option.value}, context={discussion_context}")
        result = f"🔥🔥🔥 NEW FALLBACK METHOD CALLED! Role: {role.value}, Treatment: {treatment_option.value}, Context: {discussion_context} 🔥🔥🔥"
        print(f"🔥🔥🔥 Returning: {result}")
        return result
    
    def _generate_template_dialogue_fallback(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        discussion_context: str
    ) -> str:
        """生成上下文相关的模板化对话回应"""
        return self._generate_template_dialogue_fallback_NEW(patient_state, role, treatment_option, discussion_context)
    
    def _build_dialogue_response_prompt(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        discussion_context: str,
        knowledge_context: Dict[str, Any] = None,
        current_stance: RoleOpinion = None,
        dialogue_history: List[Dict] = None
    ) -> str:
        """构建对话回应提示词 - 强调自然性和个性化"""
        
        # 构建对话历史上下文
        history_context = ""
        if dialogue_history:
            recent_exchanges = dialogue_history 
            history_context = "\n上一轮对话:\n"
            for i, exchange in enumerate(recent_exchanges):
                history_context += f"上一轮{i+1}: {exchange.get('role', 'Unknown')} - {exchange.get('content', '')}...\n"
        
        logger.info(f"上一轮非自己的对话: {history_context}")
        
        # 构建立场信息
        stance_info = ""

        if current_stance:
            stance_value = current_stance.treatment_preferences.get(treatment_option.value, 0)
            if stance_value > 0.5:
                stance_info = "你对该治疗方案持积极态度"
            elif stance_value > 0:
                stance_info = "你对该治疗方案持谨慎支持态度"
            elif stance_value < -0.5:
                stance_info = "你对该治疗方案有较大担忧"
            else:
                stance_info = "你对该治疗方案持中性态度"
        logger.info(f"{role.value}当前立场Stance info: {stance_info}")
        prompt = f"""
作为{role.value}，请针对以下情况给出自然、专业的回应：

患者情况：
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

讨论的治疗方案: {treatment_option.value}

当前讨论背景: {discussion_context}

{stance_info}

{history_context}

请注意：
1. 回应要自然流畅，避免模板化表达
2. 体现你的专业角色特点
3. 考虑之前的对话内容，保持连贯性
4. 表达要有个人色彩，不要千篇一律
5. 长度控制在2-3句话，简洁有力
6. 如果有不同意见，要礼貌但坚定地表达
"""
        return prompt
    
    def _build_professional_reasoning_prompt(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None
    ) -> str:
        """构建专业推理提示词"""
        
        knowledge_info = ""
        if knowledge_context:
            knowledge_info = f"\n相关知识背景:\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}"
        
        prompt = f"""
作为{role.value}，请基于专业知识对以下治疗方案进行深入分析：

患者信息：
- 年龄: {patient_state.age}岁
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 生活质量: {patient_state.quality_of_life_score}
- 合并症: {', '.join(patient_state.comorbidities) if patient_state.comorbidities else '无'}

治疗方案: {treatment_option.value}

{knowledge_info}

请从你的专业角度提供：
1. 对该治疗方案的专业评估
2. 基于患者特征的适用性分析
3. 潜在风险和获益评估
4. 专业建议

请保持专业性和客观性，避免模板化表达：
"""
        return prompt
    
    def _get_role_system_prompt(self, role: RoleType) -> str:
        """获取角色特定的系统提示词"""
        
        role_prompts = {
            RoleType.ONCOLOGIST: "你是一位经验丰富的肿瘤科医生，专注于癌症治疗的疗效和安全性。你的回应应该基于循证医学，同时考虑患者的整体状况。说话风格专业但易懂，偶尔会引用临床经验。",
            
            RoleType.NURSE: "你是一位资深的肿瘤科护士，关注患者的日常护理和生活质量。你的回应应该实用、贴心，关注治疗的可行性和患者的舒适度。说话风格温和关怀，经常从护理角度思考问题。",
            
            RoleType.PSYCHOLOGIST: "你是一位临床心理学家，专注于癌症患者的心理健康。你的回应应该考虑患者的心理承受能力和情感需求。说话风格温暖支持，善于从心理角度分析问题。",
            
            RoleType.RADIOLOGIST: "你是一位放射科医生，专精于医学影像和放射治疗。你的回应应该基于影像学证据和放射治疗的技术特点。说话风格精确客观，经常引用影像学发现。",
            
            RoleType.PATIENT_ADVOCATE: "你是一位患者权益代表，致力于维护患者的最佳利益。你的回应应该平衡医疗建议和患者的价值观、偏好。说话风格坚定但富有同理心，经常站在患者角度思考。"
        }
        
        return role_prompts.get(role, f"你是一位{role.value}，请基于你的专业背景提供建议。")
    
    def _get_professional_system_prompt(self, role: RoleType) -> str:
        """获取专业推理的系统提示词"""
        
        return f"你是一位资深的{role.value}，请基于你的专业知识和临床经验，对医疗方案进行深入的专业分析。你的分析应该客观、全面，体现专业水准。"
    
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
                "timestamp": datetime.now().isoformat()
            }
    
    def _parse_timeline_events_response(self, response: str) -> List[Dict[str, Any]]:
        """解析时间线事件响应"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            events = json.loads(json_str)
            for event in events:
                event["generated_by"] = "llm"
                event["timestamp"] = datetime.now().isoformat()
            
            return events
        except:
            return [{
                "day": 1,
                "event_type": "解析错误",
                "description": response[:200],
                "severity": 1,
                "requires_intervention": False,
                "generated_by": "llm_error"
            }]
    
    def _generate_template_dialogue_fallback(
        self,
        patient_state: PatientState,
        role: RoleType,
        treatment_option: TreatmentOption,
        discussion_context: str
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
                TreatmentOption.WATCHFUL_WAITING: "密切观察策略在当前阶段是合理的选择。"
            },
            RoleType.SURGEON: {
                TreatmentOption.SURGERY: "从外科角度评估，患者的手术适应症和风险需要综合考虑。",
                TreatmentOption.CHEMOTHERAPY: "新辅助化疗或辅助化疗的时机选择对手术效果很重要。",
                TreatmentOption.RADIOTHERAPY: "放疗与手术的配合时机需要多学科团队讨论决定。"
            },
            RoleType.RADIOLOGIST: {
                TreatmentOption.RADIOTHERAPY: "基于影像学评估，放疗的靶区设计和剂量分布需要精确规划。",
                TreatmentOption.SURGERY: "影像学检查为手术方案的制定提供了重要的解剖学参考。"
            },
            RoleType.NURSE: {
                TreatmentOption.CHEMOTHERAPY: "化疗期间的护理管理和不良反应监测是治疗成功的关键。",
                TreatmentOption.SURGERY: "围手术期护理对患者的康复具有重要意义。",
                TreatmentOption.PALLIATIVE_CARE: "姑息护理能够显著提升患者的舒适度和生活质量。"
            },
            RoleType.PSYCHOLOGIST: {
                TreatmentOption.CHEMOTHERAPY: "化疗期间的心理支持有助于患者更好地配合治疗。",
                TreatmentOption.SURGERY: "术前心理准备和术后心理康复同样重要。",
                TreatmentOption.PALLIATIVE_CARE: "心理关怀在姑息治疗中发挥着不可替代的作用。"
            }
        }
        
        # 获取角色特定模板
        role_specific = role_templates.get(role, {})
        base_template = role_specific.get(
            treatment_option, 
            f"作为{role.value}，我认为{treatment_option.value}是值得考虑的治疗选择。"
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