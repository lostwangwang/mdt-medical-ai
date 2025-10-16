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

from ..core.data_models import PatientState, TreatmentOption, RoleType

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
        """设置LLM客户端"""
        try:
            if self.config.api_key:
                openai.api_key = self.config.api_key
            if self.config.base_url:
                openai.base_url = self.config.base_url
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            logger.info(f"LLM client initialized with model: {self.config.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def generate_treatment_reasoning(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None
    ) -> str:
        """生成治疗推理"""
        
        prompt = self._build_treatment_reasoning_prompt(
            patient_state, role, treatment_option, knowledge_context
        )
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": f"你是一位专业的{role.value}，请基于患者信息和角色专业性提供治疗推理。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content.strip()
            else:
                # 降级到模板化回复
                return self._generate_template_reasoning(patient_state, role, treatment_option)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_template_reasoning(patient_state, role, treatment_option)
    
    def generate_treatment_plan(
        self,
        patient_state: PatientState,
        memory_context: Dict[str, Any],
        knowledge_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """生成完整的治疗方案"""
        
        prompt = self._build_treatment_plan_prompt(
            patient_state, memory_context, knowledge_context
        )
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "你是一位资深的肿瘤科医生，请基于患者的完整病史和当前状态制定综合治疗方案。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # 解析LLM响应为结构化数据
                return self._parse_treatment_plan_response(response.choices[0].message.content)
            else:
                return self._generate_template_treatment_plan(patient_state)
        except Exception as e:
            logger.error(f"Treatment plan generation failed: {e}")
            return self._generate_template_treatment_plan(patient_state)
    
    def generate_patient_timeline_events(
        self,
        patient_state: PatientState,
        memory_context: Dict[str, Any],
        days_ahead: int = 30
    ) -> List[Dict[str, Any]]:
        """生成患者时间线事件"""
        
        prompt = self._build_timeline_events_prompt(
            patient_state, memory_context, days_ahead
        )
        
        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "你是一位医疗数据分析师，请基于患者状态生成合理的医疗事件时间线。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                return self._parse_timeline_events_response(response.choices[0].message.content)
            else:
                return self._generate_template_timeline_events(patient_state, days_ahead)
        except Exception as e:
            logger.error(f"Timeline events generation failed: {e}")
            return self._generate_template_timeline_events(patient_state, days_ahead)
    
    def _build_treatment_reasoning_prompt(
        self, 
        patient_state: PatientState, 
        role: RoleType,
        treatment_option: TreatmentOption,
        knowledge_context: Dict[str, Any] = None
    ) -> str:
        """构建治疗推理提示词"""
        
        role_descriptions = {
            RoleType.ONCOLOGIST: "肿瘤科医生，关注治疗效果和生存率",
            RoleType.NURSE: "护士，关注护理可行性和患者舒适度",
            RoleType.PSYCHOLOGIST: "心理医生，关注患者心理健康",
            RoleType.RADIOLOGIST: "放射科医生，关注影像学表现和放射治疗",
            RoleType.PATIENT_ADVOCATE: "患者代表，关注患者权益、自主选择和生活质量"
        }
        
        prompt = f"""
患者信息：
- 患者ID: {patient_state.patient_id}
- 诊断: {patient_state.diagnosis}
- 分期: {patient_state.stage}
- 年龄: {patient_state.age}
- 心理状态: {patient_state.psychological_status}
- 生活质量评分: {patient_state.quality_of_life_score}

角色身份: {role_descriptions.get(role, role.value)}

治疗选项: {treatment_option.value}

请从{role.value}的专业角度，为该患者的{treatment_option.value}治疗提供详细的推理分析，包括：
1. 支持该治疗的理由
2. 可能的风险和注意事项
3. 与患者具体情况的匹配度

请用专业但易懂的语言回答，控制在200字以内。
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
        
        templates = {
            RoleType.ONCOLOGIST: {
                TreatmentOption.SURGERY: f"基于{patient_state.diagnosis}的临床特征，手术切除是标准治疗选择，可提供最佳的局部控制。",
                TreatmentOption.CHEMOTHERAPY: f"对于{patient_state.stage}期患者，化疗可有效控制全身微转移病灶。"
            },
            RoleType.NURSE: {
                TreatmentOption.SURGERY: "患者的体能状态适合手术，术后护理计划已制定完善。",
                TreatmentOption.CHEMOTHERAPY: "化疗期间需密切监测副作用，确保患者安全和依从性。"
            }
        }
        
        role_templates = templates.get(role, {})
        return role_templates.get(treatment_option, f"{role.value}建议考虑{treatment_option.value}治疗。")
    
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