"""
LangGraph MDT 工作流
文件路径: src/workflow/langgraph_mdt_workflow.py
作者: 改造方案示例
功能: 使用 LangGraph 重构 MDT 多智能体协作系统
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import logging
from dataclasses import dataclass

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..core.data_models import PatientState, TreatmentOption, RoleType
from ..consensus.role_agents import RoleAgent
from ..knowledge.enhanced_faiss_integration import EnhancedFAISSManager
from ..utils.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class MDTState(TypedDict):
    """MDT 工作流状态"""
    # 患者信息
    patient_state: PatientState
    patient_id: str
    
    # 消息历史
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 工作流状态
    current_step: str
    iteration_count: int
    max_iterations: int
    
    # 角色和智能体
    selected_roles: List[RoleType]
    active_agents: Dict[str, RoleAgent]
    
    # 知识检索
    similar_patients: List[Dict[str, Any]]
    knowledge_context: Dict[str, Any]
    
    # 讨论和共识
    role_opinions: Dict[str, Dict[str, Any]]
    consensus_matrix: Dict[str, float]
    convergence_achieved: bool
    
    # 最终结果
    treatment_recommendations: Dict[str, Any]
    final_decision: Optional[Dict[str, Any]]
    
    # 元数据
    workflow_metadata: Dict[str, Any]


@dataclass
class LangGraphMDTWorkflow:
    """LangGraph MDT 工作流管理器"""
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        faiss_manager: EnhancedFAISSManager,
        max_iterations: int = 5,
        convergence_threshold: float = 0.8
    ):
        self.llm_interface = llm_interface
        self.faiss_manager = faiss_manager
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # 构建工作流图
        self.workflow = self._build_workflow_graph()
        
    def _build_workflow_graph(self) -> StateGraph:
        """构建 LangGraph 工作流图"""
        
        # 创建状态图
        workflow = StateGraph(MDTState)
        
        # 添加节点
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("role_selector", self._select_optimal_roles)
        workflow.add_node("knowledge_retrieval", self._retrieve_knowledge)
        workflow.add_node("generate_opinions", self._generate_role_opinions)
        workflow.add_node("consensus_check", self._check_consensus)
        workflow.add_node("mdt_discussion", self._conduct_mdt_round)
        workflow.add_node("finalize_decision", self._finalize_treatment_decision)
        
        # 设置入口点
        workflow.set_entry_point("initialize")
        
        # 添加边和条件路由
        workflow.add_edge("initialize", "role_selector")
        workflow.add_edge("role_selector", "knowledge_retrieval")
        workflow.add_edge("knowledge_retrieval", "generate_opinions")
        workflow.add_edge("generate_opinions", "consensus_check")
        
        # 条件边：根据共识状态决定下一步
        workflow.add_conditional_edges(
            "consensus_check",
            self._should_continue_discussion,
            {
                "continue": "mdt_discussion",
                "finalize": "finalize_decision",
                "max_iterations": "finalize_decision"
            }
        )
        
        workflow.add_edge("mdt_discussion", "consensus_check")
        workflow.add_edge("finalize_decision", END)
        
        return workflow.compile()
    
    def _initialize_workflow(self, state: MDTState) -> MDTState:
        """初始化工作流"""
        logger.info(f"Initializing MDT workflow for patient {state['patient_id']}")
        
        state["current_step"] = "initialization"
        state["iteration_count"] = 0
        state["max_iterations"] = self.max_iterations
        state["messages"] = [
            HumanMessage(content=f"开始为患者 {state['patient_id']} 进行 MDT 讨论")
        ]
        state["workflow_metadata"] = {
            "start_time": datetime.now().isoformat(),
            "workflow_version": "langgraph_v1.0"
        }
        
        return state
    
    def _select_optimal_roles(self, state: MDTState) -> MDTState:
        """选择最优角色组合"""
        logger.info("Selecting optimal roles for MDT discussion")
        
        patient_state = state["patient_state"]
        
        # 基于患者状态智能选择角色
        selected_roles = []
        
        # 核心角色（总是包含）
        selected_roles.extend([RoleType.ONCOLOGIST, RoleType.SURGEON])
        
        # 基于病情选择额外角色
        if patient_state.psychological_status in ["anxious", "depressed"]:
            selected_roles.append(RoleType.PSYCHOLOGIST)
            
        if len(patient_state.comorbidities) > 2:
            selected_roles.append(RoleType.NURSE)
            
        if patient_state.quality_of_life_score < 0.6:
            selected_roles.extend([RoleType.PATIENT_ADVOCATE, RoleType.NUTRITIONIST])
            
        # 影像相关
        if "imaging" in patient_state.diagnosis.lower():
            selected_roles.append(RoleType.RADIOLOGIST)
            
        state["selected_roles"] = list(set(selected_roles))
        state["active_agents"] = {
            role.value: RoleAgent(role, self.llm_interface) 
            for role in selected_roles
        }
        
        state["messages"].append(
            AIMessage(content=f"已选择角色: {[role.value for role in selected_roles]}")
        )
        
        return state
    
    def _retrieve_knowledge(self, state: MDTState) -> MDTState:
        """检索相关知识"""
        logger.info("Retrieving relevant medical knowledge")
        
        patient_state = state["patient_state"]
        
        # FAISS 检索相似患者
        try:
            query_text = f"{patient_state.diagnosis} {patient_state.stage} age:{patient_state.age}"
            similar_patients = self.faiss_manager.search_similar_patients(
                query_text, top_k=5
            )
            state["similar_patients"] = [
                {
                    "patient_id": result.metadata.get("patient_id"),
                    "similarity": result.score,
                    "treatment": result.metadata.get("treatment"),
                    "outcome": result.metadata.get("outcome")
                }
                for result in similar_patients
            ]
        except Exception as e:
            logger.warning(f"FAISS retrieval failed: {e}")
            state["similar_patients"] = []
        
        # 构建知识上下文
        state["knowledge_context"] = {
            "similar_cases_count": len(state["similar_patients"]),
            "patient_complexity": len(patient_state.comorbidities),
            "urgency_level": self._assess_urgency(patient_state)
        }
        
        return state
    
    async def run_workflow(self, patient_state: PatientState) -> Dict[str, Any]:
        """运行完整的 MDT 工作流"""
        
        # 初始化状态
        initial_state = MDTState(
            patient_state=patient_state,
            patient_id=patient_state.patient_id,
            messages=[],
            current_step="",
            iteration_count=0,
            max_iterations=self.max_iterations,
            selected_roles=[],
            active_agents={},
            similar_patients=[],
            knowledge_context={},
            role_opinions={},
            consensus_matrix={},
            convergence_achieved=False,
            treatment_recommendations={},
            final_decision=None,
            workflow_metadata={}
        )
        
        # 执行工作流
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": True,
                "final_decision": final_state["final_decision"],
                "workflow_metadata": final_state["workflow_metadata"],
                "consensus_matrix": final_state["consensus_matrix"],
                "role_opinions": final_state["role_opinions"],
                "messages": [msg.content for msg in final_state["messages"]]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_state": initial_state
            }