"""
MDT医疗智能体系统 API服务器
文件路径: src/api/main.py
作者: Tianyu (系统集成)
功能: 提供RESTful API接口，连接前端和后端系统
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 自定义JSON编码器处理datetime对象
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from src.core.data_models import PatientState, TreatmentOption
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.integration.agent_rl_coordinator import AgentRLCoordinator
from src.integration.workflow_manager import IntegratedWorkflowManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.utils.data_generator import IntelligentDataGenerator, DataGenerationConfig
from src.utils.system_optimizer import get_system_optimizer, optimized_function

# 初始化系统优化器
system_optimizer = get_system_optimizer()

# 使用优化的日志系统
logger = system_optimizer.get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="MDT医疗智能体系统API",
    description="多智能体医疗决策支持系统的RESTful API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局系统实例
dialogue_manager = None
agent_rl_coordinator = None
workflow_manager = None
data_generator = None

# 请求/响应模型
class ChatMessage(BaseModel):
    message: str
    patient_context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    response: str
    confidence: float
    reasoning: str

class ConsensusResult(BaseModel):
    consensus_text: str
    confidence_level: float
    recommended_treatment: str
    expert_opinions: List[AgentResponse]

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]

class DataGenerationRequest(BaseModel):
    num_patients: int = 10
    include_timeline: bool = True
    timeline_days: int = 30
    output_format: str = "json"  # json, csv, excel
    
class DataGenerationResponse(BaseModel):
    status: str
    message: str
    num_generated: int
    file_path: Optional[str] = None
    data_summary: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化系统组件"""
    global dialogue_manager, agent_rl_coordinator, workflow_manager, data_generator
    
    # 启动系统优化器
    logger.info("启动系统优化器...")
    system_optimizer.initialize()
    
    logger.info("初始化MDT系统组件...")
    
    try:
        # 初始化RAG系统
        rag_system = MedicalKnowledgeRAG()
        
        # 初始化对话管理器
        dialogue_manager = MultiAgentDialogueManager(rag_system)
        
        # 初始化智能体-RL协调器
        agent_rl_coordinator = AgentRLCoordinator(rag_system)
        
        # 初始化工作流管理器
        workflow_manager = IntegratedWorkflowManager()
        
        # 初始化数据生成器
        data_config = DataGenerationConfig(
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.7,
            llm_max_tokens=2000
        )
        data_generator = IntelligentDataGenerator(config=data_config)
        
        logger.info("系统组件初始化完成")
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径，返回API信息"""
    return {
        "message": "MDT医疗智能体系统API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """健康检查端点"""
    components_status = {
        "dialogue_manager": "healthy" if dialogue_manager else "not_initialized",
        "agent_rl_coordinator": "healthy" if agent_rl_coordinator else "not_initialized",
        "workflow_manager": "healthy" if workflow_manager else "not_initialized",
        "data_generator": "healthy" if data_generator else "not_initialized"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components_status.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components_status
    )

@app.post("/chat/agent/{agent_id}", response_model=AgentResponse)
@optimized_function
async def chat_with_agent(agent_id: str, message: ChatMessage):
    """与特定智能体对话"""
    if not dialogue_manager:
        raise HTTPException(status_code=500, detail="对话管理器未初始化")
    
    try:
        # 创建患者状态（如果提供了上下文）
        patient_state = None
        if message.patient_context:
            patient_state = PatientState(
                patient_id=message.patient_context.get("patient_id", "default"),
                age=message.patient_context.get("age", 50),
                diagnosis=message.patient_context.get("diagnosis", "unknown"),
                stage=message.patient_context.get("stage", "unknown"),
                lab_results=message.patient_context.get("lab_results", {}),
                vital_signs=message.patient_context.get("vital_signs", {}),
                symptoms=message.patient_context.get("symptoms", []),
                comorbidities=message.patient_context.get("comorbidities", []),
                psychological_status=message.patient_context.get("psychological_status", "stable"),
                quality_of_life_score=message.patient_context.get("quality_of_life_score", 0.7),
                timestamp=datetime.now()
            )
        
        # 获取智能体响应
        agent_response = dialogue_manager.get_agent_opinion(agent_id, message.message, patient_state)
        
        return AgentResponse(
            agent_id=agent_id,
            agent_name=agent_response.get("agent_name", agent_id),
            response=agent_response.get("response", ""),
            confidence=agent_response.get("confidence", 0.5),
            reasoning=agent_response.get("reasoning", "")
        )
        
    except Exception as e:
        logger.error(f"智能体对话错误: {e}")
        raise HTTPException(status_code=500, detail=f"智能体对话失败: {str(e)}")

@app.post("/consensus/generate", response_model=ConsensusResult)
@optimized_function
async def generate_consensus(expert_opinions: List[AgentResponse]):
    """基于专家意见生成共识"""
    if not dialogue_manager:
        raise HTTPException(status_code=500, detail="对话管理器未初始化")
    
    try:
        # 转换专家意见格式
        opinions_data = []
        for opinion in expert_opinions:
            opinions_data.append({
                "agent_id": opinion.agent_id,
                "response": opinion.response,
                "confidence": opinion.confidence,
                "reasoning": opinion.reasoning
            })
        
        # 生成共识
        consensus_result = dialogue_manager.generate_consensus_from_opinions(opinions_data)
        
        return ConsensusResult(
            consensus_text=consensus_result.get("consensus_text", ""),
            confidence_level=consensus_result.get("confidence_level", 0.5),
            recommended_treatment=consensus_result.get("recommended_treatment", ""),
            expert_opinions=expert_opinions
        )
        
    except Exception as e:
        logger.error(f"共识生成错误: {e}")
        raise HTTPException(status_code=500, detail=f"共识生成失败: {str(e)}")

@app.post("/mdt/discussion")
@optimized_function
async def start_mdt_discussion(message: ChatMessage):
    """启动MDT讨论"""
    if not dialogue_manager or not agent_rl_coordinator:
        raise HTTPException(status_code=500, detail="系统组件未初始化")
    
    try:
        # 创建患者状态
        patient_state = PatientState(
            patient_id="mdt_session",
            age=message.patient_context.get("age", 50) if message.patient_context else 50,
            diagnosis=message.patient_context.get("diagnosis", "unknown") if message.patient_context else "unknown",
            stage=message.patient_context.get("stage", "unknown") if message.patient_context else "unknown",
            lab_results=message.patient_context.get("lab_results", {}) if message.patient_context else {},
            vital_signs=message.patient_context.get("vital_signs", {}) if message.patient_context else {},
            symptoms=message.patient_context.get("symptoms", []) if message.patient_context else [],
            comorbidities=message.patient_context.get("comorbidities", []) if message.patient_context else [],
            psychological_status=message.patient_context.get("psychological_status", "stable") if message.patient_context else "stable",
            quality_of_life_score=message.patient_context.get("quality_of_life_score", 0.7) if message.patient_context else 0.7,
            timestamp=datetime.now()
        )
        
        # 进行MDT讨论
        discussion_result = dialogue_manager.conduct_mdt_discussion(patient_state)
        
        # 使用智能体-RL协调器进行决策融合
        coordination_result = agent_rl_coordinator.coordinate_decision(
            patient_state=patient_state,
            agent_opinions=discussion_result.role_opinions,
            mode="COLLABORATIVE"
        )
        
        return {
            "discussion_result": {
                "consensus_achieved": discussion_result.convergence_achieved,
                "total_rounds": discussion_result.total_rounds,
                "recommended_treatment": coordination_result.final_decision,
                "confidence_level": coordination_result.confidence_score
            },
            "expert_opinions": [
                {
                    "agent_id": opinion.role.value,
                    "response": opinion.opinion_text,
                    "confidence": opinion.confidence_score,
                    "reasoning": opinion.reasoning
                }
                for opinion in discussion_result.role_opinions
            ]
        }
        
    except Exception as e:
        logger.error(f"MDT讨论错误: {e}")
        raise HTTPException(status_code=500, detail=f"MDT讨论失败: {str(e)}")

@app.post("/data/generate", response_model=DataGenerationResponse)
async def generate_treatment_data(request: DataGenerationRequest):
    """生成治疗方案数据集"""
    if not data_generator:
        raise HTTPException(status_code=500, detail="数据生成器未初始化")
    
    try:
        logger.info(f"开始生成数据: {request.num_patients}个患者, 时间线{request.timeline_days}天")
        
        # 生成患者治疗数据集
        dataset = data_generator.generate_patient_treatment_dataset(
            num_patients=request.num_patients
        )
        
        # 导出数据
        output_dir = "generated_data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"treatment_data_{timestamp}"
        
        if request.output_format.lower() == "csv":
            file_path = data_generator.export_to_csv(dataset, f"{output_dir}/{filename}")
        elif request.output_format.lower() == "excel":
            file_path = data_generator.export_to_excel(dataset, f"{output_dir}/{filename}")
        else:  # json
            file_path = data_generator.export_to_json(dataset, f"{output_dir}/{filename}")
        
        # 生成数据摘要
        summary = data_generator.generate_dataset_summary(dataset)
        
        logger.info(f"数据生成完成: {len(dataset)}个记录, 保存至 {file_path}")
        
        response_data = {
            "status": "success",
            "message": f"成功生成 {len(dataset)} 条治疗方案数据",
            "num_generated": len(dataset),
            "file_path": file_path,
            "data_summary": summary
        }
        
        # 使用自定义编码器处理datetime对象
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=DateTimeEncoder))
        )
        
    except Exception as e:
        logger.error(f"数据生成错误: {e}")
        raise HTTPException(status_code=500, detail=f"数据生成失败: {str(e)}")

@app.post("/data/generate/enhanced-plan")
async def generate_enhanced_treatment_plan(message: ChatMessage):
    """生成增强的治疗方案（结合memory、RAG、MDT和LLM）"""
    if not data_generator:
        raise HTTPException(status_code=500, detail="数据生成器未初始化")
    
    try:
        # 创建患者状态
        patient_state = PatientState(
            patient_id=message.patient_context.get("patient_id", "enhanced_plan") if message.patient_context else "enhanced_plan",
            age=message.patient_context.get("age", 50) if message.patient_context else 50,
            diagnosis=message.patient_context.get("diagnosis", "breast_cancer") if message.patient_context else "breast_cancer",
            stage=message.patient_context.get("stage", "II") if message.patient_context else "II",
            lab_results=message.patient_context.get("lab_results", {}) if message.patient_context else {},
            vital_signs=message.patient_context.get("vital_signs", {}) if message.patient_context else {},
            symptoms=message.patient_context.get("symptoms", []) if message.patient_context else [],
            comorbidities=message.patient_context.get("comorbidities", []) if message.patient_context else [],
            psychological_status=message.patient_context.get("psychological_status", "stable") if message.patient_context else "stable",
            quality_of_life_score=message.patient_context.get("quality_of_life_score", 0.7) if message.patient_context else 0.7,
            timestamp=datetime.now()
        )
        
        # 生成增强治疗方案
        enhanced_plan = data_generator.generate_enhanced_treatment_plan(
            patient_id=patient_state.patient_id,
            use_memory_context=True,
            use_llm_enhancement=True
        )
        
        response_data = {
            "status": "success",
            "patient_id": patient_state.patient_id,
            "enhanced_plan": enhanced_plan,
            "timestamp": datetime.now()
        }
        
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=DateTimeEncoder))
        )
        
    except Exception as e:
        logger.error(f"增强治疗方案生成错误: {e}")
        raise HTTPException(status_code=500, detail=f"增强治疗方案生成失败: {str(e)}")

@app.post("/data/generate/timeline")
async def generate_patient_timeline(patient_id: str, days: int = 30):
    """生成患者时间线数据"""
    if not data_generator:
        raise HTTPException(status_code=500, detail="数据生成器未初始化")
    
    try:
        # 生成患者时间线模拟
        timeline = data_generator.generate_patient_timeline_simulation(
            patient_id=patient_id,
            simulation_days=days
        )
        
        response_data = {
            "status": "success",
            "patient_id": patient_id,
            "timeline_days": days,
            "timeline_simulation": timeline
        }
        
        return JSONResponse(
            content=json.loads(json.dumps(response_data, cls=DateTimeEncoder))
        )
        
    except Exception as e:
        logger.error(f"患者时间线生成错误: {e}")
        raise HTTPException(status_code=500, detail=f"患者时间线生成失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)