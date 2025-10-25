"""
Disabled RAG stub
文件路径: src/knowledge/disabled_rag.py
功能: 提供一个禁用版的RAG接口以保持系统兼容性
"""
from typing import Dict, Any, Optional
from ..core.data_models import PatientState, TreatmentOption

class DisabledRAG:
    """一个禁用版的 RAG 系统，返回空或最小信息结构。
    目的：当不需要RAG时，仍然让依赖 rag_system 的模块安全运行。
    """
    def __init__(self):
        # 标记禁用状态，便于日志或UI提示
        self.disabled = True

    def retrieve_relevant_knowledge(
        self,
        patient_state: PatientState,
        query_type: str,
        treatment_focus: Optional[TreatmentOption] = None,
    ) -> Dict[str, Any]:
        """返回一个结构化但空的知识包，保持下游调用不出错"""
        return {
            "guidelines": [],
            "similar_cases": [],
            "contraindications": [],
            "evidence_level": "N/A",
            "success_rates": {},
            "side_effects": {},
            "comorbidity_considerations": {},
            "drug_interactions": [],
            "priority": "rag_disabled"
        }

    def search_knowledge(self, query: str, max_results: int = 5):
        """禁用状态下返回空列表"""
        return []