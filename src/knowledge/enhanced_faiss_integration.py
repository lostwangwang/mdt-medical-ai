"""
增强的FAISS向量数据库集成模块
文件路径: src/knowledge/enhanced_faiss_integration.py
作者: 姚刚
功能: 提供完整的向量数据库查询、检索和知识管理功能
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu langchain-community")

from ..core.data_models import PatientState, TreatmentOption

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    
    content: str
    metadata: Dict[str, Any]
    score: float
    patient_id: Optional[str] = None
    document_type: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class KnowledgeDocument:
    """知识文档"""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    document_type: str = "clinical"
    created_at: datetime = field(default_factory=datetime.now)


class EnhancedFAISSManager:
    """增强的FAISS向量数据库管理器"""
    
    def __init__(
        self,
        db_path: str = "clinical_memory_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        dimension: int = 384
    ):
        """初始化FAISS管理器"""
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Please install required packages.")
        
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.device = device
        self.dimension = dimension
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 数据库实例
        self.vector_db: Optional[FAISS] = None
        self.document_cache: Dict[str, KnowledgeDocument] = {}
        
        # 线程池用于异步操作
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 初始化数据库
        self._initialize_database()
        
        logger.info(f"Enhanced FAISS Manager initialized with model: {embedding_model}")
    
    def _initialize_database(self):
        """初始化数据库"""
        
        try:
            if self.db_path.exists():
                # 加载现有数据库
                self.vector_db = FAISS.load_local(
                    str(self.db_path), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS database from {self.db_path}")
                logger.info(f"Database contains {self.vector_db.index.ntotal} documents")
            else:
                # 创建新数据库
                self._create_empty_database()
                logger.info(f"Created new FAISS database at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS database: {e}")
            self._create_empty_database()
    
    def _create_empty_database(self):
        """创建空数据库"""
        
        # 创建一个临时文档来初始化数据库
        temp_text = "Temporary initialization document"
        temp_metadata = {"type": "initialization", "temporary": True}
        
        self.vector_db = FAISS.from_texts(
            [temp_text], 
            self.embeddings, 
            metadatas=[temp_metadata]
        )
        
        # 确保目录存在
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(self.db_path))
    
    def add_patient_data(
        self, 
        patient_id: str, 
        patient_data: Dict[str, Any],
        document_type: str = "clinical_memory"
    ) -> str:
        """添加患者数据到向量数据库"""
        
        try:
            # 准备文档内容
            content = self._prepare_patient_content(patient_data)
            
            # 准备元数据
            metadata = {
                "patient_id": patient_id,
                "document_type": document_type,
                "timestamp": datetime.now().isoformat(),
                "diagnosis": patient_data.get("diagnosis", ""),
                "age": patient_data.get("age", 0),
                "stage": patient_data.get("stage", ""),
                "comorbidities": patient_data.get("comorbidities", [])
            }
            
            # 生成文档ID
            doc_id = f"{document_type}_{patient_id}_{int(datetime.now().timestamp())}"
            
            # 添加到向量数据库
            self.vector_db.add_texts(
                texts=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            # 保存数据库
            self.vector_db.save_local(str(self.db_path))
            
            # 缓存文档
            self.document_cache[doc_id] = KnowledgeDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                document_type=document_type
            )
            
            logger.info(f"Added patient data for {patient_id} to FAISS database")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add patient data to FAISS: {e}")
            raise
    
    def _prepare_patient_content(self, patient_data: Dict[str, Any]) -> str:
        """准备患者数据内容用于向量化"""
        
        content_parts = []
        
        # 基本信息
        if "diagnosis" in patient_data:
            content_parts.append(f"诊断: {patient_data['diagnosis']}")
        
        if "age" in patient_data:
            content_parts.append(f"年龄: {patient_data['age']}岁")
        
        if "stage" in patient_data:
            content_parts.append(f"分期: {patient_data['stage']}")
        
        # 症状
        if "symptoms" in patient_data and patient_data["symptoms"]:
            symptoms_text = ", ".join(patient_data["symptoms"])
            content_parts.append(f"症状: {symptoms_text}")
        
        # 合并症
        if "comorbidities" in patient_data and patient_data["comorbidities"]:
            comorbidities_text = ", ".join(patient_data["comorbidities"])
            content_parts.append(f"合并症: {comorbidities_text}")
        
        # 实验室结果
        if "lab_results" in patient_data:
            lab_parts = []
            for key, value in patient_data["lab_results"].items():
                lab_parts.append(f"{key}: {value}")
            if lab_parts:
                content_parts.append(f"实验室结果: {', '.join(lab_parts)}")
        
        # 生命体征
        if "vital_signs" in patient_data:
            vital_parts = []
            for key, value in patient_data["vital_signs"].items():
                vital_parts.append(f"{key}: {value}")
            if vital_parts:
                content_parts.append(f"生命体征: {', '.join(vital_parts)}")
        
        # 心理状态
        if "psychological_status" in patient_data:
            content_parts.append(f"心理状态: {patient_data['psychological_status']}")
        
        # 生活质量评分
        if "quality_of_life_score" in patient_data:
            content_parts.append(f"生活质量评分: {patient_data['quality_of_life_score']}")
        
        return ". ".join(content_parts)
    
    def search_similar_patients(
        self, 
        query_patient: Union[PatientState, Dict[str, Any]], 
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[SearchResult]:
        """搜索相似患者"""
        
        try:
            # 准备查询内容
            if isinstance(query_patient, PatientState):
                query_content = self._prepare_patient_state_content(query_patient)
            else:
                query_content = self._prepare_patient_content(query_patient)
            
            # 执行相似性搜索
            docs_and_scores = self.vector_db.similarity_search_with_score(
                query_content, k=k
            )
            
            # 处理结果
            results = []
            for doc, score in docs_and_scores:
                # 过滤低分结果
                if score < score_threshold:
                    continue
                
                result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    patient_id=doc.metadata.get("patient_id"),
                    document_type=doc.metadata.get("document_type"),
                    timestamp=self._parse_timestamp(doc.metadata.get("timestamp"))
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar patients")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar patients: {e}")
            return []
    
    def _prepare_patient_state_content(self, patient_state: PatientState) -> str:
        """准备PatientState对象的内容"""
        
        content_parts = []
        
        content_parts.append(f"诊断: {patient_state.diagnosis}")
        content_parts.append(f"年龄: {patient_state.age}岁")
        
        if hasattr(patient_state, 'stage') and patient_state.stage:
            content_parts.append(f"分期: {patient_state.stage}")
        
        if patient_state.symptoms:
            symptoms_text = ", ".join(patient_state.symptoms)
            content_parts.append(f"症状: {symptoms_text}")
        
        if patient_state.comorbidities:
            comorbidities_text = ", ".join(patient_state.comorbidities)
            content_parts.append(f"合并症: {comorbidities_text}")
        
        if patient_state.lab_results:
            lab_parts = []
            for key, value in patient_state.lab_results.items():
                lab_parts.append(f"{key}: {value}")
            content_parts.append(f"实验室结果: {', '.join(lab_parts)}")
        
        if patient_state.vital_signs:
            vital_parts = []
            for key, value in patient_state.vital_signs.items():
                vital_parts.append(f"{key}: {value}")
            content_parts.append(f"生命体征: {', '.join(vital_parts)}")
        
        content_parts.append(f"心理状态: {patient_state.psychological_status}")
        content_parts.append(f"生活质量评分: {patient_state.quality_of_life_score}")
        
        return ". ".join(content_parts)
    
    def search_by_condition(
        self, 
        condition: str, 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """根据疾病条件搜索"""
        
        try:
            # 构建查询
            query = f"诊断: {condition}"
            
            # 执行搜索
            docs_and_scores = self.vector_db.similarity_search_with_score(query, k=k)
            
            # 处理结果
            results = []
            for doc, score in docs_and_scores:
                # 应用过滤器
                if filters and not self._apply_filters(doc.metadata, filters):
                    continue
                
                result = SearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    patient_id=doc.metadata.get("patient_id"),
                    document_type=doc.metadata.get("document_type"),
                    timestamp=self._parse_timestamp(doc.metadata.get("timestamp"))
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} patients with condition: {condition}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by condition: {e}")
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """应用搜索过滤器"""
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict):
                # 范围过滤
                if "min" in value and metadata[key] < value["min"]:
                    return False
                if "max" in value and metadata[key] > value["max"]:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_patient_by_id(self, patient_id: str) -> Optional[SearchResult]:
        """根据患者ID获取患者数据"""
        
        try:
            # 搜索特定患者
            results = self.search_by_condition(
                condition="",  # 空条件
                k=100,  # 获取更多结果
                filters={"patient_id": patient_id}
            )
            
            # 返回最新的记录
            if results:
                return max(results, key=lambda x: x.timestamp or datetime.min)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get patient by ID: {e}")
            return None
    
    def get_treatment_recommendations(
        self, 
        patient_state: PatientState,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """获取治疗建议"""
        
        try:
            # 搜索相似患者
            similar_patients = self.search_similar_patients(patient_state, k=k)
            
            # 分析治疗模式
            treatment_analysis = self._analyze_treatment_patterns(similar_patients)
            
            return treatment_analysis
            
        except Exception as e:
            logger.error(f"Failed to get treatment recommendations: {e}")
            return []
    
    def _analyze_treatment_patterns(self, similar_patients: List[SearchResult]) -> List[Dict[str, Any]]:
        """分析治疗模式"""
        
        treatment_counts = {}
        total_patients = len(similar_patients)
        
        for patient in similar_patients:
            # 从内容中提取治疗信息（简化实现）
            content = patient.content.lower()
            
            # 检测常见治疗方式
            treatments = []
            if "手术" in content or "surgery" in content:
                treatments.append("surgery")
            if "化疗" in content or "chemotherapy" in content:
                treatments.append("chemotherapy")
            if "放疗" in content or "radiotherapy" in content:
                treatments.append("radiotherapy")
            if "靶向" in content or "targeted" in content:
                treatments.append("targeted_therapy")
            if "免疫" in content or "immunotherapy" in content:
                treatments.append("immunotherapy")
            
            for treatment in treatments:
                treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
        
        # 计算推荐度
        recommendations = []
        for treatment, count in treatment_counts.items():
            confidence = count / total_patients if total_patients > 0 else 0
            recommendations.append({
                "treatment": treatment,
                "confidence": confidence,
                "supporting_cases": count,
                "total_cases": total_patients
            })
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """解析时间戳"""
        
        if not timestamp_str:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return None
    
    async def async_search_similar_patients(
        self, 
        query_patient: Union[PatientState, Dict[str, Any]], 
        k: int = 5
    ) -> List[SearchResult]:
        """异步搜索相似患者"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.search_similar_patients, 
            query_patient, 
            k
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        
        try:
            stats = {
                "total_documents": self.vector_db.index.ntotal,
                "vector_dimension": self.vector_db.index.d,
                "database_path": str(self.db_path),
                "embedding_model": self.embedding_model_name,
                "device": self.device
            }
            
            # 统计文档类型
            doc_types = {}
            patient_count = 0
            
            # 这里需要遍历所有文档来统计，简化实现
            if hasattr(self.vector_db, 'docstore') and hasattr(self.vector_db.docstore, '_dict'):
                for doc in self.vector_db.docstore._dict.values():
                    if hasattr(doc, 'metadata'):
                        doc_type = doc.metadata.get("document_type", "unknown")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                        
                        if doc.metadata.get("patient_id"):
                            patient_count += 1
            
            stats["document_types"] = doc_types
            stats["unique_patients"] = patient_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def export_database(self, output_path: str, format: str = "json"):
        """导出数据库"""
        
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                export_data = {
                    "metadata": self.get_database_stats(),
                    "documents": []
                }
                
                # 导出所有文档
                if hasattr(self.vector_db, 'docstore') and hasattr(self.vector_db.docstore, '_dict'):
                    for doc_id, doc in self.vector_db.docstore._dict.items():
                        export_data["documents"].append({
                            "id": doc_id,
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        })
                
                # 保存到文件
                export_file = output_path / f"faiss_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Database exported to {export_file}")
                return str(export_file)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("FAISS Manager cleaned up")


# 使用示例和测试函数
async def demonstrate_enhanced_faiss():
    """演示增强FAISS功能"""
    
    logger.info("=== 增强FAISS功能演示 ===")
    
    # 创建FAISS管理器
    faiss_manager = EnhancedFAISSManager(
        db_path="demo_clinical_memory_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 模拟患者数据
    patient_data_1 = {
        "patient_id": "demo_001",
        "diagnosis": "breast_cancer",
        "age": 55,
        "stage": "II",
        "symptoms": ["fatigue", "pain"],
        "comorbidities": ["diabetes", "hypertension"],
        "lab_results": {"hemoglobin": 10.5, "creatinine": 1.2},
        "vital_signs": {"bp": "140/90", "hr": 80},
        "psychological_status": "anxious",
        "quality_of_life_score": 0.6
    }
    
    patient_data_2 = {
        "patient_id": "demo_002",
        "diagnosis": "breast_cancer",
        "age": 62,
        "stage": "III",
        "symptoms": ["fatigue", "weight_loss"],
        "comorbidities": ["diabetes"],
        "lab_results": {"hemoglobin": 9.8, "creatinine": 1.5},
        "vital_signs": {"bp": "150/95", "hr": 85},
        "psychological_status": "depressed",
        "quality_of_life_score": 0.4
    }
    
    # 添加患者数据
    doc_id_1 = faiss_manager.add_patient_data("demo_001", patient_data_1)
    doc_id_2 = faiss_manager.add_patient_data("demo_002", patient_data_2)
    
    logger.info(f"Added patients: {doc_id_1}, {doc_id_2}")
    
    # 搜索相似患者
    query_patient = {
        "diagnosis": "breast_cancer",
        "age": 58,
        "stage": "II",
        "symptoms": ["fatigue"],
        "comorbidities": ["diabetes"]
    }
    
    similar_patients = await faiss_manager.async_search_similar_patients(query_patient, k=3)
    logger.info(f"Found {len(similar_patients)} similar patients")
    
    for patient in similar_patients:
        logger.info(f"Patient {patient.patient_id}: score={patient.score:.3f}")
    
    # 获取治疗建议
    from ..core.data_models import PatientState
    from datetime import datetime
    
    patient_state = PatientState(
        patient_id="query_patient",
        age=58,
        diagnosis="breast_cancer",
        stage="II",
        lab_results={"hemoglobin": 10.0},
        vital_signs={"bp": "140/90"},
        symptoms=["fatigue"],
        comorbidities=["diabetes"],
        psychological_status="concerned",
        quality_of_life_score=0.5,
        timestamp=datetime.now()
    )
    
    treatment_recommendations = faiss_manager.get_treatment_recommendations(patient_state)
    logger.info(f"Treatment recommendations: {treatment_recommendations}")
    
    # 获取数据库统计
    stats = faiss_manager.get_database_stats()
    logger.info(f"Database stats: {stats}")
    
    # 清理
    faiss_manager.cleanup()


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_faiss())