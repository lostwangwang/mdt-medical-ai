"""
患者对话记忆管理系统
文件路径: src/knowledge/dialogue_memory_manager.py
作者: AI Assistant
功能: 管理患者对话历史，支持FAISS向量存储和检索
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .enhanced_faiss_integration import EnhancedFAISSManager


class DialogueMemoryManager:
    """患者对话记忆管理器"""
    
    def __init__(self, memory_db_path: str = "dialogue_memory_db", 
                 embedding_model_name: str = "auto"):
        """初始化对话记忆管理器
        
        Args:
            memory_db_path: 记忆数据库路径
            embedding_model_name: 嵌入模型名称，支持：
                - "auto": 自动选择最佳医疗模型
                - "biobert": BioBERT医疗模型
                - "clinical-bert": ClinicalBERT临床模型
                - "mc-bert": 中文医疗BERT
                - "sentence-transformers/all-MiniLM-L6-v2": 通用模型（默认备选）
        """
        self.memory_db_path = memory_db_path
        self.logger = logging.getLogger(__name__)
        
        # 确保目录存在
        os.makedirs(memory_db_path, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model, self.embedding_dim = self._initialize_embedding_model(embedding_model_name)
        self.model_name = embedding_model_name
        
        # 初始化FAISS索引
        self.index_file = os.path.join(self.memory_db_path, "dialogue_index.faiss")
        self.metadata_file = os.path.join(self.memory_db_path, "dialogue_metadata.pkl")
        
        self._load_or_create_index()
    
    def _initialize_embedding_model(self, model_name: str) -> Tuple[SentenceTransformer, int]:
        """初始化嵌入模型
        
        Returns:
            Tuple[SentenceTransformer, int]: (模型实例, 嵌入维度)
        """
        try:
            if model_name == "auto":
                # 自动选择最佳可用的医疗模型
                return self._auto_select_medical_model()
            elif model_name == "biobert":
                return self._load_biobert_model()
            elif model_name == "clinical-bert":
                return self._load_clinical_bert_model()
            elif model_name == "mc-bert":
                return self._load_mc_bert_model()
            elif model_name.startswith("sentence-transformers/"):
                model = SentenceTransformer(model_name)
                return model, model.get_sentence_embedding_dimension()
            else:
                # 尝试直接加载
                model = SentenceTransformer(model_name)
                return model, model.get_sentence_embedding_dimension()
                
        except Exception as e:
            self.logger.warning(f"无法加载指定模型 {model_name}: {e}")
            self.logger.info("回退到默认通用模型")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            return model, 384
    
    def _auto_select_medical_model(self) -> Tuple[SentenceTransformer, int]:
        """自动选择最佳可用的医疗模型"""
        # 按优先级尝试加载医疗模型
        medical_models = [
            ("emilyalsentzer/Bio_ClinicalBERT", 768),  # ClinicalBERT
            ("dmis-lab/biobert-base-cased-v1.2", 768),  # BioBERT
            ("sentence-transformers/all-MiniLM-L6-v2", 384)  # 备选通用模型
        ]
        
        for model_name, dim in medical_models:
            try:
                self.logger.info(f"尝试加载医疗模型: {model_name}")
                model = SentenceTransformer(model_name)
                self.logger.info(f"成功加载医疗模型: {model_name}")
                return model, dim
            except Exception as e:
                self.logger.warning(f"无法加载 {model_name}: {e}")
                continue
        
        # 如果所有模型都失败，使用最基础的模型
        self.logger.error("所有医疗模型加载失败，使用基础模型")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model, 384
    
    def _load_biobert_model(self) -> Tuple[SentenceTransformer, int]:
        """加载BioBERT模型"""
        try:
            # 尝试多个BioBERT变体
            biobert_models = [
                "dmis-lab/biobert-base-cased-v1.2",
                "dmis-lab/biobert-large-cased-v1.1",
                "sentence-transformers/all-MiniLM-L6-v2"  # 备选
            ]
            
            for model_name in biobert_models:
                try:
                    model = SentenceTransformer(model_name)
                    dim = 768 if "biobert" in model_name else 384
                    self.logger.info(f"成功加载BioBERT模型: {model_name}")
                    return model, dim
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"BioBERT加载失败: {e}")
            
        # 备选方案
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model, 384
    
    def _load_clinical_bert_model(self) -> Tuple[SentenceTransformer, int]:
        """加载ClinicalBERT模型"""
        try:
            clinical_models = [
                "emilyalsentzer/Bio_ClinicalBERT",
                "sentence-transformers/all-MiniLM-L6-v2"  # 备选
            ]
            
            for model_name in clinical_models:
                try:
                    model = SentenceTransformer(model_name)
                    dim = 768 if "Clinical" in model_name else 384
                    self.logger.info(f"成功加载ClinicalBERT模型: {model_name}")
                    return model, dim
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"ClinicalBERT加载失败: {e}")
            
        # 备选方案
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model, 384
    
    def _load_mc_bert_model(self) -> Tuple[SentenceTransformer, int]:
        """加载中文医疗BERT模型"""
        try:
            # 中文医疗模型（需要根据实际可用模型调整）
            chinese_medical_models = [
                "bert-base-chinese",  # 基础中文BERT
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言模型
                "sentence-transformers/all-MiniLM-L6-v2"  # 备选
            ]
            
            for model_name in chinese_medical_models:
                try:
                    model = SentenceTransformer(model_name)
                    dim = model.get_sentence_embedding_dimension()
                    self.logger.info(f"成功加载中文医疗模型: {model_name}")
                    return model, dim
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"中文医疗BERT加载失败: {e}")
            
        # 备选方案
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model, 384
        
        self.logger.info(f"对话记忆管理器初始化完成，数据库路径: {self.memory_db_path}")
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # 加载现有索引
                existing_index = faiss.read_index(self.index_file)
                
                # 检查维度是否匹配
                if existing_index.d == self.embedding_dim:
                    self.index = existing_index
                    with open(self.metadata_file, 'rb') as f:
                        self.metadata = pickle.load(f)
                    self.logger.info(f"加载现有对话索引，包含 {self.index.ntotal} 条记录")
                else:
                    self.logger.warning(f"索引维度不匹配: 现有{existing_index.d}维 vs 当前{self.embedding_dim}维，创建新索引")
                    # 维度不匹配，创建新索引
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                    self.metadata = []
                    self.logger.info("创建新的对话索引")
            else:
                # 创建新索引
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # 使用内积相似度
                self.metadata = []
                self.logger.info("创建新的对话索引")
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            # 创建新索引作为备用
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.metadata = []
    
    def save_dialogue_turn(self, 
                          patient_id: str, 
                          user_query: str, 
                          agent_response: str, 
                          session_id: str = None,
                          additional_metadata: Dict[str, Any] = None) -> str:
        """保存一轮对话到记忆系统"""
        try:
            # 生成对话ID
            dialogue_id = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # 创建对话记录
            dialogue_record = {
                "dialogue_id": dialogue_id,
                "patient_id": patient_id,
                "session_id": session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "agent_response": agent_response,
                "metadata": additional_metadata or {}
            }
            
            # 生成对话内容的嵌入向量
            dialogue_text = f"患者查询: {user_query} 智能体回复: {agent_response}"
            embedding = self.embedding_model.encode([dialogue_text])
            
            # 标准化向量
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            
            # 添加到FAISS索引
            self.index.add(embedding.astype('float32'))
            
            # 添加元数据
            self.metadata.append(dialogue_record)
            
            # 保存索引和元数据
            self._save_index()
            
            # 同时保存到患者专用文件
            self._save_patient_dialogue_file(patient_id, dialogue_record)
            
            self.logger.info(f"保存对话记录: {dialogue_id}")
            return dialogue_id
            
        except Exception as e:
            self.logger.error(f"保存对话记录失败: {e}")
            raise e
    
    def _save_index(self):
        """保存FAISS索引和元数据"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            raise e
    
    def _save_patient_dialogue_file(self, patient_id: str, dialogue_record: Dict[str, Any]):
        """保存患者专用对话文件"""
        try:
            patient_file = os.path.join(self.memory_db_path, f"patient_{patient_id}_dialogues.json")
            
            # 读取现有记录
            if os.path.exists(patient_file):
                with open(patient_file, 'r', encoding='utf-8') as f:
                    patient_dialogues = json.load(f)
            else:
                patient_dialogues = {
                    "patient_id": patient_id,
                    "created_at": datetime.now().isoformat(),
                    "dialogues": []
                }
            
            # 添加新记录
            patient_dialogues["dialogues"].append(dialogue_record)
            patient_dialogues["last_updated"] = datetime.now().isoformat()
            patient_dialogues["total_dialogues"] = len(patient_dialogues["dialogues"])
            
            # 保存文件
            with open(patient_file, 'w', encoding='utf-8') as f:
                json.dump(patient_dialogues, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存患者对话文件失败: {e}")
            raise e
    
    def search_similar_dialogues(self, 
                                query: str, 
                                patient_id: str = None, 
                                k: int = 5,
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索相似的历史对话"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # 搜索相似向量
            scores, indices = self.index.search(query_embedding.astype('float32'), min(k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and score >= similarity_threshold:
                    dialogue_record = self.metadata[idx].copy()
                    dialogue_record['similarity_score'] = float(score)
                    
                    # 如果指定了患者ID，只返回该患者的对话
                    if patient_id is None or dialogue_record.get('patient_id') == patient_id:
                        results.append(dialogue_record)
                        
                        if len(results) >= k:
                            break
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜索相似对话失败: {e}")
            return []
    
    def get_patient_dialogue_history(self, 
                                   patient_id: str, 
                                   limit: int = 20,
                                   session_id: str = None) -> List[Dict[str, Any]]:
        """获取患者的对话历史"""
        try:
            patient_file = os.path.join(self.memory_db_path, f"patient_{patient_id}_dialogues.json")
            
            if not os.path.exists(patient_file):
                return []
            
            with open(patient_file, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            dialogues = patient_data.get("dialogues", [])
            
            # 如果指定了会话ID，只返回该会话的对话
            if session_id:
                dialogues = [d for d in dialogues if d.get('session_id') == session_id]
            
            # 按时间倒序排列，返回最近的对话
            dialogues.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return dialogues[:limit]
            
        except Exception as e:
            self.logger.error(f"获取患者对话历史失败: {e}")
            return []
    
    def get_patient_sessions(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取患者的所有会话信息"""
        try:
            dialogues = self.get_patient_dialogue_history(patient_id, limit=1000)
            
            sessions = {}
            for dialogue in dialogues:
                session_id = dialogue.get('session_id')
                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "start_time": dialogue.get('timestamp'),
                        "end_time": dialogue.get('timestamp'),
                        "dialogue_count": 0,
                        "last_query": "",
                        "last_response": ""
                    }
                
                session = sessions[session_id]
                session["dialogue_count"] += 1
                session["last_query"] = dialogue.get('user_query', '')
                session["last_response"] = dialogue.get('agent_response', '')
                
                # 更新时间范围
                if dialogue.get('timestamp', '') > session["end_time"]:
                    session["end_time"] = dialogue.get('timestamp')
                if dialogue.get('timestamp', '') < session["start_time"]:
                    session["start_time"] = dialogue.get('timestamp')
            
            return list(sessions.values())
            
        except Exception as e:
            self.logger.error(f"获取患者会话信息失败: {e}")
            return []
    
    def analyze_dialogue_patterns(self, patient_id: str) -> Dict[str, Any]:
        """分析患者的对话模式 - 智能增强版本"""
        try:
            dialogues = self.get_patient_dialogue_history(patient_id, limit=1000)
            
            if not dialogues:
                return {"error": "没有找到对话记录"}
            
            # 基础统计信息
            total_dialogues = len(dialogues)
            sessions = self.get_patient_sessions(patient_id)
            
            # 智能查询分类 - 使用语义相似度
            query_classification = self._intelligent_query_classification(dialogues)
            
            # 时间模式分析
            temporal_patterns = self._analyze_temporal_patterns(dialogues)
            
            # 情感和紧急程度分析
            sentiment_analysis = self._analyze_dialogue_sentiment(dialogues)
            
            # 对话复杂度分析
            complexity_analysis = self._analyze_dialogue_complexity(dialogues)
            
            # 主题演化分析
            topic_evolution = self._analyze_topic_evolution(dialogues)
            
            # 个性化洞察
            personalized_insights = self._generate_personalized_insights(
                dialogues, query_classification, temporal_patterns, sentiment_analysis
            )
            
            # 计算时间跨度
            timestamps = [d.get('timestamp') for d in dialogues if d.get('timestamp')]
            time_span = ""
            if timestamps:
                timestamps.sort()
                start_time = timestamps[0]
                end_time = timestamps[-1]
                time_span = f"{start_time} 到 {end_time}"
            
            return {
                "patient_id": patient_id,
                "basic_statistics": {
                    "total_dialogues": total_dialogues,
                    "total_sessions": len(sessions),
                    "time_span": time_span,
                    "average_dialogues_per_session": total_dialogues / len(sessions) if sessions else 0,
                },
                "intelligent_classification": query_classification,
                "temporal_patterns": temporal_patterns,
                "sentiment_analysis": sentiment_analysis,
                "complexity_analysis": complexity_analysis,
                "topic_evolution": topic_evolution,
                "personalized_insights": personalized_insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"分析对话模式失败: {e}")
            return {"error": str(e)}
    
    def evaluate_model_performance(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """评估当前模型在医疗文本上的性能"""
        try:
            if test_queries is None:
                # 默认医疗测试查询
                test_queries = [
                    "患者出现胸痛症状，需要进行心电图检查",
                    "肿瘤标志物升高，建议进一步影像学检查",
                    "血压控制不佳，需要调整降压药物剂量",
                    "术后恢复良好，伤口愈合正常",
                    "化疗副作用明显，患者出现恶心呕吐"
                ]
            
            # 生成嵌入向量
            embeddings = self.embedding_model.encode(test_queries)
            
            # 计算向量质量指标
            embedding_stats = {
                "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
                "dimension": self.embedding_dim,
                "model_name": self.model_name
            }
            
            # 计算语义相似度矩阵
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # 评估语义理解质量
            semantic_quality = {
                "avg_similarity": float(np.mean(similarity_matrix)),
                "similarity_variance": float(np.var(similarity_matrix)),
                "max_similarity": float(np.max(similarity_matrix[similarity_matrix < 1.0])),
                "min_similarity": float(np.min(similarity_matrix))
            }
            
            return {
                "model_info": {
                    "name": self.model_name,
                    "dimension": self.embedding_dim,
                    "type": self._get_model_type()
                },
                "embedding_stats": embedding_stats,
                "semantic_quality": semantic_quality,
                "test_queries_count": len(test_queries),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"模型性能评估失败: {e}")
            return {"error": str(e)}
    
    def _get_model_type(self) -> str:
        """获取模型类型"""
        model_name = self.model_name.lower()
        if "biobert" in model_name or "bio" in model_name:
            return "生物医学BERT"
        elif "clinical" in model_name:
            return "临床BERT"
        elif "chinese" in model_name or "bert-base-chinese" in model_name:
            return "中文医疗BERT"
        elif "sentence-transformers" in model_name:
            return "通用句子嵌入模型"
        else:
            return "未知类型"
    
    def switch_embedding_model(self, new_model_name: str) -> Dict[str, Any]:
        """切换嵌入模型"""
        try:
            old_model_name = self.model_name
            old_dimension = self.embedding_dim
            
            # 加载新模型
            new_model, new_dimension = self._initialize_embedding_model(new_model_name)
            
            # 检查维度兼容性
            if new_dimension != old_dimension:
                self.logger.warning(f"模型维度变化: {old_dimension} -> {new_dimension}")
                self.logger.warning("需要重建FAISS索引")
                
                # 备份旧索引
                backup_index_file = f"{self.index_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_metadata_file = f"{self.metadata_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if os.path.exists(self.index_file):
                    os.rename(self.index_file, backup_index_file)
                if os.path.exists(self.metadata_file):
                    os.rename(self.metadata_file, backup_metadata_file)
            
            # 更新模型
            self.embedding_model = new_model
            self.embedding_dim = new_dimension
            self.model_name = new_model_name
            
            # 重建索引（如果维度变化）
            if new_dimension != old_dimension:
                self._load_or_create_index()
            
            return {
                "success": True,
                "old_model": {
                    "name": old_model_name,
                    "dimension": old_dimension
                },
                "new_model": {
                    "name": new_model_name,
                    "dimension": new_dimension,
                    "type": self._get_model_type()
                },
                "dimension_changed": new_dimension != old_dimension,
                "switch_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"模型切换失败: {e}")
            return {"error": str(e), "success": False}
    
    def get_available_medical_models(self) -> Dict[str, Any]:
        """获取可用的医疗模型列表"""
        models = {
            "biobert_models": [
                {
                    "name": "dmis-lab/biobert-base-cased-v1.2",
                    "description": "BioBERT基础版本，基于PubMed训练",
                    "dimension": 768,
                    "language": "English",
                    "domain": "生物医学"
                },
                {
                    "name": "dmis-lab/biobert-large-cased-v1.1", 
                    "description": "BioBERT大型版本",
                    "dimension": 1024,
                    "language": "English",
                    "domain": "生物医学"
                }
            ],
            "clinical_models": [
                {
                    "name": "emilyalsentzer/Bio_ClinicalBERT",
                    "description": "ClinicalBERT，基于临床笔记训练",
                    "dimension": 768,
                    "language": "English", 
                    "domain": "临床医学"
                }
            ],
            "chinese_models": [
                {
                    "name": "bert-base-chinese",
                    "description": "中文BERT基础模型",
                    "dimension": 768,
                    "language": "Chinese",
                    "domain": "通用中文"
                }
            ],
            "general_models": [
                {
                    "name": "sentence-transformers/all-MiniLM-L6-v2",
                    "description": "轻量级通用句子嵌入模型",
                    "dimension": 384,
                    "language": "Multilingual",
                    "domain": "通用"
                }
            ]
        }
        
        return {
            "available_models": models,
            "current_model": {
                "name": self.model_name,
                "dimension": self.embedding_dim,
                "type": self._get_model_type()
            },
            "recommendation": self._get_model_recommendation()
        }
    
    def _get_model_recommendation(self) -> str:
        """获取模型推荐"""
        if "chinese" in self.model_name.lower() or "中文" in self.model_name:
            return "当前使用中文模型，适合中文医疗文本处理"
        elif "biobert" in self.model_name.lower():
            return "当前使用BioBERT，在生物医学文本理解方面表现优秀"
        elif "clinical" in self.model_name.lower():
            return "当前使用ClinicalBERT，在临床文本处理方面表现优秀"
        elif "sentence-transformers" in self.model_name:
            return "当前使用通用模型，建议切换到专业医疗模型以获得更好的性能"
        else:
            return "建议根据具体应用场景选择合适的医疗专业模型"
    
    def _intelligent_query_classification(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """智能查询分类 - 使用语义相似度和机器学习"""
        try:
            # 扩展的医疗查询类别定义
            medical_categories = {
                "诊断相关": [
                    "诊断", "diagnosis", "病情", "症状", "表现", "征象", "体征",
                    "确诊", "疑似", "鉴别", "differential", "symptom", "sign"
                ],
                "治疗相关": [
                    "治疗", "treatment", "疗法", "方案", "手术", "surgery", "操作",
                    "intervention", "procedure", "therapy", "rehabilitation", "康复"
                ],
                "药物相关": [
                    "药物", "medication", "drug", "用药", "剂量", "dosage", "副作用",
                    "side effect", "药品", "处方", "prescription", "服用", "给药"
                ],
                "检查相关": [
                    "检查", "examination", "化验", "lab", "test", "影像", "imaging",
                    "CT", "MRI", "X光", "超声", "血检", "尿检", "病理", "pathology"
                ],
                "病史相关": [
                    "病史", "history", "既往", "家族史", "过敏史", "手术史",
                    "previous", "family", "allergy", "surgical", "medical history"
                ],
                "预后相关": [
                    "预后", "prognosis", "恢复", "recovery", "生存", "survival",
                    "复发", "recurrence", "转移", "metastasis", "预期", "outlook"
                ],
                "生活质量": [
                    "生活质量", "quality of life", "日常", "活动", "工作", "饮食",
                    "睡眠", "运动", "心理", "情绪", "anxiety", "depression"
                ],
                "费用相关": [
                    "费用", "cost", "价格", "医保", "insurance", "报销", "经济",
                    "负担", "支付", "payment", "expense"
                ]
            }
            
            # 使用向量相似度进行分类
            category_scores = {category: 0 for category in medical_categories.keys()}
            category_scores["其他"] = 0
            
            detailed_classification = []
            
            for dialogue in dialogues:
                query = dialogue.get('user_query', '').lower()
                if not query:
                    continue
                
                # 生成查询向量
                query_embedding = self.embedding_model.encode([query])
                
                best_category = "其他"
                best_score = 0
                category_confidences = {}
                
                # 对每个类别计算相似度
                for category, keywords in medical_categories.items():
                    # 计算与关键词的语义相似度
                    keyword_embeddings = self.embedding_model.encode(keywords)
                    similarities = np.dot(query_embedding, keyword_embeddings.T).flatten()
                    max_similarity = np.max(similarities)
                    
                    category_confidences[category] = float(max_similarity)
                    
                    if max_similarity > best_score and max_similarity > 0.3:  # 阈值
                        best_score = max_similarity
                        best_category = category
                
                category_scores[best_category] += 1
                
                detailed_classification.append({
                    "query": query[:100],  # 截断显示
                    "category": best_category,
                    "confidence": best_score,
                    "all_confidences": category_confidences,
                    "timestamp": dialogue.get('timestamp')
                })
            
            return {
                "category_distribution": category_scores,
                "most_common_category": max(category_scores.items(), key=lambda x: x[1])[0],
                "detailed_classification": detailed_classification[-10:],  # 最近10条
                "classification_method": "semantic_similarity"
            }
            
        except Exception as e:
            self.logger.error(f"智能查询分类失败: {e}")
            return {"error": str(e)}
    
    def _analyze_temporal_patterns(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间模式"""
        try:
            from collections import defaultdict
            from datetime import datetime, timedelta
            
            # 按时间段统计
            hourly_distribution = defaultdict(int)
            daily_distribution = defaultdict(int)
            weekly_distribution = defaultdict(int)
            
            dialogue_intervals = []
            
            for i, dialogue in enumerate(dialogues):
                timestamp_str = dialogue.get('timestamp')
                if not timestamp_str:
                    continue
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    # 时间分布统计
                    hourly_distribution[timestamp.hour] += 1
                    daily_distribution[timestamp.strftime('%A')] += 1
                    weekly_distribution[timestamp.strftime('%Y-W%U')] += 1
                    
                    # 计算对话间隔
                    if i > 0:
                        prev_timestamp_str = dialogues[i-1].get('timestamp')
                        if prev_timestamp_str:
                            prev_timestamp = datetime.fromisoformat(prev_timestamp_str.replace('Z', '+00:00'))
                            interval = (timestamp - prev_timestamp).total_seconds() / 3600  # 小时
                            dialogue_intervals.append(interval)
                            
                except ValueError:
                    continue
            
            # 分析活跃时间段
            peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None
            peak_day = max(daily_distribution.items(), key=lambda x: x[1])[0] if daily_distribution else None
            
            # 计算平均对话间隔
            avg_interval = np.mean(dialogue_intervals) if dialogue_intervals else 0
            
            return {
                "hourly_distribution": dict(hourly_distribution),
                "daily_distribution": dict(daily_distribution),
                "peak_hour": peak_hour,
                "peak_day": peak_day,
                "average_dialogue_interval_hours": avg_interval,
                "total_active_weeks": len(weekly_distribution),
                "dialogue_frequency_pattern": self._classify_frequency_pattern(dialogue_intervals)
            }
            
        except Exception as e:
            self.logger.error(f"时间模式分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_dialogue_sentiment(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析对话情感和紧急程度"""
        try:
            # 情感关键词
            anxiety_keywords = ["担心", "焦虑", "害怕", "紧张", "不安", "worried", "anxious", "scared"]
            urgency_keywords = ["急", "紧急", "马上", "立即", "urgent", "emergency", "immediately"]
            positive_keywords = ["好转", "改善", "恢复", "满意", "感谢", "better", "improved", "satisfied"]
            
            sentiment_scores = {"焦虑": 0, "紧急": 0, "积极": 0, "中性": 0}
            sentiment_timeline = []
            
            for dialogue in dialogues:
                query = dialogue.get('user_query', '').lower()
                response = dialogue.get('agent_response', '').lower()
                combined_text = f"{query} {response}"
                
                anxiety_score = sum(1 for keyword in anxiety_keywords if keyword in combined_text)
                urgency_score = sum(1 for keyword in urgency_keywords if keyword in combined_text)
                positive_score = sum(1 for keyword in positive_keywords if keyword in combined_text)
                
                if urgency_score > 0:
                    sentiment = "紧急"
                elif anxiety_score > positive_score:
                    sentiment = "焦虑"
                elif positive_score > 0:
                    sentiment = "积极"
                else:
                    sentiment = "中性"
                
                sentiment_scores[sentiment] += 1
                sentiment_timeline.append({
                    "timestamp": dialogue.get('timestamp'),
                    "sentiment": sentiment,
                    "anxiety_score": anxiety_score,
                    "urgency_score": urgency_score,
                    "positive_score": positive_score
                })
            
            return {
                "sentiment_distribution": sentiment_scores,
                "dominant_sentiment": max(sentiment_scores.items(), key=lambda x: x[1])[0],
                "sentiment_timeline": sentiment_timeline[-20:],  # 最近20条
                "emotional_stability": self._calculate_emotional_stability(sentiment_timeline)
            }
            
        except Exception as e:
            self.logger.error(f"情感分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_dialogue_complexity(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析对话复杂度"""
        try:
            query_lengths = []
            response_lengths = []
            medical_term_counts = []
            
            # 医学术语示例（可扩展）
            medical_terms = [
                "症状", "诊断", "治疗", "药物", "手术", "检查", "病理", "影像",
                "化疗", "放疗", "免疫", "靶向", "预后", "复发", "转移"
            ]
            
            for dialogue in dialogues:
                query = dialogue.get('user_query', '')
                response = dialogue.get('agent_response', '')
                
                query_lengths.append(len(query))
                response_lengths.append(len(response))
                
                # 计算医学术语数量
                medical_count = sum(1 for term in medical_terms if term in f"{query} {response}")
                medical_term_counts.append(medical_count)
            
            return {
                "average_query_length": np.mean(query_lengths) if query_lengths else 0,
                "average_response_length": np.mean(response_lengths) if response_lengths else 0,
                "average_medical_terms": np.mean(medical_term_counts) if medical_term_counts else 0,
                "complexity_trend": self._calculate_complexity_trend(query_lengths, medical_term_counts),
                "dialogue_depth_score": np.mean(response_lengths) / np.mean(query_lengths) if query_lengths and np.mean(query_lengths) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"复杂度分析失败: {e}")
            return {"error": str(e)}
    
    def _analyze_topic_evolution(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析主题演化"""
        try:
            if len(dialogues) < 5:
                return {"message": "对话数量不足，无法分析主题演化"}
            
            # 将对话分为时间段
            time_segments = self._segment_dialogues_by_time(dialogues, num_segments=5)
            
            topic_evolution = []
            for i, segment in enumerate(time_segments):
                if not segment:
                    continue
                
                # 提取该时间段的主要主题
                segment_queries = [d.get('user_query', '') for d in segment]
                main_topics = self._extract_main_topics(segment_queries)
                
                topic_evolution.append({
                    "time_segment": i + 1,
                    "dialogue_count": len(segment),
                    "main_topics": main_topics,
                    "time_range": f"{segment[0].get('timestamp', '')} - {segment[-1].get('timestamp', '')}"
                })
            
            return {
                "topic_evolution": topic_evolution,
                "topic_consistency": self._calculate_topic_consistency(topic_evolution),
                "emerging_topics": self._identify_emerging_topics(topic_evolution)
            }
            
        except Exception as e:
            self.logger.error(f"主题演化分析失败: {e}")
            return {"error": str(e)}
    
    def _generate_personalized_insights(self, dialogues, classification, temporal, sentiment) -> Dict[str, Any]:
        """生成个性化洞察"""
        try:
            insights = []
            
            # 基于查询类型的洞察
            if classification.get("category_distribution"):
                most_common = classification["most_common_category"]
                insights.append(f"患者最关注 {most_common} 相关问题")
            
            # 基于时间模式的洞察
            if temporal.get("peak_hour") is not None:
                peak_hour = temporal["peak_hour"]
                if 6 <= peak_hour <= 12:
                    insights.append("患者倾向于在上午咨询，可能是工作日程安排的结果")
                elif 18 <= peak_hour <= 22:
                    insights.append("患者倾向于在晚间咨询，建议关注其日间症状变化")
            
            # 基于情感的洞察
            if sentiment.get("dominant_sentiment"):
                dominant = sentiment["dominant_sentiment"]
                if dominant == "焦虑":
                    insights.append("患者表现出较高的焦虑水平，建议加强心理支持")
                elif dominant == "紧急":
                    insights.append("患者经常表达紧急需求，建议优化响应时间")
            
            # 对话频率洞察
            if temporal.get("average_dialogue_interval_hours"):
                interval = temporal["average_dialogue_interval_hours"]
                if interval < 24:
                    insights.append("患者咨询频率较高，可能需要更多关注")
                elif interval > 168:  # 一周
                    insights.append("患者咨询间隔较长，建议主动跟进")
            
            return {
                "insights": insights,
                "recommendation_priority": self._calculate_priority_score(classification, temporal, sentiment),
                "suggested_actions": self._suggest_actions(insights)
            }
            
        except Exception as e:
            self.logger.error(f"生成个性化洞察失败: {e}")
            return {"error": str(e)}
    
    # 辅助方法
    def _classify_frequency_pattern(self, intervals):
        """分类对话频率模式"""
        if not intervals:
            return "无规律"
        
        avg_interval = np.mean(intervals)
        if avg_interval < 1:
            return "高频"
        elif avg_interval < 24:
            return "日常"
        elif avg_interval < 168:
            return "周期性"
        else:
            return "偶发"
    
    def _calculate_emotional_stability(self, sentiment_timeline):
        """计算情感稳定性"""
        if len(sentiment_timeline) < 3:
            return "数据不足"
        
        sentiment_changes = 0
        for i in range(1, len(sentiment_timeline)):
            if sentiment_timeline[i]["sentiment"] != sentiment_timeline[i-1]["sentiment"]:
                sentiment_changes += 1
        
        stability_ratio = 1 - (sentiment_changes / len(sentiment_timeline))
        
        if stability_ratio > 0.8:
            return "稳定"
        elif stability_ratio > 0.6:
            return "较稳定"
        else:
            return "波动较大"
    
    def _calculate_complexity_trend(self, query_lengths, medical_terms):
        """计算复杂度趋势"""
        if len(query_lengths) < 5:
            return "数据不足"
        
        # 简单的趋势分析
        recent_avg = np.mean(query_lengths[-5:])
        early_avg = np.mean(query_lengths[:5])
        
        if recent_avg > early_avg * 1.2:
            return "复杂度上升"
        elif recent_avg < early_avg * 0.8:
            return "复杂度下降"
        else:
            return "复杂度稳定"
    
    def _segment_dialogues_by_time(self, dialogues, num_segments=5):
        """按时间分段对话"""
        if len(dialogues) < num_segments:
            return [dialogues]
        
        segment_size = len(dialogues) // num_segments
        segments = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(dialogues)
            segments.append(dialogues[start_idx:end_idx])
        
        return segments
    
    def _extract_main_topics(self, queries):
        """提取主要主题（简化版本）"""
        # 这里可以使用更复杂的NLP技术，如LDA主题建模
        word_freq = {}
        for query in queries:
            words = query.split()
            for word in words:
                if len(word) > 2:  # 过滤短词
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的前3个词
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        return [word for word, freq in top_words]
    
    def _calculate_topic_consistency(self, topic_evolution):
        """计算主题一致性"""
        if len(topic_evolution) < 2:
            return "数据不足"
        
        # 简化的一致性计算
        all_topics = set()
        for segment in topic_evolution:
            all_topics.update(segment.get("main_topics", []))
        
        if len(all_topics) <= 3:
            return "高度一致"
        elif len(all_topics) <= 6:
            return "中等一致"
        else:
            return "主题分散"
    
    def _identify_emerging_topics(self, topic_evolution):
        """识别新兴主题"""
        if len(topic_evolution) < 3:
            return []
        
        early_topics = set()
        for segment in topic_evolution[:2]:
            early_topics.update(segment.get("main_topics", []))
        
        recent_topics = set()
        for segment in topic_evolution[-2:]:
            recent_topics.update(segment.get("main_topics", []))
        
        emerging = recent_topics - early_topics
        return list(emerging)
    
    def _calculate_priority_score(self, classification, temporal, sentiment):
        """计算优先级分数"""
        score = 0
        
        # 基于情感的优先级
        if sentiment.get("dominant_sentiment") == "紧急":
            score += 3
        elif sentiment.get("dominant_sentiment") == "焦虑":
            score += 2
        
        # 基于频率的优先级
        if temporal.get("dialogue_frequency_pattern") == "高频":
            score += 2
        
        # 基于查询类型的优先级
        urgent_categories = ["诊断相关", "治疗相关", "药物相关"]
        if classification.get("most_common_category") in urgent_categories:
            score += 1
        
        if score >= 4:
            return "高"
        elif score >= 2:
            return "中"
        else:
            return "低"
    
    def _suggest_actions(self, insights):
        """基于洞察建议行动"""
        actions = []
        
        for insight in insights:
            if "焦虑" in insight:
                actions.append("安排心理咨询师介入")
            elif "紧急" in insight:
                actions.append("建立快速响应机制")
            elif "高频" in insight:
                actions.append("增加主动关怀频次")
            elif "间隔较长" in insight:
                actions.append("设置定期随访提醒")
        
        return actions
    
    def get_dialogue_context(self, 
                           patient_id: str, 
                           current_query: str,
                           context_window: int = 5) -> Dict[str, Any]:
        """
        获取对话上下文，用于智能体分析
        
        参数:
        patient_id: 患者ID
        current_query: 当前患者的查询语句
        context_window: 上下文窗口大小，默认5条对话
        
        返回:
        Dict[str, Any]: 包含患者ID、当前查询语句、最近对话历史、相似对话历史、对话模式分析和上下文获取时间戳的字典
        """
        try:
            # 获取最近的对话历史
            recent_dialogues = self.get_patient_dialogue_history(patient_id, limit=context_window)
            
            # 搜索相似的历史对话
            similar_dialogues = self.search_similar_dialogues(current_query, patient_id, k=3)
            
            # 分析对话模式
            patterns = self.analyze_dialogue_patterns(patient_id)
            
            return {
                "patient_id": patient_id,
                "current_query": current_query,
                "recent_dialogues": recent_dialogues,
                "similar_dialogues": similar_dialogues,
                "dialogue_patterns": patterns,
                "context_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取对话上下文失败: {e}")
            return {
                "patient_id": patient_id,
                "current_query": current_query,
                "error": str(e)
            }
    
    def export_patient_memory(self, patient_id: str, output_path: str = None) -> str:
        """导出患者的完整记忆数据"""
        try:
            if output_path is None:
                output_path = f"patient_{patient_id}_memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 收集所有相关数据
            export_data = {
                "patient_id": patient_id,
                "export_timestamp": datetime.now().isoformat(),
                "dialogue_history": self.get_patient_dialogue_history(patient_id, limit=10000),
                "sessions": self.get_patient_sessions(patient_id),
                "dialogue_patterns": self.analyze_dialogue_patterns(patient_id),
                "total_records": 0
            }
            
            export_data["total_records"] = len(export_data["dialogue_history"])
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"导出患者记忆数据到: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"导出患者记忆数据失败: {e}")
            raise
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        try:
            # 统计患者数量
            patient_files = [f for f in os.listdir(self.memory_db_path) if f.startswith('patient_') and f.endswith('_dialogues.json')]
            patient_count = len(patient_files)
            
            # 统计总对话数
            total_dialogues = self.index.ntotal if hasattr(self, 'index') else 0
            
            # 统计各患者的对话数量
            patient_stats = {}
            for patient_file in patient_files:
                patient_id = patient_file.replace('patient_', '').replace('_dialogues.json', '')
                try:
                    with open(os.path.join(self.memory_db_path, patient_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        patient_stats[patient_id] = {
                            "dialogue_count": len(data.get("dialogues", [])),
                            "last_updated": data.get("last_updated", ""),
                            "created_at": data.get("created_at", "")
                        }
                except Exception as e:
                    self.logger.warning(f"读取患者文件失败 {patient_file}: {e}")
            
            return {
                "total_patients": patient_count,
                "total_dialogues": total_dialogues,
                "database_path": self.memory_db_path,
                "patient_statistics": patient_stats,
                "index_file_exists": os.path.exists(self.index_file),
                "metadata_file_exists": os.path.exists(self.metadata_file),
                "statistics_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取记忆统计信息失败: {e}")
            return {"error": str(e)}