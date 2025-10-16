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
    
    def __init__(self, memory_db_path: str = "dialogue_memory_db"):
        self.memory_db_path = memory_db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化目录
        os.makedirs(self.memory_db_path, exist_ok=True)
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384  # all-MiniLM-L6-v2的维度
        
        # 初始化FAISS索引
        self.index_file = os.path.join(self.memory_db_path, "dialogue_index.faiss")
        self.metadata_file = os.path.join(self.memory_db_path, "dialogue_metadata.pkl")
        
        self._load_or_create_index()
        
        self.logger.info(f"对话记忆管理器初始化完成，数据库路径: {self.memory_db_path}")
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # 加载现有索引
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                self.logger.info(f"加载现有对话索引，包含 {self.index.ntotal} 条记录")
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
            raise
    
    def _save_index(self):
        """保存FAISS索引和元数据"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            raise
    
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
        """分析患者的对话模式"""
        try:
            dialogues = self.get_patient_dialogue_history(patient_id, limit=1000)
            
            if not dialogues:
                return {"error": "没有找到对话记录"}
            
            # 分析统计信息
            total_dialogues = len(dialogues)
            sessions = self.get_patient_sessions(patient_id)
            
            # 分析查询类型
            query_types = {
                "诊断相关": 0,
                "治疗相关": 0,
                "药物相关": 0,
                "检查相关": 0,
                "病史相关": 0,
                "其他": 0
            }
            
            for dialogue in dialogues:
                query = dialogue.get('user_query', '').lower()
                if any(keyword in query for keyword in ["诊断", "diagnosis"]):
                    query_types["诊断相关"] += 1
                elif any(keyword in query for keyword in ["治疗", "treatment"]):
                    query_types["治疗相关"] += 1
                elif any(keyword in query for keyword in ["药物", "medication", "drug"]):
                    query_types["药物相关"] += 1
                elif any(keyword in query for keyword in ["检查", "lab", "test"]):
                    query_types["检查相关"] += 1
                elif any(keyword in query for keyword in ["病史", "history"]):
                    query_types["病史相关"] += 1
                else:
                    query_types["其他"] += 1
            
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
                "total_dialogues": total_dialogues,
                "total_sessions": len(sessions),
                "time_span": time_span,
                "query_type_distribution": query_types,
                "average_dialogues_per_session": total_dialogues / len(sessions) if sessions else 0,
                "most_common_query_type": max(query_types.items(), key=lambda x: x[1])[0] if any(query_types.values()) else "无",
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"分析对话模式失败: {e}")
            return {"error": str(e)}
    
    def get_dialogue_context(self, 
                           patient_id: str, 
                           current_query: str,
                           context_window: int = 5) -> Dict[str, Any]:
        """获取对话上下文，用于智能体分析"""
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