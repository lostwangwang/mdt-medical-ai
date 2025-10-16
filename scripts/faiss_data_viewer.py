#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS数据库查看工具
用于查看和分析clinical_memory_db中的患者数据
支持读取index.faiss和index.pkl文件
"""

import pickle
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

class FAISSDataViewer:
    """FAISS数据库查看器"""
    
    def __init__(self, db_path: str = "clinical_memory_db"):
        self.db_path = Path(db_path)
        self.pkl_file = self.db_path / "index.pkl"
        self.faiss_file = self.db_path / "index.faiss"
        self.data = None
        self.faiss_index = None
        
    def check_files(self) -> bool:
        """检查必要文件是否存在"""
        print("🔍 检查FAISS数据库文件...")
        print(f"📂 数据库路径: {self.db_path.absolute()}")
        
        pkl_exists = self.pkl_file.exists()
        faiss_exists = self.faiss_file.exists()
        
        print(f"📄 index.pkl: {'✅ 存在' if pkl_exists else '❌ 不存在'}")
        if pkl_exists:
            size = self.pkl_file.stat().st_size
            print(f"   大小: {size:,} bytes ({size/1024:.1f} KB)")
            
        print(f"📄 index.faiss: {'✅ 存在' if faiss_exists else '❌ 不存在'}")
        if faiss_exists:
            size = self.faiss_file.stat().st_size
            print(f"   大小: {size:,} bytes ({size/1024:.1f} KB)")
        
        return pkl_exists
    
    def load_pkl_data(self) -> bool:
        """加载PKL文件数据"""
        try:
            print(f"\n📥 正在加载 {self.pkl_file}...")
            with open(self.pkl_file, 'rb') as f:
                self.data = pickle.load(f)
            
            print(f"✅ 成功加载PKL数据")
            print(f"🔍 数据类型: {type(self.data)}")
            
            if hasattr(self.data, '__dict__'):
                attrs = list(self.data.__dict__.keys())
                print(f"📋 对象属性: {attrs}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载PKL文件失败: {e}")
            return False
    
    def load_faiss_index(self) -> bool:
        """加载FAISS索引"""
        if not self.faiss_file.exists():
            print("⚠️ FAISS文件不存在，跳过索引加载")
            return False
            
        try:
            import faiss
            print(f"\n📥 正在加载 {self.faiss_file}...")
            self.faiss_index = faiss.read_index(str(self.faiss_file))
            
            print(f"✅ 成功加载FAISS索引")
            print(f"🔢 向量数量: {self.faiss_index.ntotal}")
            print(f"📏 向量维度: {self.faiss_index.d}")
            print(f"🏷️ 索引类型: {type(self.faiss_index)}")
            
            return True
            
        except ImportError:
            print("⚠️ 未安装faiss库，无法加载FAISS索引")
            return False
        except Exception as e:
            print(f"❌ 加载FAISS索引失败: {e}")
            return False
    
    def get_documents(self) -> Dict[str, Any]:
        """获取所有文档"""
        if not self.data or not hasattr(self.data, 'docstore'):
            return {}
        
        if hasattr(self.data.docstore, '_dict'):
            return self.data.docstore._dict
        else:
            print("⚠️ 无法访问文档存储")
            return {}
    
    def analyze_documents(self) -> List[Dict[str, Any]]:
        """分析所有文档并提取患者信息"""
        docs = self.get_documents()
        patients = []
        
        print(f"\n📊 分析文档数据...")
        print(f"📄 总文档数: {len(docs)}")
        
        for doc_id, doc in docs.items():
            try:
                # 解析文档内容
                if hasattr(doc, 'page_content'):
                    patient_data = json.loads(doc.page_content)
                    
                    # 提取关键信息
                    patient_info = {
                        'doc_id': doc_id,
                        'subject_id': patient_data.get('subject_id', 'N/A'),
                        'gender': patient_data.get('gender', 'N/A'),
                        'age': patient_data.get('anchor_age', 'N/A'),
                        'allergies_count': len(patient_data.get('allergies', [])),
                        'cancer_history_count': len(patient_data.get('cancer_history', [])),
                        'chronic_diseases_count': len(patient_data.get('chronic_diseases', [])),
                        'baseline_labs_count': len(patient_data.get('baseline_labs', {})),
                        'baseline_vitals': patient_data.get('baseline_vitals', {}),
                        'medications_count': len(patient_data.get('discharge_medications', [])),
                        'daily_summaries': patient_data.get('daily_summaries', {}),
                        'metadata': getattr(doc, 'metadata', {}),
                        'full_data': patient_data
                    }
                    
                    patients.append(patient_info)
                    
            except json.JSONDecodeError as e:
                print(f"❌ 文档 {doc_id} JSON解析失败: {e}")
            except Exception as e:
                print(f"❌ 处理文档 {doc_id} 时出错: {e}")
        
        return patients
    
    def display_summary(self, patients: List[Dict[str, Any]]):
        """显示数据库摘要"""
        print(f"\n" + "="*60)
        print(f"📊 FAISS数据库摘要")
        print(f"="*60)
        
        if not patients:
            print("❌ 没有找到有效的患者数据")
            return
        
        print(f"👥 患者总数: {len(patients)}")
        
        # 性别统计
        genders = [p['gender'] for p in patients if p['gender'] != 'N/A']
        if genders:
            from collections import Counter
            gender_counts = Counter(genders)
            print(f"👫 性别分布: {dict(gender_counts)}")
        
        # 年龄统计
        ages = [p['age'] for p in patients if isinstance(p['age'], (int, float))]
        if ages:
            print(f"🎂 年龄范围: {min(ages)} - {max(ages)} 岁 (平均: {np.mean(ages):.1f})")
        
        # 疾病统计
        total_chronic = sum(p['chronic_diseases_count'] for p in patients)
        total_cancer = sum(p['cancer_history_count'] for p in patients)
        print(f"🏥 慢性疾病总数: {total_chronic}")
        print(f"🎗️ 癌症病史总数: {total_cancer}")
        
        # 检验数据统计
        total_labs = sum(p['baseline_labs_count'] for p in patients)
        print(f"🧪 基线检验项目总数: {total_labs}")
        
        # 药物统计
        total_meds = sum(p['medications_count'] for p in patients)
        print(f"💊 出院药物总数: {total_meds}")
    
    def display_patient_details(self, patients: List[Dict[str, Any]]):
        """显示患者详细信息"""
        print(f"\n" + "="*60)
        print(f"👥 患者详细信息")
        print(f"="*60)
        
        for i, patient in enumerate(patients, 1):
            print(f"\n--- 患者 {i} ---")
            print(f"📋 文档ID: {patient['doc_id']}")
            print(f"🆔 患者ID: {patient['subject_id']}")
            print(f"👤 性别: {patient['gender']}")
            print(f"🎂 年龄: {patient['age']}")
            print(f"🚫 过敏史: {patient['allergies_count']} 项")
            print(f"🎗️ 癌症病史: {patient['cancer_history_count']} 项")
            print(f"🏥 慢性疾病: {patient['chronic_diseases_count']} 项")
            print(f"🧪 基线检验: {patient['baseline_labs_count']} 项")
            print(f"💊 出院药物: {patient['medications_count']} 项")
            
            # 生命体征
            vitals = patient['baseline_vitals']
            if vitals:
                print(f"💓 基线生命体征:")
                for vital, value in vitals.items():
                    print(f"   {vital}: {value}")
            
            # 病情总结
            summaries = patient['daily_summaries']
            if summaries:
                trend = summaries.get('trend', 'N/A')
                summary = summaries.get('summary', '')
                print(f"📈 病情趋势: {trend}")
                if summary:
                    print(f"📝 病情总结: {summary[:100]}{'...' if len(summary) > 100 else ''}")
            
            # 元数据
            if patient['metadata']:
                print(f"🏷️ 元数据: {patient['metadata']}")
    
    def export_patient_data(self, patients: List[Dict[str, Any]], output_dir: str = "exported_data"):
        """导出患者数据到JSON文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 导出患者数据到 {output_path.absolute()}...")
        
        for patient in patients:
            subject_id = patient['subject_id']
            filename = f"patient_{subject_id}.json"
            filepath = output_path / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(patient['full_data'], f, ensure_ascii=False, indent=2)
                print(f"✅ 导出患者 {subject_id}: {filename}")
            except Exception as e:
                print(f"❌ 导出患者 {subject_id} 失败: {e}")
        
        # 导出汇总信息
        summary_file = output_path / "patients_summary.json"
        summary_data = {
            'total_patients': len(patients),
            'patients': [
                {
                    'subject_id': p['subject_id'],
                    'gender': p['gender'],
                    'age': p['age'],
                    'chronic_diseases_count': p['chronic_diseases_count'],
                    'baseline_labs_count': p['baseline_labs_count'],
                    'trend': p['daily_summaries'].get('trend', 'N/A') if p['daily_summaries'] else 'N/A'
                }
                for p in patients
            ]
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 导出汇总信息: patients_summary.json")
        except Exception as e:
            print(f"❌ 导出汇总信息失败: {e}")
    
    def search_patient(self, subject_id: str, patients: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """搜索特定患者"""
        for patient in patients:
            if str(patient['subject_id']) == str(subject_id):
                return patient
        return None
    
    def display_patient_full_data(self, patient: Dict[str, Any]):
        """显示患者完整数据"""
        print(f"\n" + "="*60)
        print(f"📋 患者 {patient['subject_id']} 完整数据")
        print(f"="*60)
        
        full_data = patient['full_data']
        
        # 基本信息
        print(f"👤 基本信息:")
        print(f"  患者ID: {full_data.get('subject_id')}")
        print(f"  性别: {full_data.get('gender')}")
        print(f"  年龄: {full_data.get('anchor_age')}")
        
        # 过敏史
        allergies = full_data.get('allergies', [])
        print(f"\n🚫 过敏史 ({len(allergies)} 项):")
        for allergy in allergies:
            print(f"  - {allergy}")
        
        # 癌症病史
        cancer_history = full_data.get('cancer_history', [])
        print(f"\n🎗️ 癌症病史 ({len(cancer_history)} 项):")
        for cancer in cancer_history:
            print(f"  - {cancer.get('desc', 'N/A')} (ICD9: {cancer.get('icd9', 'N/A')})")
        
        # 慢性疾病
        chronic_diseases = full_data.get('chronic_diseases', [])
        print(f"\n🏥 慢性疾病 ({len(chronic_diseases)} 项):")
        for disease in chronic_diseases:
            print(f"  - {disease.get('desc', 'N/A')} (ICD9: {disease.get('icd9', 'N/A')})")
        
        # 基线检验
        baseline_labs = full_data.get('baseline_labs', {})
        print(f"\n🧪 基线检验 ({len(baseline_labs)} 项):")
        for lab_name, value in baseline_labs.items():
            print(f"  - {lab_name}: {value}")
        
        # 基线生命体征
        baseline_vitals = full_data.get('baseline_vitals', {})
        print(f"\n💓 基线生命体征:")
        for vital_name, value in baseline_vitals.items():
            print(f"  - {vital_name}: {value}")
        
        # 出院药物
        medications = full_data.get('discharge_medications', [])
        print(f"\n💊 出院药物 ({len(medications)} 项):")
        for med in medications:
            drug = med.get('drug', 'N/A')
            dose = med.get('dose', 'N/A')
            route = med.get('route', 'N/A')
            print(f"  - {drug} ({dose}, {route})")
        
        # 病情总结
        daily_summaries = full_data.get('daily_summaries', {})
        if daily_summaries:
            print(f"\n📊 病情总结:")
            print(f"  趋势: {daily_summaries.get('trend', 'N/A')}")
            summary = daily_summaries.get('summary', '')
            if summary:
                print(f"  总结: {summary}")
    
    def analyze_vectors(self):
        """分析向量数据"""
        if not self.faiss_index:
            print("⚠️ FAISS索引未加载，无法分析向量")
            return
        
        print(f"\n" + "="*60)
        print(f"🔢 向量数据分析")
        print(f"="*60)
        
        try:
            # 获取所有向量
            vectors = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
            
            print(f"📊 向量统计:")
            print(f"  形状: {vectors.shape}")
            print(f"  数据类型: {vectors.dtype}")
            print(f"  均值: {np.mean(vectors):.6f}")
            print(f"  标准差: {np.std(vectors):.6f}")
            print(f"  最小值: {np.min(vectors):.6f}")
            print(f"  最大值: {np.max(vectors):.6f}")
            
            # 计算向量间的相似度
            if self.faiss_index.ntotal > 1:
                # 计算第一个向量与其他向量的相似度
                query_vector = vectors[0:1]
                distances, indices = self.faiss_index.search(query_vector, min(5, self.faiss_index.ntotal))
                
                print(f"\n🔍 相似度分析 (以第一个向量为查询):")
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    print(f"  排名 {i+1}: 索引 {idx}, 距离 {dist:.6f}")
            
        except Exception as e:
            print(f"❌ 向量分析失败: {e}")
    
    def run_interactive_mode(self):
        """运行交互模式"""
        patients = self.analyze_documents()
        
        while True:
            print(f"\n" + "="*60)
            print(f"🔍 FAISS数据库交互查看器")
            print(f"="*60)
            print(f"1. 显示数据库摘要")
            print(f"2. 显示所有患者列表")
            print(f"3. 查看特定患者详细信息")
            print(f"4. 导出患者数据")
            print(f"5. 分析向量数据")
            print(f"6. 退出")
            
            choice = input(f"\n请选择操作 (1-6): ").strip()
            
            if choice == '1':
                self.display_summary(patients)
            elif choice == '2':
                self.display_patient_details(patients)
            elif choice == '3':
                subject_id = input("请输入患者ID: ").strip()
                patient = self.search_patient(subject_id, patients)
                if patient:
                    self.display_patient_full_data(patient)
                else:
                    print(f"❌ 未找到患者ID为 {subject_id} 的数据")
            elif choice == '4':
                output_dir = input("请输入导出目录 (默认: exported_data): ").strip()
                if not output_dir:
                    output_dir = "exported_data"
                self.export_patient_data(patients, output_dir)
            elif choice == '5':
                self.analyze_vectors()
            elif choice == '6':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重试")
    
    def run(self, interactive: bool = True):
        """运行查看器"""
        print("🏥 FAISS数据库查看工具")
        print("="*60)
        
        # 检查文件
        if not self.check_files():
            print("❌ 必要文件不存在，无法继续")
            return
        
        # 加载数据
        if not self.load_pkl_data():
            print("❌ 无法加载PKL数据，无法继续")
            return
        
        # 加载FAISS索引（可选）
        self.load_faiss_index()
        
        if interactive:
            self.run_interactive_mode()
        else:
            # 非交互模式，直接显示所有信息
            patients = self.analyze_documents()
            self.display_summary(patients)
            self.display_patient_details(patients)
            if self.faiss_index:
                self.analyze_vectors()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS数据库查看工具")
    parser.add_argument("--db-path", default="clinical_memory_db", help="数据库路径")
    parser.add_argument("--patient-id", help="查看特定患者ID")
    parser.add_argument("--export", help="导出数据到指定目录")
    parser.add_argument("--no-interactive", action="store_true", help="非交互模式")
    
    args = parser.parse_args()
    
    viewer = FAISSDataViewer(args.db_path)
    
    if args.patient_id:
        # 查看特定患者
        if not viewer.check_files() or not viewer.load_pkl_data():
            return
        
        patients = viewer.analyze_documents()
        patient = viewer.search_patient(args.patient_id, patients)
        if patient:
            viewer.display_patient_full_data(patient)
        else:
            print(f"❌ 未找到患者ID为 {args.patient_id} 的数据")
    
    elif args.export:
        # 导出数据
        if not viewer.check_files() or not viewer.load_pkl_data():
            return
        
        patients = viewer.analyze_documents()
        viewer.export_patient_data(patients, args.export)
    
    else:
        # 正常运行
        viewer.run(interactive=not args.no_interactive)


if __name__ == "__main__":
    main()