#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS数据库快速查看工具
简化版本，用于快速查看index.pkl和index.faiss文件内容
"""

import pickle
import json
import os
from pathlib import Path

def check_faiss_files(db_path="clinical_memory_db"):
    """检查FAISS文件"""
    db_path = Path(db_path)
    pkl_file = db_path / "index.pkl"
    faiss_file = db_path / "index.faiss"
    
    print("🔍 检查FAISS数据库文件...")
    print(f"📂 数据库路径: {db_path.absolute()}")
    
    # 检查PKL文件
    if pkl_file.exists():
        size = pkl_file.stat().st_size
        print(f"✅ index.pkl: 存在 ({size:,} bytes)")
    else:
        print(f"❌ index.pkl: 不存在")
        return False
    
    # 检查FAISS文件
    if faiss_file.exists():
        size = faiss_file.stat().st_size
        print(f"✅ index.faiss: 存在 ({size:,} bytes)")
    else:
        print(f"⚠️ index.faiss: 不存在")
    
    return True

def read_pkl_data(pkl_file="clinical_memory_db/index.pkl"):
    """读取PKL文件数据"""
    pkl_path = Path(pkl_file)
    
    try:
        print(f"\n📥 读取 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 成功加载数据")
        print(f"🔍 数据类型: {type(data)}")
        
        # 显示对象属性
        if hasattr(data, '__dict__'):
            attrs = list(data.__dict__.keys())
            print(f"📋 对象属性: {attrs}")
        
        # 获取文档
        if hasattr(data, 'docstore') and hasattr(data.docstore, '_dict'):
            docs = data.docstore._dict
            print(f"📄 文档数量: {len(docs)}")
            
            # 分析文档
            print(f"\n📊 文档分析:")
            for i, (doc_id, doc) in enumerate(docs.items()):
                if i >= 3:  # 只显示前3个文档
                    print(f"... 还有 {len(docs) - 3} 个文档")
                    break
                
                print(f"\n--- 文档 {i+1} ---")
                print(f"ID: {doc_id}")
                
                if hasattr(doc, 'page_content'):
                    try:
                        patient_data = json.loads(doc.page_content)
                        print(f"患者ID: {patient_data.get('subject_id', 'N/A')}")
                        print(f"性别: {patient_data.get('gender', 'N/A')}")
                        print(f"年龄: {patient_data.get('anchor_age', 'N/A')}")
                        print(f"慢性疾病: {len(patient_data.get('chronic_diseases', []))} 项")
                        print(f"基线检验: {len(patient_data.get('baseline_labs', {}))} 项")
                        
                        # 显示病情总结
                        summaries = patient_data.get('daily_summaries', {})
                        if summaries:
                            trend = summaries.get('trend', 'N/A')
                            print(f"病情趋势: {trend}")
                        
                    except json.JSONDecodeError:
                        print("❌ JSON解析失败")
                
                if hasattr(doc, 'metadata'):
                    print(f"元数据: {doc.metadata}")
        
        return data
        
    except Exception as e:
        print(f"❌ 读取PKL文件失败: {e}")
        return None

def read_faiss_index(faiss_file="clinical_memory_db/index.faiss"):
    """读取FAISS索引文件"""
    faiss_path = Path(faiss_file)
    
    if not faiss_path.exists():
        print(f"⚠️ FAISS文件不存在: {faiss_path}")
        return None
    
    try:
        import faiss
        print(f"\n📥 读取 {faiss_path}...")
        index = faiss.read_index(str(faiss_path))
        
        print(f"✅ 成功加载FAISS索引")
        print(f"🔢 向量数量: {index.ntotal}")
        print(f"📏 向量维度: {index.d}")
        print(f"🏷️ 索引类型: {type(index)}")
        
        return index
        
    except ImportError:
        print(f"⚠️ 未安装faiss库，无法读取FAISS索引")
        return None
    except Exception as e:
        print(f"❌ 读取FAISS索引失败: {e}")
        return None

def list_all_patients(data):
    """列出所有患者"""
    if not data or not hasattr(data, 'docstore'):
        print("❌ 无法访问文档数据")
        return []
    
    docs = data.docstore._dict
    patients = []
    
    print(f"\n👥 患者列表:")
    print("-" * 80)
    print(f"{'序号':<4} {'患者ID':<12} {'性别':<6} {'年龄':<6} {'慢性疾病':<8} {'检验项目':<8} {'病情趋势':<10}")
    print("-" * 80)
    
    for i, (doc_id, doc) in enumerate(docs.items(), 1):
        try:
            if hasattr(doc, 'page_content'):
                patient_data = json.loads(doc.page_content)
                
                subject_id = patient_data.get('subject_id', 'N/A')
                gender = patient_data.get('gender', 'N/A')
                age = patient_data.get('anchor_age', 'N/A')
                chronic_count = len(patient_data.get('chronic_diseases', []))
                labs_count = len(patient_data.get('baseline_labs', {}))
                
                summaries = patient_data.get('daily_summaries', {})
                trend = summaries.get('trend', 'N/A') if summaries else 'N/A'
                
                print(f"{i:<4} {subject_id:<12} {gender:<6} {age:<6} {chronic_count:<8} {labs_count:<8} {trend:<10}")
                
                patients.append({
                    'subject_id': subject_id,
                    'data': patient_data
                })
                
        except Exception as e:
            print(f"{i:<4} 解析失败: {e}")
    
    print("-" * 80)
    print(f"总计: {len(patients)} 位患者")
    
    return patients

def show_patient_detail(patients, subject_id):
    """显示特定患者详细信息"""
    patient_data = None
    for patient in patients:
        if str(patient['subject_id']) == str(subject_id):
            patient_data = patient['data']
            break
    
    if not patient_data:
        print(f"❌ 未找到患者ID为 {subject_id} 的数据")
        return
    
    print(f"\n" + "="*60)
    print(f"📋 患者 {subject_id} 详细信息")
    print(f"="*60)
    
    # 基本信息
    print(f"👤 基本信息:")
    print(f"  患者ID: {patient_data.get('subject_id')}")
    print(f"  性别: {patient_data.get('gender')}")
    print(f"  年龄: {patient_data.get('anchor_age')}")
    
    # 过敏史
    allergies = patient_data.get('allergies', [])
    print(f"\n🚫 过敏史 ({len(allergies)} 项):")
    for allergy in allergies[:5]:  # 只显示前5项
        print(f"  - {allergy}")
    if len(allergies) > 5:
        print(f"  ... 还有 {len(allergies) - 5} 项")
    
    # 慢性疾病
    chronic_diseases = patient_data.get('chronic_diseases', [])
    print(f"\n🏥 慢性疾病 ({len(chronic_diseases)} 项):")
    for disease in chronic_diseases[:5]:  # 只显示前5项
        desc = disease.get('desc', 'N/A')
        icd9 = disease.get('icd9', 'N/A')
        print(f"  - {desc} (ICD9: {icd9})")
    if len(chronic_diseases) > 5:
        print(f"  ... 还有 {len(chronic_diseases) - 5} 项")
    
    # 基线检验
    baseline_labs = patient_data.get('baseline_labs', {})
    print(f"\n🧪 基线检验 ({len(baseline_labs)} 项):")
    for i, (lab_name, value) in enumerate(baseline_labs.items()):
        if i >= 5:  # 只显示前5项
            print(f"  ... 还有 {len(baseline_labs) - 5} 项")
            break
        print(f"  - {lab_name}: {value}")
    
    # 生命体征
    baseline_vitals = patient_data.get('baseline_vitals', {})
    if baseline_vitals:
        print(f"\n💓 基线生命体征:")
        for vital_name, value in baseline_vitals.items():
            print(f"  - {vital_name}: {value}")
    
    # 病情总结
    daily_summaries = patient_data.get('daily_summaries', {})
    if daily_summaries:
        print(f"\n📊 病情总结:")
        trend = daily_summaries.get('trend', 'N/A')
        summary = daily_summaries.get('summary', '')
        print(f"  趋势: {trend}")
        if summary:
            # 显示总结的前200个字符
            summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"  总结: {summary_preview}")

def main():
    """主函数"""
    print("🏥 FAISS数据库快速查看工具")
    print("="*60)
    
    # 检查文件
    if not check_faiss_files():
        return
    
    # 读取PKL数据
    data = read_pkl_data()
    if not data:
        return
    
    # 读取FAISS索引
    faiss_index = read_faiss_index()
    
    # 列出所有患者
    patients = list_all_patients(data)
    
    if not patients:
        print("❌ 没有找到患者数据")
        return
    
    # 交互式查看
    while True:
        print(f"\n" + "="*40)
        print(f"请选择操作:")
        print(f"1. 重新显示患者列表")
        print(f"2. 查看特定患者详细信息")
        print(f"3. 退出")
        
        choice = input(f"\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            list_all_patients(data)
        elif choice == '2':
            subject_id = input("请输入患者ID: ").strip()
            show_patient_detail(patients, subject_id)
        elif choice == '3':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()