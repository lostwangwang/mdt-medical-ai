#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的FAISS数据库查看工具
适配实际的数据结构：(InMemoryDocstore, dict)
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

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

def load_faiss_data(pkl_file="clinical_memory_db/index.pkl"):
    """加载FAISS数据"""
    pkl_path = Path(pkl_file)
    
    try:
        print(f"\n📥 读取 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 成功加载数据")
        print(f"🔍 数据类型: {type(data)}")
        
        if isinstance(data, tuple) and len(data) >= 2:
            docstore = data[0]
            index_to_docstore_id = data[1]
            
            print(f"📦 元组结构:")
            print(f"  元素 0 (docstore): {type(docstore)}")
            print(f"  元素 1 (index映射): {type(index_to_docstore_id)}")
            
            return docstore, index_to_docstore_id
        else:
            print(f"❌ 意外的数据结构")
            return None, None
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None, None

def extract_documents(docstore, index_to_docstore_id):
    """提取文档数据"""
    if not docstore or not hasattr(docstore, '_dict'):
        print("❌ 无法访问docstore")
        return []
    
    docs_dict = docstore._dict
    print(f"📄 docstore中的文档数量: {len(docs_dict)}")
    print(f"🔗 索引映射数量: {len(index_to_docstore_id)}")
    
    patients = []
    
    for doc_id, doc in docs_dict.items():
        try:
            if hasattr(doc, 'page_content'):
                # 尝试解析JSON内容
                try:
                    patient_data = json.loads(doc.page_content)
                    patients.append({
                        'doc_id': doc_id,
                        'patient_data': patient_data,
                        'metadata': getattr(doc, 'metadata', {})
                    })
                except json.JSONDecodeError as e:
                    print(f"❌ 文档 {doc_id} JSON解析失败: {e}")
                    # 显示原始内容的一部分
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    print(f"   内容预览: {content_preview}")
            else:
                print(f"⚠️ 文档 {doc_id} 没有page_content属性")
                
        except Exception as e:
            print(f"❌ 处理文档 {doc_id} 时出错: {e}")
    
    print(f"✅ 成功解析 {len(patients)} 位患者数据")
    return patients

def display_patients_summary(patients):
    """显示患者汇总信息"""
    if not patients:
        print("❌ 没有患者数据")
        return
    
    print(f"\n" + "="*60)
    print(f"📊 患者数据汇总")
    print(f"="*60)
    
    print(f"👥 总患者数: {len(patients)}")
    
    # 统计基本信息
    genders = []
    ages = []
    
    for patient in patients:
        data = patient['patient_data']
        gender = data.get('gender')
        age = data.get('anchor_age')
        
        if gender:
            genders.append(gender)
        if isinstance(age, (int, float)):
            ages.append(age)
    
    # 性别分布
    if genders:
        from collections import Counter
        gender_counts = Counter(genders)
        print(f"👫 性别分布: {dict(gender_counts)}")
    
    # 年龄统计
    if ages:
        print(f"🎂 年龄范围: {min(ages)} - {max(ages)} 岁 (平均: {np.mean(ages):.1f})")
    
    # 医疗数据统计
    total_chronic = sum(len(p['patient_data'].get('chronic_diseases', [])) for p in patients)
    total_labs = sum(len(p['patient_data'].get('baseline_labs', {})) for p in patients)
    total_meds = sum(len(p['patient_data'].get('discharge_medications', [])) for p in patients)
    
    print(f"🏥 慢性疾病总数: {total_chronic}")
    print(f"🧪 基线检验总数: {total_labs}")
    print(f"💊 出院药物总数: {total_meds}")

def display_patients_list(patients):
    """显示患者列表"""
    print(f"\n" + "="*80)
    print(f"👥 患者详细列表")
    print(f"="*80)
    
    print(f"{'序号':<4} {'患者ID':<12} {'性别':<6} {'年龄':<6} {'慢性疾病':<8} {'检验项目':<8} {'药物':<6} {'病情趋势':<12}")
    print("-" * 80)
    
    for i, patient in enumerate(patients, 1):
        data = patient['patient_data']
        
        subject_id = data.get('subject_id', 'N/A')
        gender = data.get('gender', 'N/A')
        age = data.get('anchor_age', 'N/A')
        chronic_count = len(data.get('chronic_diseases', []))
        labs_count = len(data.get('baseline_labs', {}))
        meds_count = len(data.get('discharge_medications', []))
        
        # 病情趋势
        summaries = data.get('daily_summaries', {})
        trend = summaries.get('trend', 'N/A') if summaries else 'N/A'
        
        print(f"{i:<4} {subject_id:<12} {gender:<6} {age:<6} {chronic_count:<8} {labs_count:<8} {meds_count:<6} {trend:<12}")

def show_patient_detail(patients, subject_id):
    """显示特定患者详细信息"""
    patient = None
    for p in patients:
        if str(p['patient_data'].get('subject_id')) == str(subject_id):
            patient = p
            break
    
    if not patient:
        print(f"❌ 未找到患者ID为 {subject_id} 的数据")
        return
    
    data = patient['patient_data']
    
    print(f"\n" + "="*60)
    print(f"📋 患者 {subject_id} 详细信息")
    print(f"="*60)
    
    # 基本信息
    print(f"👤 基本信息:")
    print(f"  患者ID: {data.get('subject_id')}")
    print(f"  性别: {data.get('gender')}")
    print(f"  年龄: {data.get('anchor_age')}")
    
    # 过敏史
    allergies = data.get('allergies', [])
    print(f"\n🚫 过敏史 ({len(allergies)} 项):")
    for allergy in allergies[:5]:
        print(f"  - {allergy}")
    if len(allergies) > 5:
        print(f"  ... 还有 {len(allergies) - 5} 项")
    
    # 癌症病史
    cancer_history = data.get('cancer_history', [])
    print(f"\n🎗️ 癌症病史 ({len(cancer_history)} 项):")
    for cancer in cancer_history[:5]:
        desc = cancer.get('desc', 'N/A')
        icd9 = cancer.get('icd9', 'N/A')
        print(f"  - {desc} (ICD9: {icd9})")
    if len(cancer_history) > 5:
        print(f"  ... 还有 {len(cancer_history) - 5} 项")
    
    # 慢性疾病
    chronic_diseases = data.get('chronic_diseases', [])
    print(f"\n🏥 慢性疾病 ({len(chronic_diseases)} 项):")
    for disease in chronic_diseases[:5]:
        desc = disease.get('desc', 'N/A')
        icd9 = disease.get('icd9', 'N/A')
        print(f"  - {desc} (ICD9: {icd9})")
    if len(chronic_diseases) > 5:
        print(f"  ... 还有 {len(chronic_diseases) - 5} 项")
    
    # 基线检验
    baseline_labs = data.get('baseline_labs', {})
    print(f"\n🧪 基线检验 ({len(baseline_labs)} 项):")
    for i, (lab_name, value) in enumerate(baseline_labs.items()):
        if i >= 5:
            print(f"  ... 还有 {len(baseline_labs) - 5} 项")
            break
        print(f"  - {lab_name}: {value}")
    
    # 生命体征
    baseline_vitals = data.get('baseline_vitals', {})
    if baseline_vitals:
        print(f"\n💓 基线生命体征:")
        for vital_name, value in baseline_vitals.items():
            print(f"  - {vital_name}: {value}")
    
    # 出院药物
    medications = data.get('discharge_medications', [])
    print(f"\n💊 出院药物 ({len(medications)} 项):")
    for med in medications[:5]:
        drug = med.get('drug', 'N/A')
        dose = med.get('dose', 'N/A')
        route = med.get('route', 'N/A')
        print(f"  - {drug} ({dose}, {route})")
    if len(medications) > 5:
        print(f"  ... 还有 {len(medications) - 5} 项")
    
    # 病情总结
    daily_summaries = data.get('daily_summaries', {})
    if daily_summaries:
        print(f"\n📊 病情总结:")
        trend = daily_summaries.get('trend', 'N/A')
        summary = daily_summaries.get('summary', '')
        print(f"  趋势: {trend}")
        if summary:
            summary_preview = summary[:300] + "..." if len(summary) > 300 else summary
            print(f"  总结: {summary_preview}")

def main():
    """主函数"""
    print("🏥 修复后的FAISS数据库查看工具")
    print("="*60)
    
    # 检查文件
    if not check_faiss_files():
        return
    
    # 加载数据
    docstore, index_to_docstore_id = load_faiss_data()
    if not docstore:
        return
    
    # 提取患者数据
    patients = extract_documents(docstore, index_to_docstore_id)
    if not patients:
        return
    
    # 显示汇总信息
    display_patients_summary(patients)
    
    # 显示患者列表
    display_patients_list(patients)
    
    # 交互式查看
    while True:
        print(f"\n" + "="*40)
        print(f"请选择操作:")
        print(f"1. 重新显示患者列表")
        print(f"2. 查看特定患者详细信息")
        print(f"3. 显示汇总统计")
        print(f"4. 退出")
        
        choice = input(f"\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            display_patients_list(patients)
        elif choice == '2':
            subject_id = input("请输入患者ID: ").strip()
            show_patient_detail(patients, subject_id)
        elif choice == '3':
            display_patients_summary(patients)
        elif choice == '4':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()