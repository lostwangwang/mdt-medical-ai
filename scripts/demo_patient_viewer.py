#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者数据演示查看器
展示FAISS数据库中的患者详细信息
"""

import pickle
import json
import numpy as np
from pathlib import Path

def load_and_show_patients():
    """加载并展示患者数据"""
    pkl_path = Path("clinical_memory_db/index.pkl")
    
    print("🏥 FAISS数据库患者数据演示")
    print("="*60)
    
    # 加载数据
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        docstore = data[0]
        index_to_docstore_id = data[1]
        
        print(f"✅ 成功加载数据")
        print(f"📄 文档数量: {len(docstore._dict)}")
        print(f"🔗 索引映射: {len(index_to_docstore_id)}")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 提取患者数据
    patients = []
    for doc_id, doc in docstore._dict.items():
        try:
            patient_data = json.loads(doc.page_content)
            patients.append({
                'doc_id': doc_id,
                'data': patient_data
            })
        except:
            continue
    
    print(f"✅ 解析出 {len(patients)} 位患者数据")
    
    # 显示汇总统计
    print(f"\n📊 数据库统计:")
    genders = [p['data'].get('gender') for p in patients]
    ages = [p['data'].get('anchor_age') for p in patients if isinstance(p['data'].get('anchor_age'), (int, float))]
    
    from collections import Counter
    gender_counts = Counter(genders)
    print(f"  性别分布: {dict(gender_counts)}")
    print(f"  年龄范围: {min(ages)} - {max(ages)} 岁 (平均: {np.mean(ages):.1f})")
    
    total_chronic = sum(len(p['data'].get('chronic_diseases', [])) for p in patients)
    total_labs = sum(len(p['data'].get('baseline_labs', {})) for p in patients)
    total_meds = sum(len(p['data'].get('discharge_medications', [])) for p in patients)
    
    print(f"  慢性疾病总数: {total_chronic}")
    print(f"  基线检验总数: {total_labs}")
    print(f"  出院药物总数: {total_meds}")
    
    # 显示患者列表
    print(f"\n👥 患者列表:")
    print("-" * 80)
    print(f"{'序号':<4} {'患者ID':<12} {'性别':<6} {'年龄':<6} {'慢性疾病':<8} {'检验':<6} {'药物':<6} {'趋势':<10}")
    print("-" * 80)
    
    for i, patient in enumerate(patients, 1):
        data = patient['data']
        subject_id = data.get('subject_id', 'N/A')
        gender = data.get('gender', 'N/A')
        age = data.get('anchor_age', 'N/A')
        chronic_count = len(data.get('chronic_diseases', []))
        labs_count = len(data.get('baseline_labs', {}))
        meds_count = len(data.get('discharge_medications', []))
        trend = data.get('daily_summaries', {}).get('trend', 'N/A')
        
        print(f"{i:<4} {subject_id:<12} {gender:<6} {age:<6} {chronic_count:<8} {labs_count:<6} {meds_count:<6} {trend:<10}")
    
    # 展示第一个患者的详细信息
    if patients:
        print(f"\n" + "="*60)
        print(f"📋 患者详细信息示例 (患者ID: {patients[0]['data'].get('subject_id')})")
        print(f"="*60)
        
        show_patient_detail(patients[0]['data'])
    
    # 展示有慢性疾病的患者
    chronic_patients = [p for p in patients if len(p['data'].get('chronic_diseases', [])) > 0]
    if chronic_patients:
        print(f"\n" + "="*60)
        print(f"📋 有慢性疾病的患者示例 (患者ID: {chronic_patients[0]['data'].get('subject_id')})")
        print(f"="*60)
        
        show_patient_detail(chronic_patients[0]['data'])

def show_patient_detail(data):
    """显示患者详细信息"""
    # 基本信息
    print(f"👤 基本信息:")
    print(f"  患者ID: {data.get('subject_id')}")
    print(f"  性别: {data.get('gender')}")
    print(f"  年龄: {data.get('anchor_age')}")
    
    # 过敏史
    allergies = data.get('allergies', [])
    if allergies:
        print(f"\n🚫 过敏史 ({len(allergies)} 项):")
        for allergy in allergies[:3]:
            print(f"  - {allergy}")
        if len(allergies) > 3:
            print(f"  ... 还有 {len(allergies) - 3} 项")
    
    # 癌症病史
    cancer_history = data.get('cancer_history', [])
    if cancer_history:
        print(f"\n🎗️ 癌症病史 ({len(cancer_history)} 项):")
        for cancer in cancer_history[:3]:
            desc = cancer.get('desc', 'N/A')
            icd9 = cancer.get('icd9', 'N/A')
            print(f"  - {desc} (ICD9: {icd9})")
        if len(cancer_history) > 3:
            print(f"  ... 还有 {len(cancer_history) - 3} 项")
    
    # 慢性疾病
    chronic_diseases = data.get('chronic_diseases', [])
    if chronic_diseases:
        print(f"\n🏥 慢性疾病 ({len(chronic_diseases)} 项):")
        for disease in chronic_diseases[:3]:
            desc = disease.get('desc', 'N/A')
            icd9 = disease.get('icd9', 'N/A')
            print(f"  - {desc} (ICD9: {icd9})")
        if len(chronic_diseases) > 3:
            print(f"  ... 还有 {len(chronic_diseases) - 3} 项")
    
    # 基线检验
    baseline_labs = data.get('baseline_labs', {})
    if baseline_labs:
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
    if medications:
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
            summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"  总结: {summary_preview}")

if __name__ == "__main__":
    load_and_show_patients()