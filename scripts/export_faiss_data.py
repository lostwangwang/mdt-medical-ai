#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS数据导出工具
将index.pkl中的患者数据导出为JSON、CSV等格式
"""

import pickle
import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def load_faiss_data(pkl_file="clinical_memory_db/index.pkl"):
    """加载FAISS数据"""
    pkl_path = Path(pkl_file)
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        return None
    
    try:
        print(f"📥 加载 {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 成功加载数据")
        return data
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def extract_patients_data(data):
    """提取患者数据"""
    if not data or not hasattr(data, 'docstore'):
        print("❌ 无法访问文档数据")
        return []
    
    docs = data.docstore._dict
    patients = []
    
    print(f"📊 提取患者数据...")
    
    for doc_id, doc in docs.items():
        try:
            if hasattr(doc, 'page_content'):
                patient_data = json.loads(doc.page_content)
                patients.append(patient_data)
                
        except json.JSONDecodeError as e:
            print(f"❌ 文档 {doc_id} JSON解析失败: {e}")
        except Exception as e:
            print(f"❌ 处理文档 {doc_id} 失败: {e}")
    
    print(f"✅ 成功提取 {len(patients)} 位患者数据")
    return patients

def export_to_json(patients, output_dir="exported_data"):
    """导出为JSON格式"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"💾 导出JSON格式到 {output_path}...")
    
    # 导出完整数据
    full_data_file = output_path / "all_patients_full.json"
    try:
        with open(full_data_file, 'w', encoding='utf-8') as f:
            json.dump(patients, f, ensure_ascii=False, indent=2)
        print(f"✅ 完整数据: {full_data_file}")
    except Exception as e:
        print(f"❌ 导出完整数据失败: {e}")
    
    # 导出每个患者的单独文件
    patients_dir = output_path / "individual_patients"
    patients_dir.mkdir(exist_ok=True)
    
    for patient in patients:
        subject_id = patient.get('subject_id', 'unknown')
        patient_file = patients_dir / f"patient_{subject_id}.json"
        
        try:
            with open(patient_file, 'w', encoding='utf-8') as f:
                json.dump(patient, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 导出患者 {subject_id} 失败: {e}")
    
    print(f"✅ 个人文件: {patients_dir} ({len(patients)} 个文件)")

def export_to_csv(patients, output_dir="exported_data"):
    """导出为CSV格式"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📊 导出CSV格式到 {output_path}...")
    
    # 基本信息CSV
    basic_info = []
    for patient in patients:
        basic_info.append({
            'subject_id': patient.get('subject_id'),
            'gender': patient.get('gender'),
            'anchor_age': patient.get('anchor_age'),
            'allergies_count': len(patient.get('allergies', [])),
            'cancer_history_count': len(patient.get('cancer_history', [])),
            'chronic_diseases_count': len(patient.get('chronic_diseases', [])),
            'baseline_labs_count': len(patient.get('baseline_labs', {})),
            'medications_count': len(patient.get('discharge_medications', [])),
            'trend': patient.get('daily_summaries', {}).get('trend', ''),
            'summary_length': len(patient.get('daily_summaries', {}).get('summary', ''))
        })
    
    basic_csv = output_path / "patients_basic_info.csv"
    try:
        df = pd.DataFrame(basic_info)
        df.to_csv(basic_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 基本信息: {basic_csv}")
    except Exception as e:
        print(f"❌ 导出基本信息CSV失败: {e}")
    
    # 慢性疾病CSV
    chronic_diseases = []
    for patient in patients:
        subject_id = patient.get('subject_id')
        for disease in patient.get('chronic_diseases', []):
            chronic_diseases.append({
                'subject_id': subject_id,
                'disease_desc': disease.get('desc', ''),
                'icd9_code': disease.get('icd9', '')
            })
    
    if chronic_diseases:
        chronic_csv = output_path / "chronic_diseases.csv"
        try:
            df = pd.DataFrame(chronic_diseases)
            df.to_csv(chronic_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 慢性疾病: {chronic_csv}")
        except Exception as e:
            print(f"❌ 导出慢性疾病CSV失败: {e}")
    
    # 基线检验CSV
    baseline_labs = []
    for patient in patients:
        subject_id = patient.get('subject_id')
        for lab_name, value in patient.get('baseline_labs', {}).items():
            baseline_labs.append({
                'subject_id': subject_id,
                'lab_name': lab_name,
                'value': value
            })
    
    if baseline_labs:
        labs_csv = output_path / "baseline_labs.csv"
        try:
            df = pd.DataFrame(baseline_labs)
            df.to_csv(labs_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 基线检验: {labs_csv}")
        except Exception as e:
            print(f"❌ 导出基线检验CSV失败: {e}")
    
    # 药物CSV
    medications = []
    for patient in patients:
        subject_id = patient.get('subject_id')
        for med in patient.get('discharge_medications', []):
            medications.append({
                'subject_id': subject_id,
                'drug': med.get('drug', ''),
                'dose': med.get('dose', ''),
                'route': med.get('route', '')
            })
    
    if medications:
        meds_csv = output_path / "medications.csv"
        try:
            df = pd.DataFrame(medications)
            df.to_csv(meds_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 药物信息: {meds_csv}")
        except Exception as e:
            print(f"❌ 导出药物信息CSV失败: {e}")

def export_summary_report(patients, output_dir="exported_data"):
    """生成汇总报告"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📋 生成汇总报告...")
    
    # 统计信息
    total_patients = len(patients)
    
    # 性别统计
    genders = [p.get('gender') for p in patients if p.get('gender')]
    gender_counts = {}
    for gender in genders:
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
    
    # 年龄统计
    ages = [p.get('anchor_age') for p in patients if isinstance(p.get('anchor_age'), (int, float))]
    age_stats = {
        'min': min(ages) if ages else 0,
        'max': max(ages) if ages else 0,
        'mean': sum(ages) / len(ages) if ages else 0
    }
    
    # 疾病统计
    total_chronic_diseases = sum(len(p.get('chronic_diseases', [])) for p in patients)
    total_cancer_history = sum(len(p.get('cancer_history', [])) for p in patients)
    total_allergies = sum(len(p.get('allergies', [])) for p in patients)
    
    # 检验和药物统计
    total_labs = sum(len(p.get('baseline_labs', {})) for p in patients)
    total_medications = sum(len(p.get('discharge_medications', [])) for p in patients)
    
    # 病情趋势统计
    trends = [p.get('daily_summaries', {}).get('trend') for p in patients]
    trend_counts = {}
    for trend in trends:
        if trend:
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
    
    # 生成报告
    report = {
        'export_time': datetime.now().isoformat(),
        'total_patients': total_patients,
        'gender_distribution': gender_counts,
        'age_statistics': age_stats,
        'medical_data': {
            'total_chronic_diseases': total_chronic_diseases,
            'total_cancer_history': total_cancer_history,
            'total_allergies': total_allergies,
            'total_baseline_labs': total_labs,
            'total_medications': total_medications
        },
        'trend_distribution': trend_counts,
        'averages_per_patient': {
            'chronic_diseases': total_chronic_diseases / total_patients if total_patients > 0 else 0,
            'baseline_labs': total_labs / total_patients if total_patients > 0 else 0,
            'medications': total_medications / total_patients if total_patients > 0 else 0
        }
    }
    
    # 保存报告
    report_file = output_path / "summary_report.json"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ 汇总报告: {report_file}")
    except Exception as e:
        print(f"❌ 生成汇总报告失败: {e}")
    
    # 生成文本报告
    text_report = f"""
FAISS数据库汇总报告
==================
导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

患者统计:
- 总患者数: {total_patients}
- 性别分布: {gender_counts}
- 年龄范围: {age_stats['min']:.0f} - {age_stats['max']:.0f} 岁 (平均: {age_stats['mean']:.1f})

医疗数据统计:
- 慢性疾病总数: {total_chronic_diseases} (平均每人: {total_chronic_diseases/total_patients:.1f})
- 癌症病史总数: {total_cancer_history}
- 过敏史总数: {total_allergies}
- 基线检验总数: {total_labs} (平均每人: {total_labs/total_patients:.1f})
- 出院药物总数: {total_medications} (平均每人: {total_medications/total_patients:.1f})

病情趋势分布:
{chr(10).join(f'- {trend}: {count} 人' for trend, count in trend_counts.items())}
"""
    
    text_report_file = output_path / "summary_report.txt"
    try:
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        print(f"✅ 文本报告: {text_report_file}")
    except Exception as e:
        print(f"❌ 生成文本报告失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FAISS数据导出工具")
    parser.add_argument("--input", default="clinical_memory_db/index.pkl", help="输入PKL文件路径")
    parser.add_argument("--output", default="exported_data", help="输出目录")
    parser.add_argument("--format", choices=['json', 'csv', 'all'], default='all', help="导出格式")
    parser.add_argument("--no-report", action="store_true", help="不生成汇总报告")
    
    args = parser.parse_args()
    
    print("📤 FAISS数据导出工具")
    print("="*60)
    
    # 加载数据
    data = load_faiss_data(args.input)
    if not data:
        return
    
    # 提取患者数据
    patients = extract_patients_data(data)
    if not patients:
        print("❌ 没有找到患者数据")
        return
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    print(f"📁 输出目录: {output_path.absolute()}")
    
    # 导出数据
    if args.format in ['json', 'all']:
        export_to_json(patients, args.output)
    
    if args.format in ['csv', 'all']:
        export_to_csv(patients, args.output)
    
    # 生成汇总报告
    if not args.no_report:
        export_summary_report(patients, args.output)
    
    print(f"\n✅ 导出完成! 共处理 {len(patients)} 位患者的数据")
    print(f"📁 所有文件已保存到: {output_path.absolute()}")

if __name__ == "__main__":
    main()