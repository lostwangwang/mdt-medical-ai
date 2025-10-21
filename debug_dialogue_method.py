#!/usr/bin/env python3
"""
Debug script to test which dialogue method is being called
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm_interface import LLMInterface
from src.core.data_models import PatientState, RoleType, TreatmentOption
from datetime import datetime

def test_dialogue_method():
    """Test which dialogue method is actually being called"""
    
    # Create test patient
    patient = PatientState(
        patient_id="TEST001",
        age=68,
        diagnosis="乳腺癌",
        stage="II期",
        lab_results={"CEA": 4.5, "CA153": 28.7},
        vital_signs={"BP": 140.0, "HR": 78.0},
        symptoms=["疲劳", "食欲不振"],
        comorbidities=["糖尿病", "高血压"],
        psychological_status="中度抑郁",
        quality_of_life_score=60.0,
        timestamp=datetime.now()
    )
    
    llm = LLMInterface()
    
    # Test scenarios
    test_cases = [
        {
            'role': RoleType.ONCOLOGIST,
            'treatment': TreatmentOption.CHEMOTHERAPY,
            'context': '患者担心化疗副作用',
            'expected_keywords': ['副作用', '管理', '监测', '预防']
        },
        {
            'role': RoleType.NURSE,
            'treatment': TreatmentOption.SURGERY,
            'context': '术前准备咨询',
            'expected_keywords': ['术前', '准备', '注意事项', '护理']
        },
        {
            'role': RoleType.PSYCHOLOGIST,
            'treatment': TreatmentOption.RADIOTHERAPY,
            'context': '患者对放疗感到恐惧',
            'expected_keywords': ['心理', '恐惧', '支持', '缓解']
        }
    ]
    
    print("=== 对话方法调试测试 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试 {i}: {case['role'].value} - {case['treatment'].value}")
        print(f"讨论背景: {case['context']}")
        
        # Test dialogue response
        response = llm.generate_dialogue_response(
            patient_state=patient,
            role=case['role'],
            treatment_option=case['treatment'],
            discussion_context=case['context'],
            knowledge_context={},
            current_stance={'position': 'neutral', 'confidence': 0.7},
            dialogue_history=[]
        )
        
        print(f"生成响应: {response}")
        print(f"响应长度: {len(response)} 字符")
        
        # Check for expected keywords
        found_keywords = []
        for keyword in case['expected_keywords']:
            if keyword in response:
                found_keywords.append(keyword)
        
        print(f"期望关键词: {case['expected_keywords']}")
        print(f"找到关键词: {found_keywords}")
        print(f"匹配率: {len(found_keywords)}/{len(case['expected_keywords'])}")
        
        # Test direct template method call
        print("\n--- 直接调用模板方法 ---")
        template_response = llm._generate_template_dialogue_fallback(
            patient, case['role'], case['treatment'], case['context']
        )
        print(f"模板响应: {template_response}")
        print(f"模板长度: {len(template_response)} 字符")
        
        template_keywords = []
        for keyword in case['expected_keywords']:
            if keyword in template_response:
                template_keywords.append(keyword)
        print(f"模板匹配: {template_keywords}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_dialogue_method()