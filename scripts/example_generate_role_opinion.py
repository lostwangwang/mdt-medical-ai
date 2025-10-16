from datetime import datetime
# from src.core.data_models import RoleType, TreatmentOption, PatientState
# from src.consensus.role_agents import RoleAgent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.data_models import RoleType, TreatmentOption, PatientState
from src.consensus.role_agents import RoleAgent

def main():
    # 注意：RoleAgent内部使用的生活质量判断阈值是0~1尺度
    # 若你的数据是0~100，请先转换到[0,1]
    qol_0_to_1 = 0.72  # 例如把72/100转换为0.72

    patient = PatientState(
        patient_id="DEMO_001",
        age=68,
        diagnosis="lung_cancer",
        stage="III",
        lab_results={"CEA": 5.1, "LDH": 210},  # 项目数量会影响数据完整性
        vital_signs={"BP_sys": 128, "BP_dia": 82, "HR": 86},
        symptoms=["cough", "fatigue"],
        comorbidities=["hypertension"],
        psychological_status="mild_anxiety",
        quality_of_life_score=qol_0_to_1,  # 建议传入0~1以匹配RoleAgent逻辑
        timestamp=datetime.now(),
    )

    # 模拟知识库检索结果（真实项目里由RAG系统提供）
    knowledge = {
        "guidelines": ["NCCN 2024 lung cancer"],
        "evidence": ["Trial-XYZ shows chemo+RT benefit in stage III"],
    }

    oncologist = RoleAgent(RoleType.ONCOLOGIST)
    opinion = oncologist.generate_initial_opinion(patient, knowledge)

    print("Role:", opinion.role.value)
    print("Confidence:", round(opinion.confidence, 3))
    print("Reasoning:", opinion.reasoning)
    print("Concerns:", opinion.concerns)
    print("Top-3 treatment preferences:")
    top3 = sorted(opinion.treatment_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
    for t, s in top3:
        print(f"  - {t.value}: {round(s, 3)}")

if __name__ == "__main__":
    main()