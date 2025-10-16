from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.llm_interface import LLMInterface
from src.core.data_models import PatientState

llm = LLMInterface()

patient = PatientState(
    patient_id="P001",
    age=65,
    diagnosis="breast_cancer",
    stage="II",
    lab_results={"creatinine": 1.2},
    vital_signs={"bp_systolic": 140},
    symptoms=["fatigue"],
    comorbidities=["hypertension"],
    psychological_status="anxious",
    quality_of_life_score=7.5,
    timestamp=datetime.now(),
)

plan = llm.generate_treatment_plan(patient, memory_context={"previous_treatments": ["surgery"]})
print(plan)