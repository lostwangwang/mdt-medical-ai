"""
数据处理工具模块
文件路径: src/utils/data_processor.py
作者: 团队共同维护
功能: 提供数据清洗、转换、验证等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from dataclasses import asdict

from ..core.data_models import PatientState, TreatmentOption, MedicalEvent

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""

    def __init__(self):
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """加载验证规则"""
        return {
            "patient_id": {"type": str, "required": True, "pattern": r"^[A-Z0-9_]+$"},
            "age": {"type": int, "required": True, "min": 0, "max": 120},
            "diagnosis": {
                "type": str,
                "required": True,
                "allowed": ["breast_cancer", "lung_cancer", "colon_cancer"],
            },
            "stage": {
                "type": str,
                "required": True,
                "allowed": ["0", "I", "II", "III", "IV"],
            },
            "quality_of_life_score": {
                "type": float,
                "required": True,
                "min": 0.0,
                "max": 1.0,
            },
            "lab_results": {"type": dict, "required": True},
            "vital_signs": {"type": dict, "required": True},
            "symptoms": {"type": list, "required": False},
            "comorbidities": {"type": list, "required": False},
        }

    def validate_patient_state(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证患者状态数据"""
        errors = []

        for field, rules in self.validation_rules.items():
            # 检查必需字段
            if rules.get("required", False) and field not in data:
                errors.append(f"Missing required field: {field}")
                continue

            if field not in data:
                continue

            value = data[field]

            # 检查数据类型
            expected_type = rules["type"]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Field {field} should be {expected_type.__name__}, got {type(value).__name__}"
                )
                continue

            # 数值范围检查
            if expected_type in [int, float]:
                if "min" in rules and value < rules["min"]:
                    errors.append(
                        f"Field {field} value {value} is below minimum {rules['min']}"
                    )
                if "max" in rules and value > rules["max"]:
                    errors.append(
                        f"Field {field} value {value} is above maximum {rules['max']}"
                    )

            # 允许值检查
            if "allowed" in rules and value not in rules["allowed"]:
                errors.append(
                    f"Field {field} value '{value}' not in allowed values: {rules['allowed']}"
                )

            # 模式匹配检查
            if "pattern" in rules:
                import re

                if not re.match(rules["pattern"], str(value)):
                    errors.append(
                        f"Field {field} value '{value}' doesn't match pattern {rules['pattern']}"
                    )

        return len(errors) == 0, errors

    def validate_medical_event(
        self, event_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """验证医疗事件数据"""
        errors = []
        required_fields = ["patient_id", "time", "event_type", "name", "value"]

        for field in required_fields:
            if field not in event_data:
                errors.append(f"Missing required field: {field}")

        # 检查事件类型
        valid_event_types = ["lab", "vital", "medication", "symptom", "clinical_note"]
        if (
            "event_type" in event_data
            and event_data["event_type"] not in valid_event_types
        ):
            errors.append(f"Invalid event_type: {event_data['event_type']}")

        return len(errors) == 0, errors


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.cleaning_stats = {}

    def clean_patient_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗患者数据"""
        logger.info("Starting patient data cleaning...")

        original_count = len(df)

        # 去除重复数据
        df = df.drop_duplicates(subset=["patient_id"])
        logger.info(f"Removed {original_count - len(df)} duplicate records")

        # 清洗年龄数据
        df = self._clean_age_data(df)

        # 清洗实验室数据
        df = self._clean_lab_data(df)

        # 清洗生命体征数据
        df = self._clean_vital_signs(df)

        # 清洗分类数据
        df = self._clean_categorical_data(df)

        # 处理缺失值
        df = self._handle_missing_values(df)

        self.cleaning_stats["final_count"] = len(df)
        self.cleaning_stats["cleaned_percentage"] = (len(df) / original_count) * 100

        logger.info(
            f"Data cleaning completed. Retained {len(df)}/{original_count} records ({self.cleaning_stats['cleaned_percentage']:.1f}%)"
        )

        return df

    def _clean_age_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗年龄数据"""
        if "age" not in df.columns:
            return df

        # 移除不合理的年龄值
        invalid_age_mask = (df["age"] < 0) | (df["age"] > 120)
        invalid_count = invalid_age_mask.sum()

        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid age values, setting to NaN")
            df.loc[invalid_age_mask, "age"] = np.nan

        return df

    def _clean_lab_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗实验室数据"""
        lab_columns = [
            col
            for col in df.columns
            if "lab_" in col.lower() or col in ["creatinine", "hemoglobin", "cea"]
        ]

        for col in lab_columns:
            if col in df.columns:
                # 移除极端异常值 (超过3个标准差)
                mean_val = df[col].mean()
                std_val = df[col].std()

                outlier_mask = np.abs((df[col] - mean_val) / std_val) > 3
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    logger.info(
                        f"Found {outlier_count} outliers in {col}, capping values"
                    )
                    # 将异常值设置为边界值而不是删除
                    upper_bound = mean_val + 3 * std_val
                    lower_bound = mean_val - 3 * std_val
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    df.loc[df[col] < lower_bound, col] = lower_bound

        return df

    def _clean_vital_signs(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗生命体征数据"""
        vital_ranges = {
            "bp_systolic": (60, 250),
            "bp_diastolic": (40, 150),
            "heart_rate": (30, 200),
            "temperature": (35.0, 42.0),
            "weight": (30, 300),  # kg
        }

        for col, (min_val, max_val) in vital_ranges.items():
            if col in df.columns:
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_count = invalid_mask.sum()

                if invalid_count > 0:
                    logger.warning(
                        f"Found {invalid_count} invalid {col} values, setting to NaN"
                    )
                    df.loc[invalid_mask, col] = np.nan

        return df

    def _clean_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗分类数据"""
        # 标准化诊断名称
        if "diagnosis" in df.columns:
            diagnosis_mapping = {
                "breast cancer": "breast_cancer",
                "breast ca": "breast_cancer",
                "lung cancer": "lung_cancer",
                "lung ca": "lung_cancer",
            }

            df["diagnosis"] = (
                df["diagnosis"]
                .str.lower()
                .map(diagnosis_mapping)
                .fillna(df["diagnosis"])
            )

        # 标准化分期
        if "stage" in df.columns:
            stage_mapping = {
                "stage 0": "0",
                "stage i": "I",
                "stage ii": "II",
                "stage iii": "III",
                "stage iv": "IV",
            }

            df["stage"] = df["stage"].str.lower().map(stage_mapping).fillna(df["stage"])

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 计算缺失值比例
        missing_ratio = df.isnull().sum() / len(df)

        # 移除缺失值过多的列 (>50%)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index
        if len(high_missing_cols) > 0:
            logger.warning(
                f"Dropping columns with >50% missing values: {list(high_missing_cols)}"
            )
            df = df.drop(columns=high_missing_cols)

        # 对数值列使用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(
                    f"Filled {col} missing values with median: {median_val:.2f}"
                )

        # 对分类列使用众数填充
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {mode_val[0]}")

        return df


class DataTransformer:
    """数据转换器"""

    def __init__(self):
        self.transformation_history = []

    def transform_to_patient_states(self, df: pd.DataFrame) -> List[PatientState]:
        """将DataFrame转换为PatientState对象列表"""
        patient_states = []

        for _, row in df.iterrows():
            try:
                # 提取实验室结果
                lab_results = {}
                for col in df.columns:
                    if any(
                        lab in col.lower()
                        for lab in ["creatinine", "hemoglobin", "cea", "lab_"]
                    ):
                        if pd.notna(row[col]):
                            lab_results[col] = float(row[col])

                # 提取生命体征
                vital_signs = {}
                for col in df.columns:
                    if any(
                        vital in col.lower()
                        for vital in ["bp_", "heart_rate", "temperature", "weight"]
                    ):
                        if pd.notna(row[col]):
                            vital_signs[col] = float(row[col])

                # 处理症状列表
                symptoms = []
                if "symptoms" in row and pd.notna(row["symptoms"]):
                    if isinstance(row["symptoms"], str):
                        symptoms = [s.strip() for s in row["symptoms"].split(",")]
                    elif isinstance(row["symptoms"], list):
                        symptoms = row["symptoms"]

                # 处理并发症列表
                comorbidities = []
                if "comorbidities" in row and pd.notna(row["comorbidities"]):
                    if isinstance(row["comorbidities"], str):
                        comorbidities = [
                            c.strip() for c in row["comorbidities"].split(",")
                        ]
                    elif isinstance(row["comorbidities"], list):
                        comorbidities = row["comorbidities"]

                # 创建PatientState对象
                patient_state = PatientState(
                    patient_id=str(row["patient_id"]),
                    age=int(row["age"]) if pd.notna(row["age"]) else 65,
                    diagnosis=str(row.get("diagnosis", "breast_cancer")),
                    stage=str(row.get("stage", "II")),
                    lab_results=lab_results,
                    vital_signs=vital_signs,
                    symptoms=symptoms,
                    comorbidities=comorbidities,
                    psychological_status=str(row.get("psychological_status", "stable")),
                    quality_of_life_score=float(row.get("quality_of_life_score", 0.7)),
                    timestamp=datetime.now(),
                )

                patient_states.append(patient_state)

            except Exception as e:
                logger.error(
                    f"Error transforming patient {row.get('patient_id', 'unknown')}: {e}"
                )
                continue

        logger.info(f"Successfully transformed {len(patient_states)} patient records")
        return patient_states

    def normalize_features(
        self, df: pd.DataFrame, method: str = "minmax"
    ) -> pd.DataFrame:
        """特征标准化"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        elif method == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        df_normalized = df.copy()
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        self.transformation_history.append(
            {
                "method": "normalize_features",
                "scaler_type": method,
                "columns": list(numeric_cols),
                "timestamp": datetime.now(),
            }
        )

        logger.info(
            f"Normalized {len(numeric_cols)} numeric features using {method} scaling"
        )
        return df_normalized

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """分类特征编码"""
        categorical_cols = df.select_dtypes(include=["object"]).columns
        df_encoded = df.copy()

        for col in categorical_cols:
            # 使用标签编码
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            df_encoded[col + "_encoded"] = le.fit_transform(df_encoded[col].astype(str))

            # 保留原始列以备参考
            # df_encoded = df_encoded.drop(columns=[col])

        self.transformation_history.append(
            {
                "method": "encode_categorical_features",
                "columns": list(categorical_cols),
                "timestamp": datetime.now(),
            }
        )

        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df_encoded


class DataExporter:
    """数据导出器"""

    def __init__(self):
        self.export_history = []

    def export_patient_states(
        self, patient_states: List[PatientState], filepath: str, format: str = "json"
    ) -> bool:
        """导出患者状态数据"""
        try:
            if format.lower() == "json":
                data = [asdict(patient) for patient in patient_states]
                # 处理datetime对象
                for item in data:
                    if "timestamp" in item and isinstance(item["timestamp"], datetime):
                        item["timestamp"] = item["timestamp"].isoformat()

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            elif format.lower() == "csv":
                # 将PatientState对象转换为平铺的字典
                flattened_data = []
                for patient in patient_states:
                    flat_dict = {
                        "patient_id": patient.patient_id,
                        "age": patient.age,
                        "diagnosis": patient.diagnosis,
                        "stage": patient.stage,
                        "psychological_status": patient.psychological_status,
                        "quality_of_life_score": patient.quality_of_life_score,
                        "symptoms": ",".join(patient.symptoms),
                        "comorbidities": ",".join(patient.comorbidities),
                        "timestamp": patient.timestamp.isoformat(),
                    }

                    # 添加实验室结果
                    for key, value in patient.lab_results.items():
                        flat_dict[f"lab_{key}"] = value

                    # 添加生命体征
                    for key, value in patient.vital_signs.items():
                        flat_dict[f"vital_{key}"] = value

                    flattened_data.append(flat_dict)

                df = pd.DataFrame(flattened_data)
                df.to_csv(filepath, index=False, encoding="utf-8")

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.export_history.append(
                {
                    "filepath": filepath,
                    "format": format,
                    "count": len(patient_states),
                    "timestamp": datetime.now(),
                }
            )

            logger.info(
                f"Successfully exported {len(patient_states)} patient records to {filepath}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to export patient states: {e}")
            return False

    def export_consensus_results(
        self, results: List[Dict[str, Any]], filepath: str
    ) -> bool:
        """导出共识结果"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Exported {len(results)} consensus results to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export consensus results: {e}")
            return False


class DataPipeline:
    """数据处理管道"""

    def __init__(self, config_path: Optional[str] = None):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.exporter = DataExporter()

        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    return yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    return json.load(f)
                else:
                    raise ValueError("Unsupported config file format")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def process_raw_data(self, input_file: str, output_file: str) -> bool:
        """处理原始数据的完整管道"""
        try:
            logger.info(
                f"Starting data processing pipeline: {input_file} -> {output_file}"
            )

            # 1. 加载数据
            if input_file.endswith(".csv"):
                df = pd.read_csv(input_file)
            elif input_file.endswith(".xlsx"):
                df = pd.read_excel(input_file)
            else:
                raise ValueError(f"Unsupported input file format: {input_file}")

            logger.info(f"Loaded {len(df)} records from {input_file}")

            # 2. 清洗数据
            df_cleaned = self.cleaner.clean_patient_data(df)

            # 3. 验证数据
            validation_errors = []
            for _, row in df_cleaned.iterrows():
                is_valid, errors = self.validator.validate_patient_state(row.to_dict())
                if not is_valid:
                    validation_errors.extend(errors)

            if validation_errors:
                logger.warning(f"Found {len(validation_errors)} validation errors")
                for error in validation_errors[:10]:  # 只显示前10个错误
                    logger.warning(f"  {error}")

            # 4. 转换数据
            patient_states = self.transformer.transform_to_patient_states(df_cleaned)

            # 5. 导出数据
            export_format = "json" if output_file.endswith(".json") else "csv"
            success = self.exporter.export_patient_states(
                patient_states, output_file, export_format
            )

            if success:
                logger.info("Data processing pipeline completed successfully")
                return True
            else:
                logger.error("Data processing pipeline failed at export stage")
                return False

        except Exception as e:
            logger.error(f"Data processing pipeline failed: {e}")
            return False

    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        return {
            "cleaning_stats": self.cleaner.cleaning_stats,
            "transformation_history": self.transformer.transformation_history,
            "export_history": self.exporter.export_history,
            "timestamp": datetime.now(),
        }


def main():
    """主函数演示"""
    # 创建数据处理管道
    pipeline = DataPipeline()

    # 创建示例数据
    sample_data = pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [65, 45, 78],
            "diagnosis": ["breast_cancer", "breast_cancer", "breast_cancer"],
            "stage": ["II", "I", "III"],
            "creatinine": [1.2, 0.9, 1.8],
            "hemoglobin": [11.5, 12.8, 9.2],
            "bp_systolic": [140, 120, 160],
            "heart_rate": [78, 72, 85],
            "symptoms": [
                "fatigue,pain",
                "mild_fatigue",
                "fatigue,pain,shortness_of_breath",
            ],
            "comorbidities": [
                "diabetes,hypertension",
                "",
                "diabetes,hypertension,cardiac_dysfunction",
            ],
            "psychological_status": ["anxious", "stable", "depressed"],
            "quality_of_life_score": [0.7, 0.85, 0.4],
        }
    )

    # 保存示例数据
    sample_data.to_csv("data/examples/sample_raw_data.csv", index=False)

    # 处理数据
    success = pipeline.process_raw_data(
        "data/examples/sample_raw_data.csv", "data/processed/processed_patients.json"
    )

    if success:
        print("✅ 数据处理完成")
        summary = pipeline.get_processing_summary()
        print(f"处理摘要: {json.dumps(summary, indent=2, default=str)}")
    else:
        print("❌ 数据处理失败")


if __name__ == "__main__":
    main()
