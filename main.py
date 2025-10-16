"""
MDT医疗智能体系统主程序入口
文件路径: main.py
作者: Tianyu (系统集成) / 姚刚 (共识与RL模块)
功能: 系统主入口，提供命令行界面和演示功能
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data_models import PatientState, TreatmentOption
from src.consensus.consensus_matrix import ConsensusMatrix
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.rl.rl_environment import MDTReinforcementLearning, RLTrainer
from src.integration.workflow_manager import IntegratedWorkflowManager
from src.utils.visualization import SystemVisualizer
from src.utils.system_optimizer import get_system_optimizer, optimized_function
from experiments.baseline_comparison import ComparisonExperiment

# 初始化系统优化器
system_optimizer = get_system_optimizer()

# 使用优化的日志系统
logger = system_optimizer.get_logger(__name__)


class MDTSystemInterface:
    """MDT系统主接口"""

    def __init__(self):
        # 初始化系统优化器
        self.system_optimizer = get_system_optimizer()
        self.logger = self.system_optimizer.get_logger(self.__class__.__name__)
        
        # 初始化系统组件
        self.rag_system = MedicalKnowledgeRAG()
        self.consensus_system = ConsensusMatrix()
        self.dialogue_manager = MultiAgentDialogueManager(self.rag_system)
        self.rl_environment = MDTReinforcementLearning(self.consensus_system)
        self.workflow_manager = IntegratedWorkflowManager()
        self.visualizer = SystemVisualizer()
        
        self.logger.info("MDT系统接口初始化完成")

        logger.info("MDT System initialized successfully")

    @optimized_function
    def run_single_patient_analysis(
        self, patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行单个患者的完整分析"""
        self.logger.info(
            f"Starting analysis for patient {patient_data.get('patient_id', 'unknown')}"
        )

        # 创建患者状态对象
        patient_state = self._create_patient_state(patient_data)

        # 运行多智能体对话与共识
        self.logger.info("Running multi-agent dialogue...")
        consensus_result = self.dialogue_manager.conduct_mdt_discussion(patient_state)

        # 生成可视化
        self.logger.info("Generating visualizations...")
        visualizations = self.visualizer.create_patient_analysis_dashboard(
            patient_state, consensus_result
        )

        # 整理结果
        analysis_result = {
            "patient_info": {
                "patient_id": patient_state.patient_id,
                "age": patient_state.age,
                "diagnosis": patient_state.diagnosis,
                "stage": patient_state.stage,
            },
            "consensus_result": {
                "recommended_treatment": max(
                    consensus_result.aggregated_scores.items(), key=lambda x: x[1]
                )[0].value,
                "consensus_score": max(consensus_result.aggregated_scores.values()),
                "total_rounds": consensus_result.total_rounds,
                "convergence_achieved": consensus_result.convergence_achieved,
                "conflicts": len(consensus_result.conflicts),
                "agreements": len(consensus_result.agreements),
            },
            "dialogue_transcript": self.dialogue_manager.get_dialogue_transcript(),
            "visualizations": visualizations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        self.logger.info("Single patient analysis completed successfully")
        return analysis_result

    @optimized_function
    def run_training_experiment(self, episodes: int = 1000) -> Dict[str, Any]:
        """运行RL训练实验"""
        self.logger.info(f"Starting RL training with {episodes} episodes")

        trainer = RLTrainer(self.rl_environment)
        training_results = trainer.train_dqn(episodes=episodes)

        # 生成训练可视化
        training_visualizations = self.visualizer.create_training_dashboard(
            training_results
        )

        result = {
            "training_results": training_results,
            "visualizations": training_visualizations,
            "final_metrics": self.rl_environment.get_training_metrics(),
        }

        logger.info("RL training experiment completed")
        return result

    def run_baseline_comparison(
        self, num_patients: int = 100, num_trials: int = 50
    ) -> Dict[str, Any]:
        """运行基线模型对比实验"""
        logger.info(
            f"Starting baseline comparison with {num_patients} patients, {num_trials} trials"
        )

        experiment = ComparisonExperiment()
        experiment.generate_test_patients(num_patients)
        results = experiment.run_comparison(num_trials)

        # 生成对比报告和可视化
        report = experiment.generate_comparison_report()
        experiment.plot_comparison_results("results/figures/baseline_comparison.png")

        comparison_result = {
            "comparison_results": results.to_dict("records"),
            "report": report,
            "visualization_saved": True,
        }

        logger.info("Baseline comparison completed")
        return comparison_result

    def run_integrated_simulation(
        self, patient_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """运行集成时序模拟"""
        logger.info(f"Starting integrated simulation for {patient_id}, {days} days")

        simulation_result = self.workflow_manager.run_temporal_simulation(
            patient_id, days
        )

        # 生成时序可视化
        temporal_visualizations = self.visualizer.create_temporal_analysis_dashboard(
            simulation_result
        )

        result = {
            "simulation_result": simulation_result,
            "visualizations": temporal_visualizations,
        }

        logger.info("Integrated simulation completed")
        return result

    def _create_patient_state(self, patient_data: Dict[str, Any]) -> PatientState:
        """从输入数据创建患者状态对象"""
        return PatientState(
            patient_id=patient_data.get("patient_id", "DEMO_001"),
            age=patient_data.get("age", 65),
            diagnosis=patient_data.get("diagnosis", "breast_cancer"),
            stage=patient_data.get("stage", "II"),
            lab_results=patient_data.get(
                "lab_results", {"creatinine": 1.2, "hemoglobin": 11.5}
            ),
            vital_signs=patient_data.get(
                "vital_signs", {"bp_systolic": 140, "heart_rate": 78}
            ),
            symptoms=patient_data.get("symptoms", ["fatigue", "pain"]),
            comorbidities=patient_data.get(
                "comorbidities", ["diabetes", "hypertension"]
            ),
            psychological_status=patient_data.get("psychological_status", "anxious"),
            quality_of_life_score=patient_data.get("quality_of_life_score", 0.7),
            timestamp=datetime.now(),
        )


def create_sample_patients() -> List[Dict[str, Any]]:
    """创建示例患者数据"""
    return [
        {
            "patient_id": "DEMO_001",
            "age": 65,
            "diagnosis": "breast_cancer",
            "stage": "II",
            "lab_results": {"creatinine": 1.2, "hemoglobin": 11.5},
            "vital_signs": {"bp_systolic": 140, "heart_rate": 78},
            "symptoms": ["fatigue", "pain"],
            "comorbidities": ["diabetes", "hypertension"],
            "psychological_status": "anxious",
            "quality_of_life_score": 0.7,
        },
        {
            "patient_id": "DEMO_002",
            "age": 45,
            "diagnosis": "breast_cancer",
            "stage": "I",
            "lab_results": {"creatinine": 0.9, "hemoglobin": 12.8},
            "vital_signs": {"bp_systolic": 120, "heart_rate": 72},
            "symptoms": ["mild_fatigue"],
            "comorbidities": [],
            "psychological_status": "stable",
            "quality_of_life_score": 0.85,
        },
        {
            "patient_id": "DEMO_003",
            "age": 78,
            "diagnosis": "breast_cancer",
            "stage": "III",
            "lab_results": {"creatinine": 1.8, "hemoglobin": 9.2},
            "vital_signs": {"bp_systolic": 160, "heart_rate": 85},
            "symptoms": ["fatigue", "pain", "shortness_of_breath"],
            "comorbidities": ["diabetes", "hypertension", "cardiac_dysfunction"],
            "psychological_status": "depressed",
            "quality_of_life_score": 0.4,
        },
    ]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MDT Medical AI System")

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "patient", "training", "comparison", "simulation"],
        help="运行模式",
    )

    parser.add_argument("--patient-file", type=str, help="患者数据文件路径 (JSON格式)")

    parser.add_argument("--episodes", type=int, default=1000, help="RL训练episode数量")

    parser.add_argument(
        "--num-patients", type=int, default=100, help="对比实验中的患者数量"
    )

    parser.add_argument("--num-trials", type=int, default=50, help="对比实验的试验次数")

    parser.add_argument("--simulation-days", type=int, default=30, help="时序模拟天数")

    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")

    parser.add_argument("--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/figures", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 初始化系统
    print("=== MDT医疗智能体系统 ===")
    print("初始化系统组件...")
    
    # 启动系统优化器
    print("启动系统优化器...")
    system_optimizer.initialize()
    logger.info("系统优化器已启动")

    system = MDTSystemInterface()

    print(f"运行模式: {args.mode}")

    if args.mode == "demo":
        print("\n=== 演示模式 ===")
        sample_patients = create_sample_patients()

        for i, patient_data in enumerate(sample_patients, 1):
            print(f"\n--- 分析患者 {i}: {patient_data['patient_id']} ---")
            result = system.run_single_patient_analysis(patient_data)

            print(
                f"推荐治疗方案: {result['consensus_result']['recommended_treatment']}"
            )
            print(f"共识得分: {result['consensus_result']['consensus_score']:.3f}")
            print(f"对话轮数: {result['consensus_result']['total_rounds']}")
            print(f"是否收敛: {result['consensus_result']['convergence_achieved']}")

            # 保存结果
            import json

            output_file = (
                f"{args.output_dir}/patient_{patient_data['patient_id']}_analysis.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                # 处理不可序列化的对象
                serializable_result = result.copy()
                serializable_result.pop("visualizations", None)  # 移除可视化对象
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)

            print(f"详细结果已保存到: {output_file}")

    elif args.mode == "patient":
        print("\n=== 单患者分析模式 ===")
        if not args.patient_file:
            print("错误: 请提供患者数据文件 (--patient-file)")
            return

        # 加载患者数据
        import json

        with open(args.patient_file, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

        result = system.run_single_patient_analysis(patient_data)

        print(f"患者 {patient_data['patient_id']} 分析完成")
        print(f"推荐治疗: {result['consensus_result']['recommended_treatment']}")

        # 保存结果
        output_file = (
            f"{args.output_dir}/patient_{patient_data['patient_id']}_analysis.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_file}")

    elif args.mode == "training":
        print(f"\n=== RL训练模式 ({args.episodes} episodes) ===")
        result = system.run_training_experiment(args.episodes)

        print("训练完成!")
        print(f"最终平均奖励: {result['final_metrics']['recent_average_reward']:.3f}")
        print(f"学习改进: {result['final_metrics']['improvement']:+.3f}")

        # 保存训练结果
        import json

        output_file = f"{args.output_dir}/training_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"训练结果已保存到: {output_file}")

    elif args.mode == "comparison":
        print(
            f"\n=== 基线对比模式 ({args.num_patients} 患者, {args.num_trials} 试验) ==="
        )
        result = system.run_baseline_comparison(args.num_patients, args.num_trials)

        print("对比实验完成!")
        print("\n" + result["report"])

        # 保存对比结果
        import json

        output_file = f"{args.output_dir}/comparison_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"对比结果已保存到: {output_file}")
        print(f"对比图表已保存到: {args.output_dir}/figures/baseline_comparison.png")

    elif args.mode == "simulation":
        print(f"\n=== 时序模拟模式 ({args.simulation_days} 天) ===")
        result = system.run_integrated_simulation("SIM_001", args.simulation_days)

        print("时序模拟完成!")
        print(f"总决策次数: {result['simulation_result']['total_decisions']}")
        print(f"平均共识得分: {result['simulation_result']['avg_consensus_score']:.3f}")

        # 保存模拟结果
        import json

        output_file = f"{args.output_dir}/simulation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            serializable_result = result.copy()
            serializable_result.pop("visualizations", None)
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)

        print(f"模拟结果已保存到: {output_file}")

    print(f"\n所有输出文件保存在: {args.output_dir}/")
    
    # 生成系统性能报告
    print("生成系统性能报告...")
    try:
        report_path = system_optimizer.generate_report(args.output_dir)
        print(f"系统性能报告已保存到: {report_path}")
    except Exception as e:
        logger.error(f"生成性能报告失败: {e}")
    
    # 关闭系统优化器
    print("关闭系统优化器...")
    system_optimizer.shutdown()
    
    print("系统运行完成!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，系统退出")
    except Exception as e:
        logger.error(f"系统运行出错: {e}", exc_info=True)
        print(f"系统运行出错: {e}")
        sys.exit(1)
