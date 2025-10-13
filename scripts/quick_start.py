#!/usr/bin/env python3
"""
MDT医疗智能体系统 - 快速启动脚本
文件路径: scripts/quick_start.py
作者: 团队共同维护
功能: 提供系统的快速启动和测试功能
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """设置运行环境"""
    print("🔧 设置运行环境...")

    # 创建必要的目录
    directories = [
        "logs",
        "results",
        "results/figures",
        "data/processed",
        "data/examples",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

    print("✅ 环境设置完成")


def create_sample_patient_file():
    """创建示例患者文件"""
    sample_patient = {
        "patient_id": "QUICKSTART_001",
        "age": 62,
        "diagnosis": "breast_cancer",
        "stage": "II",
        "lab_results": {"creatinine": 1.1, "hemoglobin": 12.3, "cea": 2.8},
        "vital_signs": {
            "bp_systolic": 135,
            "heart_rate": 76,
            "temperature": 36.8,
            "weight": 68.5,
        },
        "symptoms": ["mild_fatigue", "occasional_pain"],
        "comorbidities": ["hypertension"],
        "psychological_status": "stable",
        "quality_of_life_score": 0.75,
    }

    sample_file = "data/examples/sample_patient.json"
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(sample_patient, f, ensure_ascii=False, indent=2)

    print(f"✓ 创建示例患者文件: {sample_file}")
    return sample_file


def run_quick_demo():
    """运行快速演示"""
    print("\n🚀 运行快速演示...")
    print("=" * 60)

    try:
        # 导入核心模块
        from src.core.data_models import PatientState, TreatmentOption
        from src.consensus.consensus_matrix import ConsensusMatrix
        from src.knowledge.rag_system import MedicalKnowledgeRAG

        # 创建患者状态
        print("📋 创建患者状态...")
        patient = PatientState(
            patient_id="DEMO_QUICK",
            age=62,
            diagnosis="breast_cancer",
            stage="II",
            lab_results={"creatinine": 1.1, "hemoglobin": 12.3},
            vital_signs={"bp_systolic": 135, "heart_rate": 76},
            symptoms=["mild_fatigue"],
            comorbidities=["hypertension"],
            psychological_status="stable",
            quality_of_life_score=0.75,
            timestamp=datetime.now(),
        )
        print(f"✓ 患者: {patient.patient_id}, {patient.age}岁, {patient.diagnosis}")

        # 初始化系统组件
        print("\n🧠 初始化AI系统组件...")
        rag_system = MedicalKnowledgeRAG()
        consensus_system = ConsensusMatrix()

        # 生成共识结果
        print("\n🤝 进行医疗团队共识分析...")
        consensus_result = consensus_system.generate_consensus(patient)

        # 显示结果
        print("\n📊 分析结果:")
        print("-" * 40)

        # 治疗推荐排序
        sorted_treatments = sorted(
            consensus_result.aggregated_scores.items(), key=lambda x: x[1], reverse=True
        )

        print("治疗方案推荐 (按共识得分排序):")
        for i, (treatment, score) in enumerate(sorted_treatments, 1):
            status = (
                "✅ 推荐" if score > 0.5 else "⚠️ 谨慎" if score > 0 else "❌ 不推荐"
            )
            print(f"  {i}. {treatment.value:<18} : {score:+.3f} ({status})")

        # 团队共识分析
        print(f"\n团队共识分析:")
        print(f"  发现冲突: {len(consensus_result.conflicts)} 个")
        print(f"  一致意见: {len(consensus_result.agreements)} 个")

        if consensus_result.conflicts:
            print("  主要分歧:")
            for conflict in consensus_result.conflicts[:2]:
                print(
                    f"    - {conflict['treatment'].value}: 分歧程度 {conflict['variance']:.3f}"
                )

        if consensus_result.agreements:
            print("  强烈共识:")
            for agreement in consensus_result.agreements[:2]:
                print(
                    f"    - {agreement['treatment'].value}: 一致度 {agreement['agreement_strength']:.3f}"
                )

        # 最终建议
        best_treatment = sorted_treatments[0]
        print(f"\n🎯 系统推荐:")
        print(f"   推荐治疗: {best_treatment[0].value}")
        print(f"   共识得分: {best_treatment[1]:+.3f}")
        print(f"   推荐理由: 获得医疗团队综合评估最高分")

        # 保存结果
        result_file = "results/quick_demo_result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            demo_result = {
                "patient_info": {
                    "patient_id": patient.patient_id,
                    "age": patient.age,
                    "diagnosis": patient.diagnosis,
                    "stage": patient.stage,
                },
                "recommended_treatment": best_treatment[0].value,
                "consensus_score": best_treatment[1],
                "all_scores": {
                    t.value: s for t, s in consensus_result.aggregated_scores.items()
                },
                "conflicts_count": len(consensus_result.conflicts),
                "agreements_count": len(consensus_result.agreements),
                "timestamp": datetime.now().isoformat(),
            }
            json.dump(demo_result, f, ensure_ascii=False, indent=2)

        print(f"\n💾 详细结果已保存到: {result_file}")
        print("\n✅ 快速演示完成!")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False

    return True


def run_system_check():
    """运行系统检查"""
    print("\n🔍 系统环境检查...")
    print("=" * 60)

    # 检查Python版本
    python_version = sys.version_info
    print(
        f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version < (3, 10):
        print("⚠️ 警告: 推荐Python 3.10+以获得最佳性能")
    else:
        print("✅ Python版本符合要求")

    # 检查关键依赖
    critical_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "plotly",
    ]

    missing_packages = []

    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ 缺少关键依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    # 检查项目结构
    expected_dirs = ["src", "experiments", "data", "logs", "results"]
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"✅ 目录 {directory}")
        else:
            print(f"⚠️ 目录 {directory} 不存在")

    print("\n✅ 系统检查完成")
    return True


def interactive_mode():
    """交互式模式"""
    print("\n🎮 进入交互式模式")
    print("=" * 60)

    while True:
        print("\n选择操作:")
        print("1. 运行快速演示")
        print("2. 系统环境检查")
        print("3. 创建示例患者文件")
        print("4. 运行完整系统测试")
        print("5. 查看帮助")
        print("0. 退出")

        choice = input("\n请选择 (0-5): ").strip()

        if choice == "0":
            print("👋 再见!")
            break
        elif choice == "1":
            run_quick_demo()
        elif choice == "2":
            run_system_check()
        elif choice == "3":
            create_sample_patient_file()
        elif choice == "4":
            run_full_system_test()
        elif choice == "5":
            show_help()
        else:
            print("❌ 无效选择，请重试")


def run_full_system_test():
    """运行完整系统测试"""
    print("\n🧪 运行完整系统测试...")
    print("=" * 60)

    try:
        # 这里可以调用main.py的演示模式
        import subprocess

        result = subprocess.run(
            [sys.executable, "main.py", "--mode", "demo"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ 完整系统测试通过")
            print(result.stdout)
        else:
            print("❌ 系统测试失败")
            print(result.stderr)

    except Exception as e:
        print(f"❌ 测试运行错误: {e}")


def show_help():
    """显示帮助信息"""
    help_text = """
📚 MDT医疗智能体系统 - 帮助信息

🚀 快速开始:
  python scripts/quick_start.py --demo     # 运行快速演示
  python scripts/quick_start.py --check    # 系统检查
  python scripts/quick_start.py -i         # 交互式模式

📋 主要功能:
  • 多智能体医疗决策对话
  • 共识矩阵分析
  • 强化学习优化
  • 丰富的可视化

🔧 完整系统使用:
  python main.py --mode demo              # 演示模式
  python main.py --mode patient --patient-file data.json
  python main.py --mode training --episodes 1000
  python main.py --mode comparison --num-patients 100

📞 获取帮助:
  • 查看README.md了解详细说明
  • 访问项目文档: https://docs.mdt-medical-ai.com
  • 反馈问题: https://github.com/your-team/mdt-medical-ai/issues

💡 提示:
  • 首次使用建议先运行系统检查
  • 确保已安装所有依赖包
  • 查看results/目录获取输出结果
"""
    print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MDT医疗智能体系统 - 快速启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/quick_start.py --demo     # 运行快速演示
  python scripts/quick_start.py --check    # 系统检查
  python scripts/quick_start.py -i         # 交互式模式
        """,
    )

    parser.add_argument("--demo", action="store_true", help="运行快速演示")
    parser.add_argument("--check", action="store_true", help="运行系统检查")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="进入交互式模式"
    )
    parser.add_argument("--setup", action="store_true", help="设置运行环境")

    args = parser.parse_args()

    print("🏥 MDT医疗智能体系统 - 快速启动")
    print("=" * 60)

    # 总是先设置环境
    setup_environment()

    if args.setup:
        print("✅ 环境设置完成")
        return

    if args.check:
        success = run_system_check()
        if not success:
            print("\n❌ 系统检查未通过，请解决上述问题后重试")
            return

    if args.demo:
        success = run_quick_demo()
        if not success:
            print("\n❌ 演示运行失败，请检查系统配置")
            return

    if args.interactive:
        interactive_mode()
        return

    # 如果没有指定参数，显示帮助并进入交互模式
    if not any([args.demo, args.check, args.interactive, args.setup]):
        print("\n欢迎使用MDT医疗智能体系统!")
        print("这是一个快速启动脚本，帮助您开始使用系统。")
        show_help()

        print("\n是否进入交互式模式? (y/n): ", end="")
        if input().lower().startswith("y"):
            interactive_mode()
        else:
            print("运行 'python scripts/quick_start.py --help' 查看更多选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()
