#!/usr/bin/env python3
"""
强化学习医疗决策优化演示
基于患者数据进行治疗策略的强化学习优化
作者: AI Assistant
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rl.rl_trainer import RLTrainer
from src.rl.patient_rl_optimizer import PatientRLEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_demo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """打印横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                  MDT医疗AI强化学习优化系统                    ║
    ║                Medical Decision Optimization with RL          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  环境定义: state = Memory Controller 输出的摘要              ║
    ║           action = 系统推荐的治疗策略                        ║
    ║           reward = 共识得分 + 稳定性指标                     ║
    ║                                                              ║
    ║  算法支持: Q-learning, PPO                                   ║
    ║  输出结果: 学习曲线, 性能对比, 策略分析                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_patient_data(patient_data_path: str) -> bool:
    """验证患者数据文件"""
    if not os.path.exists(patient_data_path):
        logger.error(f"患者数据文件不存在: {patient_data_path}")
        return False
    
    try:
        # 尝试创建环境来验证数据格式
        env = PatientRLEnvironment(patient_data_path)
        logger.info(f"患者数据验证成功: {patient_data_path}")
        logger.info(f"状态空间维度: {env.state_size}")
        logger.info(f"动作空间大小: {env.action_size}")
        return True
    except Exception as e:
        logger.error(f"患者数据验证失败: {e}")
        return False

def run_quick_demo(patient_data_path: str):
    """运行快速演示"""
    print("\n🚀 开始快速演示 (每个算法100个episodes)")
    print("=" * 60)
    
    # 配置
    config = {
        'q_learning': {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.99,
            'epsilon_min': 0.01,
            'episodes': 100
        },
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4,
            'episodes': 100,
            'update_timestep': 200
        },
        'visualization': {
            'save_plots': True,
            'output_dir': 'rl_results_quick'
        }
    }
    
    # 创建训练器
    trainer = RLTrainer(patient_data_path, config)
    
    # 运行实验
    results = trainer.run_complete_experiment(episodes=100)
    
    # 打印结果摘要
    print_results_summary(results)
    
    return results

def run_full_experiment(patient_data_path: str, episodes: int = 1000):
    """运行完整实验"""
    print(f"\n🔬 开始完整实验 (每个算法{episodes}个episodes)")
    print("=" * 60)
    
    # 创建训练器
    trainer = RLTrainer(patient_data_path)
    
    # 运行实验
    results = trainer.run_complete_experiment(episodes=episodes)
    
    # 打印结果摘要
    print_results_summary(results)
    
    return results

def run_algorithm_comparison(patient_data_path: str, episodes: int = 500):
    """运行算法比较"""
    print(f"\n⚖️  开始算法性能比较 (每个算法{episodes}个episodes)")
    print("=" * 60)
    
    # 创建训练器
    trainer = RLTrainer(patient_data_path)
    
    # 比较算法
    comparison_results = trainer.compare_algorithms(episodes=episodes)
    
    # 打印比较报告
    print_comparison_report(comparison_results['report'])
    
    # 生成可视化
    plot_paths = trainer.generate_learning_curves(comparison_results['results'])
    
    print(f"\n📊 可视化图表已生成:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  - {plot_name}: {plot_path}")
    
    return comparison_results

def print_results_summary(results: dict):
    """打印结果摘要"""
    print("\n📈 实验结果摘要")
    print("=" * 60)
    
    experiment_info = results.get('experiment_info', {})
    print(f"患者数据: {experiment_info.get('patient_data', 'N/A')}")
    print(f"实验时间: {experiment_info.get('timestamp', 'N/A')}")
    print(f"每算法Episodes: {experiment_info.get('episodes_per_algorithm', 'N/A')}")
    
    algorithm_results = results.get('results', {}).get('results', {})
    
    if algorithm_results:
        print(f"\n算法性能对比:")
        print("-" * 40)
        
        for alg_name, alg_data in algorithm_results.items():
            stats = alg_data.get('final_stats', {})
            print(f"\n{alg_name}:")
            print(f"  平均奖励: {stats.get('average_reward', 0):.4f}")
            print(f"  最大奖励: {stats.get('max_reward', 0):.4f}")
            print(f"  最小奖励: {stats.get('min_reward', 0):.4f}")
            print(f"  标准差:   {stats.get('std_reward', 0):.4f}")
            print(f"  最后100期平均: {stats.get('final_100_avg', 0):.4f}")
    
    output_dir = results.get('output_directory', 'N/A')
    print(f"\n💾 结果保存目录: {output_dir}")
    
    visualizations = results.get('visualizations', {})
    if visualizations:
        print(f"\n📊 生成的可视化图表:")
        for plot_name, plot_path in visualizations.items():
            print(f"  - {plot_name}: {plot_path}")

def print_comparison_report(report: dict):
    """打印比较报告"""
    print("\n📊 算法比较报告")
    print("=" * 60)
    
    # 摘要
    summary = report.get('summary', {})
    if summary:
        print("性能摘要:")
        print("-" * 30)
        for alg_name, stats in summary.items():
            print(f"\n{alg_name}:")
            for metric, value in stats.items():
                print(f"  {metric}: {value}")
    
    # 详细比较
    detailed = report.get('detailed_comparison', {})
    if detailed:
        print(f"\n详细比较:")
        print("-" * 30)
        
        if '平均奖励差异' in detailed:
            print(f"平均奖励差异: {detailed['平均奖励差异']:.4f}")
        
        if '稳定性比较' in detailed:
            stability = detailed['稳定性比较']
            print(f"稳定性比较: {stability.get('更稳定的算法', 'N/A')} 更稳定")
        
        if '收敛性比较' in detailed:
            convergence = detailed['收敛性比较']
            print(f"收敛性比较: {convergence.get('收敛更好的算法', 'N/A')} 收敛更好")
    
    # 推荐
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 推荐:")
        print("-" * 30)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

def interactive_mode():
    """交互模式"""
    print("\n🎮 进入交互模式")
    print("=" * 60)
    
    # 获取患者数据路径
    while True:
        patient_data_path = input("\n请输入患者数据文件路径: ").strip()
        if patient_data_path.lower() == 'quit':
            return
        
        if validate_patient_data(patient_data_path):
            break
        else:
            print("❌ 患者数据文件无效，请重新输入 (输入 'quit' 退出)")
    
    while True:
        print("\n请选择操作:")
        print("1. 快速演示 (100 episodes)")
        print("2. 完整实验 (1000 episodes)")
        print("3. 算法比较 (500 episodes)")
        print("4. 自定义实验")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        try:
            if choice == '1':
                run_quick_demo(patient_data_path)
            elif choice == '2':
                run_full_experiment(patient_data_path)
            elif choice == '3':
                run_algorithm_comparison(patient_data_path)
            elif choice == '4':
                episodes = int(input("请输入episodes数量: "))
                run_full_experiment(patient_data_path, episodes)
            elif choice == '5':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n\n⏹️  操作被用户中断")
        except Exception as e:
            logger.error(f"执行过程中出错: {e}")
            print(f"❌ 执行失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MDT医疗AI强化学习优化演示')
    parser.add_argument('--patient-data', type=str, 
                       default='data/10037928_clinical_memory.json',
                       help='患者数据文件路径')
    parser.add_argument('--mode', type=str, choices=['quick', 'full', 'compare', 'interactive'],
                       default='interactive', help='运行模式')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练episodes数量')
    parser.add_argument('--output-dir', type=str, default='rl_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 打印横幅
    print_banner()
    
    # 验证患者数据
    if args.mode != 'interactive':
        if not validate_patient_data(args.patient_data):
            logger.error("患者数据验证失败，退出程序")
            return
    
    try:
        if args.mode == 'quick':
            run_quick_demo(args.patient_data)
        elif args.mode == 'full':
            run_full_experiment(args.patient_data, args.episodes)
        elif args.mode == 'compare':
            run_algorithm_comparison(args.patient_data, args.episodes)
        elif args.mode == 'interactive':
            interactive_mode()
        
        print("\n✅ 程序执行完成!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 程序执行失败: {e}")

if __name__ == "__main__":
    main()