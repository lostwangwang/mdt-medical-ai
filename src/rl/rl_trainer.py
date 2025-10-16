"""
强化学习训练器
整合Q-learning和PPO算法，提供统一的训练接口
作者: AI Assistant
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
from datetime import datetime
import pickle

from .patient_rl_optimizer import PatientRLEnvironment
from .q_learning_agent import QLearningAgent
from .ppo_agent import PPOAgent
from .rl_visualizer import RLVisualizer

logger = logging.getLogger(__name__)

class RLTrainer:
    """强化学习训练器"""
    
    def __init__(self, patient_data_path: str, config: Optional[Dict] = None):
        """初始化训练器"""
        self.patient_data_path = patient_data_path
        self.config = config or self._get_default_config()
        
        # 初始化环境
        self.env = PatientRLEnvironment(patient_data_path)
        
        # 初始化可视化器
        self.visualizer = RLVisualizer()
        
        # 训练历史
        self.training_history = {
            'q_learning': {'episodes': [], 'rewards': [], 'training_results': []},
            'ppo': {'episodes': [], 'rewards': [], 'training_results': []}
        }
        
        logger.info(f"RL训练器初始化完成，患者数据: {patient_data_path}")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'q_learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'episodes': 1000
            },
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'eps_clip': 0.2,
                'k_epochs': 4,
                'episodes': 1000,
                'update_timestep': 2000
            },
            'visualization': {
                'save_plots': True,
                'plot_interval': 100,
                'output_dir': 'rl_results'
            }
        }
    
    def train_q_learning(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """训练Q-learning算法"""
        episodes = episodes or self.config['q_learning']['episodes']
        
        logger.info(f"开始Q-learning训练，episodes: {episodes}")
        
        # 初始化Q-learning智能体
        agent = QLearningAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            learning_rate=self.config['q_learning']['learning_rate'],
            discount_factor=self.config['q_learning']['discount_factor'],
            epsilon=self.config['q_learning']['epsilon'],
            epsilon_decay=self.config['q_learning']['epsilon_decay'],
            epsilon_min=self.config['q_learning']['epsilon_min']
        )
        
        # 训练
        training_output = agent.train(self.env, episodes)
        training_results = training_output['training_results']
        
        # 保存训练历史
        self.training_history['q_learning'] = {
            'episodes': list(range(1, episodes + 1)),
            'rewards': [result['total_reward'] for result in training_results],
            'training_results': training_results
        }
        
        # 计算最终统计
        rewards = self.training_history['q_learning']['rewards']
        final_stats = {
            'algorithm': 'Q-Learning',
            'total_episodes': episodes,
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'final_100_avg': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }
        
        logger.info(f"Q-learning训练完成，平均奖励: {final_stats['average_reward']:.4f}")
        
        return {
            'training_history': self.training_history['q_learning'],
            'final_stats': final_stats,
            'agent': agent
        }
    
    def train_ppo(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """训练PPO算法"""
        episodes = episodes or self.config['ppo']['episodes']
        
        logger.info(f"开始PPO训练，episodes: {episodes}")
        
        # 初始化PPO智能体
        agent = PPOAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            learning_rate=self.config['ppo']['learning_rate'],
            gamma=self.config['ppo']['gamma'],
            eps_clip=self.config['ppo']['eps_clip'],
            k_epochs=self.config['ppo']['k_epochs']
        )
        
        # 训练
        training_output = agent.train(
            self.env, 
            episodes
        )
        training_results = training_output['training_results']
        
        # 保存训练历史
        self.training_history['ppo'] = {
            'episodes': list(range(1, episodes + 1)),
            'rewards': [result['total_reward'] for result in training_results],
            'training_results': training_results
        }
        
        # 计算最终统计
        rewards = self.training_history['ppo']['rewards']
        final_stats = {
            'algorithm': 'PPO',
            'total_episodes': episodes,
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'final_100_avg': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        }
        
        logger.info(f"PPO训练完成，平均奖励: {final_stats['average_reward']:.4f}")
        
        return {
            'training_history': self.training_history['ppo'],
            'final_stats': final_stats,
            'agent': agent
        }
    
    def compare_algorithms(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """比较不同算法的性能"""
        logger.info("开始算法性能比较")
        
        # 训练Q-learning
        q_results = self.train_q_learning(episodes)
        
        # 训练PPO
        ppo_results = self.train_ppo(episodes)
        
        # 比较结果
        comparison_results = {
            'Q-Learning': q_results,
            'PPO': ppo_results
        }
        
        # 生成比较报告
        report = self._generate_comparison_report(comparison_results)
        
        logger.info("算法比较完成")
        
        return {
            'results': comparison_results,
            'report': report
        }
    
    def _generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成比较报告"""
        report = {
            'summary': {},
            'detailed_comparison': {},
            'recommendations': []
        }
        
        algorithms = list(results.keys())
        
        # 汇总统计
        for alg in algorithms:
            stats = results[alg]['final_stats']
            report['summary'][alg] = {
                '平均奖励': f"{stats['average_reward']:.4f}",
                '最大奖励': f"{stats['max_reward']:.4f}",
                '最小奖励': f"{stats['min_reward']:.4f}",
                '标准差': f"{stats['std_reward']:.4f}",
                '最后100期平均': f"{stats['final_100_avg']:.4f}"
            }
        
        # 详细比较
        if len(algorithms) >= 2:
            alg1, alg2 = algorithms[0], algorithms[1]
            stats1 = results[alg1]['final_stats']
            stats2 = results[alg2]['final_stats']
            
            report['detailed_comparison'] = {
                '平均奖励差异': stats1['average_reward'] - stats2['average_reward'],
                '稳定性比较': {
                    alg1: stats1['std_reward'],
                    alg2: stats2['std_reward'],
                    '更稳定的算法': alg1 if stats1['std_reward'] < stats2['std_reward'] else alg2
                },
                '收敛性比较': {
                    alg1: stats1['final_100_avg'],
                    alg2: stats2['final_100_avg'],
                    '收敛更好的算法': alg1 if stats1['final_100_avg'] > stats2['final_100_avg'] else alg2
                }
            }
            
            # 推荐
            if stats1['average_reward'] > stats2['average_reward']:
                if stats1['std_reward'] < stats2['std_reward']:
                    report['recommendations'].append(f"{alg1}在平均性能和稳定性方面都更优")
                else:
                    report['recommendations'].append(f"{alg1}平均性能更好，但{alg2}更稳定")
            else:
                if stats2['std_reward'] < stats1['std_reward']:
                    report['recommendations'].append(f"{alg2}在平均性能和稳定性方面都更优")
                else:
                    report['recommendations'].append(f"{alg2}平均性能更好，但{alg1}更稳定")
        
        return report
    
    def evaluate_agent(self, agent: Union[QLearningAgent, PPOAgent], 
                      episodes: int = 100) -> Dict[str, Any]:
        """评估训练好的智能体"""
        logger.info(f"开始评估智能体，episodes: {episodes}")
        
        evaluation_results = []
        total_rewards = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_actions = []
            done = False
            
            while not done:
                if isinstance(agent, QLearningAgent):
                    action = agent.choose_action(state, training=False)
                else:  # PPO
                    action = agent.select_action(state, training=False)
                
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_actions.append(action)
                state = next_state
            
            total_rewards.append(episode_reward)
            evaluation_results.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'actions': episode_actions,
                'info': info
            })
        
        # 计算评估统计
        eval_stats = {
            'total_episodes': episodes,
            'average_reward': np.mean(total_rewards),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': len([r for r in total_rewards if r > 0]) / len(total_rewards)
        }
        
        logger.info(f"评估完成，平均奖励: {eval_stats['average_reward']:.4f}")
        
        return {
            'evaluation_results': evaluation_results,
            'eval_stats': eval_stats
        }
    
    def generate_learning_curves(self, results: Dict[str, Any], 
                               output_dir: Optional[str] = None) -> Dict[str, str]:
        """生成学习曲线"""
        output_dir = output_dir or self.config['visualization']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        saved_plots = {}
        
        # 单算法学习曲线
        for alg_name, alg_results in results.items():
            if 'training_history' in alg_results:
                plot_path = os.path.join(output_dir, f'{alg_name.lower()}_learning_curve.png')
                self.visualizer.plot_learning_curve(
                    alg_results['training_history'],
                    save_path=plot_path
                )
                saved_plots[f'{alg_name}_learning_curve'] = plot_path
        
        # 算法比较图
        if len(results) > 1:
            comparison_path = os.path.join(output_dir, 'algorithm_comparison.png')
            self.visualizer.plot_algorithm_comparison(
                results,
                save_path=comparison_path
            )
            saved_plots['algorithm_comparison'] = comparison_path
        
        # 动作分析
        for alg_name, alg_results in results.items():
            if 'training_results' in alg_results['training_history']:
                action_names = ['保守治疗', '药物强化', '介入治疗', '手术治疗', '综合治疗', '紧急处理']
                action_path = os.path.join(output_dir, f'{alg_name.lower()}_action_analysis.png')
                self.visualizer.plot_action_analysis(
                    alg_results['training_history']['training_results'],
                    action_names,
                    save_path=action_path
                )
                saved_plots[f'{alg_name}_action_analysis'] = action_path
        
        # 奖励组件分析
        for alg_name, alg_results in results.items():
            if 'training_results' in alg_results['training_history']:
                reward_path = os.path.join(output_dir, f'{alg_name.lower()}_reward_components.png')
                self.visualizer.plot_reward_components(
                    alg_results['training_history']['training_results'],
                    save_path=reward_path
                )
                saved_plots[f'{alg_name}_reward_components'] = reward_path
        
        logger.info(f"学习曲线已生成，保存在: {output_dir}")
        
        return saved_plots
    
    def save_results(self, results: Dict[str, Any], 
                    output_dir: Optional[str] = None) -> str:
        """保存训练结果"""
        output_dir = output_dir or self.config['visualization']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        json_path = os.path.join(output_dir, f'rl_results_{timestamp}.json')
        
        # 准备可序列化的数据
        serializable_results = {}
        for alg_name, alg_results in results.items():
            serializable_results[alg_name] = {
                'final_stats': alg_results['final_stats'],
                'training_history': {
                    'episodes': alg_results['training_history']['episodes'],
                    'rewards': alg_results['training_history']['rewards']
                }
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存完整结果（包括智能体）
        pickle_path = os.path.join(output_dir, f'rl_complete_results_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"结果已保存: {json_path}, {pickle_path}")
        
        return output_dir
    
    def run_complete_experiment(self, episodes: Optional[int] = None) -> Dict[str, Any]:
        """运行完整的强化学习实验"""
        logger.info("开始完整的强化学习实验")
        
        # 比较算法
        comparison_results = self.compare_algorithms(episodes)
        
        # 生成可视化
        plot_paths = self.generate_learning_curves(comparison_results['results'])
        
        # 保存结果
        output_dir = self.save_results(comparison_results['results'])
        
        # 生成实验报告
        experiment_report = {
            'experiment_info': {
                'patient_data': self.patient_data_path,
                'timestamp': datetime.now().isoformat(),
                'episodes_per_algorithm': episodes or self.config['q_learning']['episodes'],
                'algorithms_tested': list(comparison_results['results'].keys())
            },
            'results': comparison_results,
            'visualizations': plot_paths,
            'output_directory': output_dir
        }
        
        logger.info(f"完整实验完成，结果保存在: {output_dir}")
        
        return experiment_report