"""
Q-Learning智能体实现
用于医疗决策优化的Q-learning算法
作者: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import pickle
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class QLearningAgent:
    """Q-Learning智能体"""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        初始化Q-Learning智能体
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 初始化Q表 (使用状态离散化)
        # 为了避免维度爆炸，使用较少的离散化箱数
        if state_size > 10:
            self.state_bins = 3  # 高维状态空间使用3个箱
        elif state_size > 5:
            self.state_bins = 5  # 中等维度使用5个箱
        else:
            self.state_bins = 10  # 低维度可以使用更多箱
        
        # 使用字典存储Q表以节省内存（稀疏表示）
        self.q_table = {}
        self.default_q_value = 0.0
        
        # 训练历史
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'epsilons': [],
            'q_values': [],
            'actions': []
        }
        
        logger.info(f"Q-Learning智能体初始化完成: 状态空间={state_size}, 动作空间={action_size}")
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """将连续状态离散化"""
        # 将0-1范围的状态值映射到离散箱
        discrete_state = []
        for i, value in enumerate(state):
            # 确保值在0-1范围内
            clipped_value = np.clip(value, 0.0, 1.0)
            # 离散化到箱中
            bin_index = int(clipped_value * (self.state_bins - 1))
            bin_index = min(bin_index, self.state_bins - 1)  # 防止越界
            discrete_state.append(bin_index)
        
        return tuple(discrete_state)
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """获取Q值"""
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        return self.q_table[discrete_state][action]
    
    def get_max_q_value(self, state: np.ndarray) -> float:
        """获取状态的最大Q值"""
        discrete_state = self.discretize_state(state)
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        return np.max(self.q_table[discrete_state])
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作 (epsilon-greedy策略)"""
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            action = np.random.randint(0, self.action_size)
        else:
            # 利用：选择Q值最大的动作
            discrete_state = self.discretize_state(state)
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.zeros(self.action_size)
            q_values = self.q_table[discrete_state]
            action = np.argmax(q_values)
        
        return action
    
    def update_q_table(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """更新Q表"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # 确保状态存在于Q表中
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_size)
        
        # 当前Q值
        current_q = self.q_table[discrete_state][action]
        
        # 下一状态的最大Q值
        next_max_q = np.max(self.q_table[discrete_next_state])
        
        # Q-learning更新公式
        target_q = reward + self.discount_factor * next_max_q
        new_q = current_q + self.learning_rate * (target_q - current_q)
        
        # 更新Q表
        self.q_table[discrete_state][action] = new_q
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, env, episode: int) -> Dict[str, Any]:
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_actions = []
        episode_q_values = []
        
        while True:
            # 选择动作
            action = self.choose_action(state, training=True)
            episode_actions.append(action)
            
            # 记录Q值
            q_value = self.get_max_q_value(state)
            episode_q_values.append(q_value)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新Q表
            self.update_q_table(state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 记录训练历史
        self.training_history['episodes'].append(episode)
        self.training_history['rewards'].append(total_reward)
        self.training_history['epsilons'].append(self.epsilon)
        self.training_history['q_values'].append(np.mean(episode_q_values) if episode_q_values else 0)
        self.training_history['actions'].append(episode_actions)
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'average_q_value': np.mean(episode_q_values) if episode_q_values else 0,
            'actions': episode_actions,
            'info': info
        }
    
    def train(self, env, episodes: int = 1000) -> Dict[str, Any]:
        """训练智能体"""
        logger.info(f"开始Q-Learning训练，共{episodes}个episodes")
        
        training_results = []
        
        for episode in range(episodes):
            episode_result = self.train_episode(env, episode)
            training_results.append(episode_result)
            
            # 每100个episode打印一次进度
            if (episode + 1) % 100 == 0:
                recent_rewards = [r['total_reward'] for r in training_results[-100:]]
                avg_reward = np.mean(recent_rewards)
                logger.info(f"Episode {episode + 1}/{episodes}, "
                          f"平均奖励: {avg_reward:.3f}, "
                          f"探索率: {self.epsilon:.3f}")
        
        # 计算最终统计
        final_stats = self.calculate_training_stats()
        
        logger.info("Q-Learning训练完成")
        
        return {
            'training_results': training_results,
            'final_stats': final_stats,
            'q_table_stats': self.get_q_table_stats(),
            'training_history': self.training_history
        }
    
    def calculate_training_stats(self) -> Dict[str, float]:
        """计算训练统计信息"""
        rewards = self.training_history['rewards']
        
        if not rewards:
            return {}
        
        return {
            'total_episodes': len(rewards),
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'std_reward': np.std(rewards),
            'final_epsilon': self.epsilon,
            'reward_improvement': np.mean(rewards[-100:]) - np.mean(rewards[:100]) if len(rewards) >= 200 else 0
        }
    
    def get_q_table_stats(self) -> Dict[str, Any]:
        """获取Q表统计信息"""
        if not self.q_table:
            return {
                'total_states': 0,
                'explored_states': 0,
                'exploration_ratio': 0.0,
                'average_q_value': 0.0,
                'max_q_value': 0.0,
                'min_q_value': 0.0
            }
        
        # 收集所有Q值
        all_q_values = []
        for state_actions in self.q_table.values():
            all_q_values.extend(state_actions)
        
        all_q_values = np.array(all_q_values)
        non_zero_q = all_q_values[all_q_values != 0]
        
        return {
            'total_states': len(self.q_table),
            'explored_states': len(self.q_table),
            'exploration_ratio': 1.0,  # 字典中只存储访问过的状态
            'average_q_value': np.mean(non_zero_q) if len(non_zero_q) > 0 else 0,
            'max_q_value': np.max(all_q_values) if len(all_q_values) > 0 else 0,
            'min_q_value': np.min(all_q_values) if len(all_q_values) > 0 else 0
        }
    
    def evaluate(self, env, episodes: int = 100) -> Dict[str, Any]:
        """评估智能体性能"""
        logger.info(f"开始评估智能体，共{episodes}个episodes")
        
        evaluation_rewards = []
        evaluation_actions = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_actions = []
            
            while True:
                # 使用贪婪策略（不探索）
                action = self.choose_action(state, training=False)
                episode_actions.append(action)
                
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            evaluation_rewards.append(total_reward)
            evaluation_actions.append(episode_actions)
        
        return {
            'average_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards),
            'max_reward': np.max(evaluation_rewards),
            'min_reward': np.min(evaluation_rewards),
            'all_rewards': evaluation_rewards,
            'action_distribution': self._analyze_action_distribution(evaluation_actions)
        }
    
    def _analyze_action_distribution(self, all_actions: List[List[int]]) -> Dict[int, float]:
        """分析动作分布"""
        action_counts = {}
        total_actions = 0
        
        for episode_actions in all_actions:
            for action in episode_actions:
                action_counts[action] = action_counts.get(action, 0) + 1
                total_actions += 1
        
        # 计算比例
        action_distribution = {}
        for action in range(self.action_size):
            action_distribution[action] = action_counts.get(action, 0) / total_actions if total_actions > 0 else 0
        
        return action_distribution
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_table': self.q_table,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'state_bins': self.state_bins,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.state_size = model_data['state_size']
        self.action_size = model_data['action_size']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.epsilon_min = model_data['epsilon_min']
        self.epsilon_decay = model_data['epsilon_decay']
        self.state_bins = model_data['state_bins']
        self.training_history = model_data.get('training_history', {
            'episodes': [], 'rewards': [], 'epsilons': [], 'q_values': [], 'actions': []
        })
        
        logger.info(f"模型已从{filepath}加载")
    
    def get_policy(self, state: np.ndarray) -> Dict[int, float]:
        """获取当前策略（各动作的概率）"""
        discrete_state = self.discretize_state(state)
        q_values = self.q_table[discrete_state]
        
        # 使用softmax转换为概率分布
        exp_q = np.exp(q_values - np.max(q_values))  # 数值稳定性
        probabilities = exp_q / np.sum(exp_q)
        
        return {i: prob for i, prob in enumerate(probabilities)}