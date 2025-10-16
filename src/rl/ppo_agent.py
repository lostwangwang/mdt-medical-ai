"""
PPO (Proximal Policy Optimization) 智能体实现
用于医疗决策优化的PPO算法
作者: AI Assistant
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 64]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        logits = self.network(state)
        # 使用数值稳定的softmax
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_size: int, hidden_sizes: List[int] = [128, 64]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # 输出层 (单个值)
        layers.append(nn.Linear(input_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    """PPO智能体"""
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 hidden_sizes: List[int] = [128, 64]):
        """
        初始化PPO智能体
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率
            gamma: 折扣因子
            eps_clip: PPO裁剪参数
            k_epochs: 每次更新的训练轮数
            hidden_sizes: 隐藏层大小
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 检查CUDA可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化网络
        self.policy_net = PolicyNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.value_net = ValueNetwork(state_size, hidden_sizes).to(self.device)
        
        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        # 训练历史
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }
        
        logger.info(f"PPO智能体初始化完成: 状态空间={state_size}, 动作空间={action_size}")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        获取动作
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        
        if training:
            # 训练时从概率分布中采样
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # 评估时选择概率最大的动作
            action = torch.argmax(action_probs, dim=1)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        log_prob: float, value: float, done: bool):
        """存储转换"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
    
    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算回报和优势"""
        rewards = self.memory['rewards']
        values = self.memory['values']
        dones = self.memory['dones']
        
        returns = []
        advantages = []
        
        # 计算折扣回报
        discounted_return = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_return = 0
            discounted_return = rewards[i] + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # 计算优势 (GAE简化版本)
        advantages = returns - values
        
        # 标准化优势 (增强数值稳定性)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        return returns, advantages
    
    def update_policy(self) -> Dict[str, float]:
        """更新策略"""
        if len(self.memory['states']) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        
        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages()
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.k_epochs):
            # 前向传播
            action_probs = self.policy_net(states)
            
            # 数值稳定性检查
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            values = self.value_net(states).squeeze()
            
            # 计算新的对数概率
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失 (确保维度匹配)
            if values.dim() != returns.dim():
                values = values.squeeze()
                returns = returns.squeeze()
            value_loss = F.mse_loss(values, returns)
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # 反向传播
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # 清空缓冲区
        self.clear_memory()
        
        return {
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs
        }
    
    def clear_memory(self):
        """清空经验缓冲区"""
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
    
    def train_episode(self, env, episode: int) -> Dict[str, Any]:
        """训练一个episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_actions = []
        
        while True:
            # 获取动作
            action, log_prob, value = self.get_action(state, training=True)
            episode_actions.append(action)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转换
            self.store_transition(state, action, reward, log_prob, value, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 更新策略
        update_info = self.update_policy()
        
        # 记录训练历史
        self.training_history['episodes'].append(episode)
        self.training_history['rewards'].append(total_reward)
        self.training_history['policy_losses'].append(update_info['policy_loss'])
        self.training_history['value_losses'].append(update_info['value_loss'])
        self.training_history['entropies'].append(update_info['entropy'])
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'policy_loss': update_info['policy_loss'],
            'value_loss': update_info['value_loss'],
            'entropy': update_info['entropy'],
            'actions': episode_actions,
            'info': info
        }
    
    def train(self, env, episodes: int = 1000) -> Dict[str, Any]:
        """训练智能体"""
        logger.info(f"开始PPO训练，共{episodes}个episodes")
        
        training_results = []
        
        for episode in range(episodes):
            episode_result = self.train_episode(env, episode)
            training_results.append(episode_result)
            
            # 每100个episode打印一次进度
            if (episode + 1) % 100 == 0:
                recent_rewards = [r['total_reward'] for r in training_results[-100:]]
                avg_reward = np.mean(recent_rewards)
                avg_policy_loss = np.mean([r['policy_loss'] for r in training_results[-100:]])
                avg_value_loss = np.mean([r['value_loss'] for r in training_results[-100:]])
                
                logger.info(f"Episode {episode + 1}/{episodes}, "
                          f"平均奖励: {avg_reward:.3f}, "
                          f"策略损失: {avg_policy_loss:.3f}, "
                          f"价值损失: {avg_value_loss:.3f}")
        
        # 计算最终统计
        final_stats = self.calculate_training_stats()
        
        logger.info("PPO训练完成")
        
        return {
            'training_results': training_results,
            'final_stats': final_stats,
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
            'reward_improvement': np.mean(rewards[-100:]) - np.mean(rewards[:100]) if len(rewards) >= 200 else 0,
            'final_policy_loss': self.training_history['policy_losses'][-1] if self.training_history['policy_losses'] else 0,
            'final_value_loss': self.training_history['value_losses'][-1] if self.training_history['value_losses'] else 0
        }
    
    def evaluate(self, env, episodes: int = 100) -> Dict[str, Any]:
        """评估智能体性能"""
        logger.info(f"开始评估PPO智能体，共{episodes}个episodes")
        
        evaluation_rewards = []
        evaluation_actions = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_actions = []
            
            while True:
                # 使用确定性策略（不探索）
                action, _, _ = self.get_action(state, training=False)
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
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_history': self.training_history,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'gamma': self.gamma,
                'eps_clip': self.eps_clip,
                'k_epochs': self.k_epochs
            }
        }, filepath)
        
        logger.info(f"PPO模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {
            'episodes': [], 'rewards': [], 'policy_losses': [], 'value_losses': [], 'entropies': []
        })
        
        logger.info(f"PPO模型已从{filepath}加载")
    
    def get_action_probabilities(self, state: np.ndarray) -> Dict[int, float]:
        """获取动作概率分布"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
        
        return {i: prob.item() for i, prob in enumerate(action_probs.squeeze())}