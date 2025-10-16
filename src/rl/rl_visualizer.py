"""
强化学习可视化工具
生成学习曲线、奖励曲线和性能分析图表
作者: AI Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class RLVisualizer:
    """强化学习可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """初始化可视化器"""
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 设置样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("RL可视化器初始化完成")
    
    def plot_learning_curve(self, training_history: Dict[str, List], 
                           window_size: int = 100, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """绘制学习曲线"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('强化学习训练曲线', fontsize=16, fontweight='bold')
        
        episodes = training_history.get('episodes', [])
        rewards = training_history.get('rewards', [])
        
        if not episodes or not rewards:
            logger.warning("训练历史数据为空")
            return fig
        
        # 1. 奖励曲线
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='原始奖励')
        
        # 移动平均
        if len(rewards) >= window_size:
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'{window_size}期移动平均')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('奖励')
        ax1.set_title('奖励变化曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励分布
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(rewards):.3f}')
        ax2.set_xlabel('奖励值')
        ax2.set_ylabel('频次')
        ax2.set_title('奖励分布直方图')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 累积奖励
        ax3 = axes[1, 0]
        cumulative_rewards = np.cumsum(rewards)
        ax3.plot(episodes, cumulative_rewards, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('累积奖励')
        ax3.set_title('累积奖励曲线')
        ax3.grid(True, alpha=0.3)
        
        # 4. 奖励改进趋势
        ax4 = axes[1, 1]
        if len(rewards) >= 200:
            # 计算每100个episode的平均奖励
            chunk_size = 100
            chunk_rewards = []
            chunk_episodes = []
            
            for i in range(0, len(rewards) - chunk_size + 1, chunk_size):
                chunk_rewards.append(np.mean(rewards[i:i+chunk_size]))
                chunk_episodes.append(episodes[i+chunk_size-1])
            
            ax4.plot(chunk_episodes, chunk_rewards, marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel(f'平均奖励 (每{chunk_size}期)')
            ax4.set_title('学习进展趋势')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '数据不足\n(需要至少200个episodes)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('学习进展趋势')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"学习曲线已保存到: {save_path}")
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict[str, Dict], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """比较不同算法的性能"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('算法性能对比', fontsize=16, fontweight='bold')
        
        algorithms = list(results.keys())
        colors = self.colors[:len(algorithms)]
        
        # 1. 平均奖励对比
        ax1 = axes[0, 0]
        avg_rewards = [results[alg]['final_stats']['average_reward'] for alg in algorithms]
        bars1 = ax1.bar(algorithms, avg_rewards, color=colors, alpha=0.7)
        ax1.set_ylabel('平均奖励')
        ax1.set_title('平均奖励对比')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars1, avg_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 学习曲线对比
        ax2 = axes[0, 1]
        for i, alg in enumerate(algorithms):
            rewards = results[alg]['training_history']['rewards']
            episodes = results[alg]['training_history']['episodes']
            
            # 移动平均
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                ax2.plot(episodes, moving_avg, color=colors[i], linewidth=2, label=alg)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('奖励 (移动平均)')
        ax2.set_title('学习曲线对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 收敛速度对比
        ax3 = axes[1, 0]
        convergence_episodes = []
        for alg in algorithms:
            rewards = results[alg]['training_history']['rewards']
            # 简单的收敛检测：找到奖励稳定的点
            target_reward = np.mean(rewards[-100:]) * 0.9  # 90%的最终性能
            convergence_ep = len(rewards)  # 默认值
            
            window_size = 50
            for i in range(window_size, len(rewards)):
                if np.mean(rewards[i-window_size:i]) >= target_reward:
                    convergence_ep = i
                    break
            
            convergence_episodes.append(convergence_ep)
        
        bars3 = ax3.bar(algorithms, convergence_episodes, color=colors, alpha=0.7)
        ax3.set_ylabel('收敛Episode数')
        ax3.set_title('收敛速度对比')
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars3, convergence_episodes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value}', ha='center', va='bottom')
        
        # 4. 稳定性对比
        ax4 = axes[1, 1]
        stabilities = []
        for alg in algorithms:
            rewards = results[alg]['training_history']['rewards']
            # 计算最后100个episode的标准差作为稳定性指标
            stability = np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
            stabilities.append(stability)
        
        bars4 = ax4.bar(algorithms, stabilities, color=colors, alpha=0.7)
        ax4.set_ylabel('奖励标准差')
        ax4.set_title('性能稳定性对比 (越小越稳定)')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars4, stabilities):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"算法对比图已保存到: {save_path}")
        
        return fig
    
    def plot_action_analysis(self, training_results: List[Dict], 
                           action_names: List[str],
                           save_path: Optional[str] = None) -> plt.Figure:
        """分析动作选择模式"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('动作选择分析', fontsize=16, fontweight='bold')
        
        # 收集所有动作
        all_actions = []
        episode_actions = []
        
        for result in training_results:
            actions = result.get('actions', [])
            all_actions.extend(actions)
            episode_actions.append(actions)
        
        if not all_actions:
            logger.warning("没有动作数据")
            return fig
        
        # 1. 动作分布饼图
        ax1 = axes[0, 0]
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        labels = [action_names[i] if i < len(action_names) else f'动作{i}' for i in action_counts.keys()]
        sizes = list(action_counts.values())
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('动作选择分布')
        
        # 2. 动作随时间变化
        ax2 = axes[0, 1]
        episodes = list(range(len(episode_actions)))
        
        # 计算每个episode的动作多样性
        action_diversity = []
        for actions in episode_actions:
            if actions:
                unique_actions = len(set(actions))
                action_diversity.append(unique_actions)
            else:
                action_diversity.append(0)
        
        ax2.plot(episodes, action_diversity, marker='o', markersize=3, alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('动作多样性')
        ax2.set_title('动作多样性变化')
        ax2.grid(True, alpha=0.3)
        
        # 3. 动作频次热力图
        ax3 = axes[1, 0]
        
        # 创建动作-episode矩阵
        num_actions = len(action_names)
        num_episodes = len(episode_actions)
        action_matrix = np.zeros((num_actions, min(num_episodes, 100)))  # 限制显示最近100个episodes
        
        start_ep = max(0, num_episodes - 100)
        for i, actions in enumerate(episode_actions[start_ep:]):
            for action in actions:
                if action < num_actions:
                    action_matrix[action, i] += 1
        
        im = ax3.imshow(action_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Episode (最近100个)')
        ax3.set_ylabel('动作类型')
        ax3.set_yticks(range(num_actions))
        ax3.set_yticklabels([action_names[i] if i < len(action_names) else f'动作{i}' for i in range(num_actions)])
        ax3.set_title('动作选择热力图')
        plt.colorbar(im, ax=ax3)
        
        # 4. 动作转换矩阵
        ax4 = axes[1, 1]
        
        # 计算动作转换概率
        transition_matrix = np.zeros((num_actions, num_actions))
        
        for actions in episode_actions:
            for i in range(len(actions) - 1):
                if actions[i] < num_actions and actions[i+1] < num_actions:
                    transition_matrix[actions[i], actions[i+1]] += 1
        
        # 归一化
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums!=0)
        
        im2 = ax4.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        ax4.set_xlabel('下一个动作')
        ax4.set_ylabel('当前动作')
        ax4.set_xticks(range(num_actions))
        ax4.set_yticks(range(num_actions))
        ax4.set_xticklabels([action_names[i] if i < len(action_names) else f'动作{i}' for i in range(num_actions)], rotation=45)
        ax4.set_yticklabels([action_names[i] if i < len(action_names) else f'动作{i}' for i in range(num_actions)])
        ax4.set_title('动作转换概率矩阵')
        plt.colorbar(im2, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"动作分析图已保存到: {save_path}")
        
        return fig
    
    def plot_reward_components(self, training_results: List[Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """分析奖励组件"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('奖励组件分析', fontsize=16, fontweight='bold')
        
        # 提取奖励组件数据
        episodes = []
        consensus_scores = []
        stability_scores = []
        safety_scores = []
        effectiveness_scores = []
        
        for result in training_results:
            if 'info' in result and 'reward_breakdown' in result['info']:
                episodes.append(result['episode'])
                breakdown = result['info']['reward_breakdown']
                consensus_scores.append(breakdown.get('consensus_score', 0))
                stability_scores.append(breakdown.get('stability_score', 0))
                safety_scores.append(breakdown.get('safety_score', 0))
                effectiveness_scores.append(breakdown.get('effectiveness_score', 0))
        
        if not episodes:
            logger.warning("没有奖励组件数据")
            return fig
        
        # 1. 奖励组件时间序列
        ax1 = axes[0, 0]
        ax1.plot(episodes, consensus_scores, label='共识得分', alpha=0.7)
        ax1.plot(episodes, stability_scores, label='稳定性得分', alpha=0.7)
        ax1.plot(episodes, safety_scores, label='安全性得分', alpha=0.7)
        ax1.plot(episodes, effectiveness_scores, label='有效性得分', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('得分')
        ax1.set_title('奖励组件变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励组件分布
        ax2 = axes[0, 1]
        components_data = [consensus_scores, stability_scores, safety_scores, effectiveness_scores]
        component_names = ['共识得分', '稳定性得分', '安全性得分', '有效性得分']
        
        ax2.boxplot(components_data, labels=component_names)
        ax2.set_ylabel('得分')
        ax2.set_title('奖励组件分布')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. 奖励组件相关性
        ax3 = axes[1, 0]
        
        # 创建相关性矩阵
        data_df = pd.DataFrame({
            '共识得分': consensus_scores,
            '稳定性得分': stability_scores,
            '安全性得分': safety_scores,
            '有效性得分': effectiveness_scores
        })
        
        correlation_matrix = data_df.corr()
        im = ax3.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        
        # 添加文本标注
        for i in range(len(component_names)):
            for j in range(len(component_names)):
                text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        ax3.set_xticks(range(len(component_names)))
        ax3.set_yticks(range(len(component_names)))
        ax3.set_xticklabels(component_names, rotation=45)
        ax3.set_yticklabels(component_names)
        ax3.set_title('奖励组件相关性')
        plt.colorbar(im, ax=ax3)
        
        # 4. 奖励组件贡献度
        ax4 = axes[1, 1]
        
        avg_contributions = [
            np.mean(consensus_scores),
            np.mean(stability_scores),
            np.mean(safety_scores),
            np.mean(effectiveness_scores)
        ]
        
        bars = ax4.bar(component_names, avg_contributions, color=self.colors[:4], alpha=0.7)
        ax4.set_ylabel('平均得分')
        ax4.set_title('平均奖励组件贡献')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, avg_contributions):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"奖励组件分析图已保存到: {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """创建交互式仪表板"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('学习曲线', '奖励分布', '动作分布', '性能指标'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        training_history = results.get('training_history', {})
        episodes = training_history.get('episodes', [])
        rewards = training_history.get('rewards', [])
        
        if episodes and rewards:
            # 1. 学习曲线
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, mode='lines', name='奖励',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # 2. 奖励分布
            fig.add_trace(
                go.Histogram(x=rewards, name='奖励分布', nbinsx=30,
                           marker_color='green', opacity=0.7),
                row=1, col=2
            )
            
            # 3. 动作分布 (如果有动作数据)
            if 'training_results' in results:
                all_actions = []
                for result in results['training_results']:
                    all_actions.extend(result.get('actions', []))
                
                if all_actions:
                    action_counts = {}
                    for action in all_actions:
                        action_counts[action] = action_counts.get(action, 0) + 1
                    
                    fig.add_trace(
                        go.Pie(labels=list(action_counts.keys()),
                              values=list(action_counts.values()),
                              name="动作分布"),
                        row=2, col=1
                    )
            
            # 4. 性能指标
            final_stats = results.get('final_stats', {})
            if final_stats:
                metrics = ['平均奖励', '最大奖励', '最小奖励', '标准差']
                values = [
                    final_stats.get('average_reward', 0),
                    final_stats.get('max_reward', 0),
                    final_stats.get('min_reward', 0),
                    final_stats.get('std_reward', 0)
                ]
                
                fig.add_trace(
                    go.Bar(x=metrics, y=values, name='性能指标',
                          marker_color='orange'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="强化学习训练仪表板",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def save_all_plots(self, results: Dict[str, Any], output_dir: str):
        """保存所有图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 学习曲线
        if 'training_history' in results:
            fig1 = self.plot_learning_curve(
                results['training_history'],
                save_path=os.path.join(output_dir, 'learning_curve.png')
            )
            plt.close(fig1)
        
        # 动作分析
        if 'training_results' in results:
            action_names = ['保守治疗', '药物强化', '介入治疗', '手术治疗', '综合治疗', '紧急处理']
            fig2 = self.plot_action_analysis(
                results['training_results'],
                action_names,
                save_path=os.path.join(output_dir, 'action_analysis.png')
            )
            plt.close(fig2)
        
        # 奖励组件分析
        if 'training_results' in results:
            fig3 = self.plot_reward_components(
                results['training_results'],
                save_path=os.path.join(output_dir, 'reward_components.png')
            )
            plt.close(fig3)
        
        logger.info(f"所有图表已保存到: {output_dir}")