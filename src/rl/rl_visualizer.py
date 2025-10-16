"""
Reinforcement Learning Visualization Tool
Generate learning curves, reward curves and performance analysis charts
Author: AI Assistant
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

# Set font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class RLVisualizer:
    """Reinforcement Learning Visualizer"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize visualizer"""
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("RL Visualizer initialized successfully")
    
    def plot_learning_curve(self, training_history: Dict[str, List], 
                           window_size: int = 100, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot learning curves"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Reinforcement Learning Training Curves', fontsize=16, fontweight='bold')
        
        episodes = training_history.get('episodes', [])
        rewards = training_history.get('rewards', [])
        
        if not episodes or not rewards:
            logger.warning("Training history data is empty")
            return fig
        
        # 1. Reward curve
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        # Moving average
        if len(rewards) >= window_size:
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'{window_size}-Episode Moving Average')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Change Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward distribution
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
        ax2.set_xlabel('Reward Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution Histogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative rewards
        ax3 = axes[1, 0]
        cumulative_rewards = np.cumsum(rewards)
        ax3.plot(episodes, cumulative_rewards, color='purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title('Cumulative Reward Curve')
        ax3.grid(True, alpha=0.3)
        
        # 4. Reward improvement trend
        ax4 = axes[1, 1]
        if len(rewards) >= 200:
            # Calculate average reward per 100 episodes
            chunk_size = 100
            chunk_rewards = []
            chunk_episodes = []
            
            for i in range(0, len(rewards) - chunk_size + 1, chunk_size):
                chunk_rewards.append(np.mean(rewards[i:i+chunk_size]))
                chunk_episodes.append(episodes[i+chunk_size-1])
            
            ax4.plot(chunk_episodes, chunk_rewards, marker='o', linewidth=2, markersize=6)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel(f'Average Reward (per {chunk_size} episodes)')
            ax4.set_title('Learning Progress Trend')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data\n(Need at least 200 episodes)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Learning Progress Trend')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curve saved to: {save_path}")
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict[str, Dict], 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Compare performance of different algorithms"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        algorithms = list(results.keys())
        colors = self.colors[:len(algorithms)]
        
        # 1. Average reward comparison
        ax1 = axes[0, 0]
        avg_rewards = [results[alg]['final_stats']['average_reward'] for alg in algorithms]
        bars1 = ax1.bar(algorithms, avg_rewards, color=colors, alpha=0.7)
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Average Reward Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, avg_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Learning curve comparison
        ax2 = axes[0, 1]
        for i, alg in enumerate(algorithms):
            rewards = results[alg]['training_history']['rewards']
            episodes = results[alg]['training_history']['episodes']
            
            # Moving average
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
                ax2.plot(episodes, moving_avg, color=colors[i], linewidth=2, label=alg)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward (Moving Average)')
        ax2.set_title('Learning Curve Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergence speed comparison
        ax3 = axes[1, 0]
        convergence_episodes = []
        for alg in algorithms:
            rewards = results[alg]['training_history']['rewards']
            # Simple convergence detection: find point where reward stabilizes
            target_reward = np.mean(rewards[-100:]) * 0.9  # 90% of final performance
            convergence_ep = len(rewards)  # Default value
            
            window_size = 50
            for i in range(window_size, len(rewards)):
                if np.mean(rewards[i-window_size:i]) >= target_reward:
                    convergence_ep = i
                    break
            
            convergence_episodes.append(convergence_ep)
        
        bars3 = ax3.bar(algorithms, convergence_episodes, color=colors, alpha=0.7)
        ax3.set_ylabel('Convergence Episodes')
        ax3.set_title('Convergence Speed Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, convergence_episodes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{value}', ha='center', va='bottom')
        
        # 4. Stability comparison
        ax4 = axes[1, 1]
        stabilities = []
        for alg in algorithms:
            rewards = results[alg]['training_history']['rewards']
            # Calculate standard deviation of last 100 episodes as stability metric
            stability = np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
            stabilities.append(stability)
        
        bars4 = ax4.bar(algorithms, stabilities, color=colors, alpha=0.7)
        ax4.set_ylabel('Reward Standard Deviation')
        ax4.set_title('Performance Stability Comparison (Lower is Better)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, stabilities):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Algorithm comparison chart saved to: {save_path}")
        
        return fig
    
    def plot_action_analysis(self, training_results: List[Dict], 
                           action_names: List[str],
                           save_path: Optional[str] = None) -> plt.Figure:
        """Analyze action selection patterns"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Action Selection Analysis', fontsize=16, fontweight='bold')
        
        # Collect all actions
        all_actions = []
        episode_actions = []
        
        for result in training_results:
            actions = result.get('actions', [])
            all_actions.extend(actions)
            episode_actions.append(actions)
        
        if not all_actions:
            logger.warning("No action data available")
            return fig
        
        # 1. Action distribution pie chart
        ax1 = axes[0, 0]
        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        labels = [action_names[i] if i < len(action_names) else f'Action{i}' for i in action_counts.keys()]
        sizes = list(action_counts.values())
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Action Selection Distribution')
        
        # 2. Action diversity over time
        ax2 = axes[0, 1]
        episodes = list(range(len(episode_actions)))
        
        # Calculate action diversity for each episode
        action_diversity = []
        for actions in episode_actions:
            if actions:
                unique_actions = len(set(actions))
                action_diversity.append(unique_actions)
            else:
                action_diversity.append(0)
        
        ax2.plot(episodes, action_diversity, marker='o', markersize=3, alpha=0.7)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Action Diversity')
        ax2.set_title('Action Diversity Changes')
        ax2.grid(True, alpha=0.3)
        
        # 3. Action frequency heatmap
        ax3 = axes[1, 0]
        
        # Create action-episode matrix
        num_actions = len(action_names)
        num_episodes = len(episode_actions)
        action_matrix = np.zeros((num_actions, min(num_episodes, 100)))  # Limit to recent 100 episodes
        
        start_ep = max(0, num_episodes - 100)
        for i, actions in enumerate(episode_actions[start_ep:]):
            for action in actions:
                if action < num_actions:
                    action_matrix[action, i] += 1
        
        im = ax3.imshow(action_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Episode (Recent 100)')
        ax3.set_ylabel('Action Type')
        ax3.set_yticks(range(num_actions))
        ax3.set_yticklabels([action_names[i] if i < len(action_names) else f'Action{i}' for i in range(num_actions)])
        ax3.set_title('Action Selection Heatmap')
        plt.colorbar(im, ax=ax3)
        
        # 4. Action transition matrix
        ax4 = axes[1, 1]
        
        # Calculate action transition probabilities
        transition_matrix = np.zeros((num_actions, num_actions))
        
        for actions in episode_actions:
            for i in range(len(actions) - 1):
                if actions[i] < num_actions and actions[i+1] < num_actions:
                    transition_matrix[actions[i], actions[i+1]] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums!=0)
        
        im2 = ax4.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
        ax4.set_xlabel('Next Action')
        ax4.set_ylabel('Current Action')
        ax4.set_xticks(range(num_actions))
        ax4.set_yticks(range(num_actions))
        ax4.set_xticklabels([action_names[i] if i < len(action_names) else f'Action{i}' for i in range(num_actions)], rotation=45)
        ax4.set_yticklabels([action_names[i] if i < len(action_names) else f'Action{i}' for i in range(num_actions)])
        ax4.set_title('Action Transition Probability Matrix')
        plt.colorbar(im2, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Action analysis chart saved to: {save_path}")
        
        return fig
    
    def plot_reward_components(self, training_results: List[Dict],
                             save_path: Optional[str] = None) -> plt.Figure:
        """Analyze reward components"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Reward Components Analysis', fontsize=16, fontweight='bold')
        
        # Extract reward component data
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
            logger.warning("No reward component data available")
            return fig
        
        # 1. Reward component time series
        ax1 = axes[0, 0]
        ax1.plot(episodes, consensus_scores, label='Consensus Score', alpha=0.7)
        ax1.plot(episodes, stability_scores, label='Stability Score', alpha=0.7)
        ax1.plot(episodes, safety_scores, label='Safety Score', alpha=0.7)
        ax1.plot(episodes, effectiveness_scores, label='Effectiveness Score', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Reward Component Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward component distribution
        ax2 = axes[0, 1]
        components_data = [consensus_scores, stability_scores, safety_scores, effectiveness_scores]
        component_names = ['Consensus Score', 'Stability Score', 'Safety Score', 'Effectiveness Score']
        
        ax2.boxplot(components_data, labels=component_names)
        ax2.set_ylabel('Score')
        ax2.set_title('Reward Component Distribution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Reward component correlation
        ax3 = axes[1, 0]
        
        # Create correlation matrix
        data_df = pd.DataFrame({
            'Consensus Score': consensus_scores,
            'Stability Score': stability_scores,
            'Safety Score': safety_scores,
            'Effectiveness Score': effectiveness_scores
        })
        
        correlation_matrix = data_df.corr()
        im = ax3.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        
        # Add text annotations
        for i in range(len(component_names)):
            for j in range(len(component_names)):
                text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        ax3.set_xticks(range(len(component_names)))
        ax3.set_yticks(range(len(component_names)))
        ax3.set_xticklabels(component_names, rotation=45)
        ax3.set_yticklabels(component_names)
        ax3.set_title('Reward Component Correlation')
        plt.colorbar(im, ax=ax3)
        
        # 4. Reward component contribution
        ax4 = axes[1, 1]
        
        avg_contributions = [
            np.mean(consensus_scores),
            np.mean(stability_scores),
            np.mean(safety_scores),
            np.mean(effectiveness_scores)
        ]
        
        bars = ax4.bar(component_names, avg_contributions, color=self.colors[:4], alpha=0.7)
        ax4.set_ylabel('Average Score')
        ax4.set_title('Average Reward Component Contribution')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, avg_contributions):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reward components analysis chart saved to: {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict[str, Any]) -> go.Figure:
        """Create interactive dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Learning Curve', 'Reward Distribution', 'Action Distribution', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        training_history = results.get('training_history', {})
        episodes = training_history.get('episodes', [])
        rewards = training_history.get('rewards', [])
        
        if episodes and rewards:
            # 1. Learning curve
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, mode='lines', name='Rewards',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # 2. Reward distribution
            fig.add_trace(
                go.Histogram(x=rewards, name='Reward Distribution', nbinsx=30,
                           marker_color='green', opacity=0.7),
                row=1, col=2
            )
            
            # 3. Action distribution (if action data available)
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
                              name="Action Distribution"),
                        row=2, col=1
                    )
            
            # 4. Performance metrics
            final_stats = results.get('final_stats', {})
            if final_stats:
                metrics = ['Average Reward', 'Max Reward', 'Min Reward', 'Std Deviation']
                values = [
                    final_stats.get('average_reward', 0),
                    final_stats.get('max_reward', 0),
                    final_stats.get('min_reward', 0),
                    final_stats.get('std_reward', 0)
                ]
                
                fig.add_trace(
                    go.Bar(x=metrics, y=values, name='Performance Metrics',
                          marker_color='orange'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Reinforcement Learning Training Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def save_all_plots(self, results: Dict[str, Any], output_dir: str):
        """Save all plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Learning curve
        if 'training_history' in results:
            fig1 = self.plot_learning_curve(
                results['training_history'],
                save_path=os.path.join(output_dir, 'learning_curve.png')
            )
            plt.close(fig1)
        
        # Action analysis
        if 'training_results' in results:
            action_names = ['Conservative Treatment', 'Drug Enhancement', 'Intervention', 'Surgery', 'Comprehensive Treatment', 'Emergency Treatment']
            fig2 = self.plot_action_analysis(
                results['training_results'],
                action_names=action_names,
                save_path=os.path.join(output_dir, 'action_analysis.png')
            )
            plt.close(fig2)
        
        # Reward components analysis
        if 'training_results' in results:
            fig3 = self.plot_reward_components(
                results['training_results'],
                save_path=os.path.join(output_dir, 'reward_components.png')
            )
            plt.close(fig3)
        
        logger.info(f"All plots saved to: {output_dir}")