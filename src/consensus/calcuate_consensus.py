# consensus_matrix.py

import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2
import matplotlib.pyplot as plt
from typing import List, Union, Dict
import sys
import os

# 获取 src 目录的绝对路径（当前脚本的父目录的父目录）
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
from src.core.data_models import RoleType, TreatmentOption, RoleOpinion, QuestionOpinion, RoleRegistry
from experiments.medqa_types import QuestionOption


class CalculateConsensus:
    def __init__(self):
        """
        data 示例结构:
        {
          'doctor': {'prefs': {'surgery': 0.9, 'chemotherapy': 0.7, ...}, 'confidence': 0.95},
          'nurse': {'prefs': {...}, 'confidence': 0.9},
          ...
        }
        """
        self.roles = list(RoleType)
        self.treatments: List[Union[TreatmentOption, QuestionOption]] = []
        self.m = len(self.roles)
        self.n = len(self.treatments)
        self.df_scores = None
        self.df_ranks = None
        self.W = None
        self.p_value = None

    def set_treatments(self, treatments: List[Union[TreatmentOption, QuestionOption]]):
        self.treatments = treatments
        self.n = len(self.treatments)

    def set_roles(self, roles: List[Union[RoleType, RoleRegistry]]):
        self.roles = roles
        self.m = len(self.roles)

    def build_weighted_matrix(self, opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]]):
        """构建偏好×置信度的加权共识矩阵"""
        score_matrix = np.zeros((self.m, self.n))
        for i, role in enumerate(self.roles):
            prefs = opinions_dict[role].scores
            conf = opinions_dict[role].evidence_strength
            role_weight = role.weight
            # 将原始分数从 -1~1 映射到 0~1
            mapped_scores = [(prefs[t.name] + 1) / 2 for t in self.treatments]
            # 这里要怎么改呢
            # score_matrix[i, :] = [prefs[t.value] * conf for t in self.treatments]
            # score_matrix[i, :] = [prefs[t.name] * conf for t in self.treatments]
            # 最终矩阵 = 映射后的分数 × 证据强度 × 角色权重
            score_matrix[i, :] = [s * conf * role_weight for s in mapped_scores]
        self.df_scores = pd.DataFrame(
            score_matrix,
            index=[role.value for role in self.roles],
            columns=[t.value for t in self.treatments]
        ).T
        return self.df_scores

    def compute_kendalls_w(self):
        """计算带置信度加权的 Kendall’s W 协调系数"""
        if self.df_scores is None:
            self.build_weighted_matrix()
        score_matrix = self.df_scores.T.values

        # 每个角色的评分 → 排名（降序）
        ranks = np.zeros_like(score_matrix)
        for i in range(self.m):
            ranks[i, :] = rankdata(-score_matrix[i, :], method='average')
        self.df_ranks = pd.DataFrame(ranks, index=self.roles, columns=self.treatments).T

        # 计算 Kendall’s W
        R_j = self.df_ranks.sum(axis=1).values
        R_bar = self.m * (self.n + 1) / 2.0
        S = np.sum((R_j - R_bar) ** 2)

        tie_correction = 0.0
        for i in range(self.m):
            vals, counts = np.unique(ranks[i, :], return_counts=True)
            for c in counts:
                if c > 1:
                    tie_correction += (c ** 3 - c)

        denominator_no_ties = self.m ** 2 * (self.n ** 3 - self.n)
        denominator = denominator_no_ties - self.m * tie_correction
        self.W = 12 * S / denominator if denominator > 0 else np.nan

        # 显著性检验
        chi2_stat = self.m * (self.n - 1) * self.W
        df = self.n - 1
        self.p_value = chi2.sf(chi2_stat, df)

        return self.W, self.p_value  # 返回 W 协调系数与 p 值

    def summarize(self, consensus_threshold=0.8):
        """输出均值、标准差与是否达成共识"""
        if self.df_scores is None:
            self.build_weighted_matrix()
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        consensus = True if self.W and self.W >= consensus_threshold else False
        print(f"Kendall's W 协调系数: {self.W}, p 值: {self.p_value}, 共识: {consensus}")
        return df, self.W, self.p_value, consensus

    def plot_matrix(self):
        """绘制每个治疗项目的均值与方差"""
        if self.df_scores is None:
            self.build_weighted_matrix()
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        df[["mean", "std"]].plot(kind="bar", figsize=(8, 4), title="Treatment Preferences (Weighted by Confidence)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()
