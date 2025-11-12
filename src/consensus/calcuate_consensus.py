# consensus_matrix.py

import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2
import matplotlib.pyplot as plt
from typing import List,Union
import sys
import os
# 获取 src 目录的绝对路径（当前脚本的父目录的父目录）
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if src_path not in sys.path:
    sys.path.append(src_path)
from src.core.data_models import RoleType, TreatmentOption, RoleOpinion
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
    
    def build_weighted_matrix(self, opinions_dict: dict[RoleType, RoleOpinion]):
        """构建偏好×置信度的加权共识矩阵"""
        score_matrix = np.zeros((self.m, self.n))
        for i, role in enumerate(self.roles):
            prefs = opinions_dict[role.value].treatment_preferences
            conf = opinions_dict[role.value].confidence
            # 这里要怎么改呢
            # score_matrix[i, :] = [prefs[t.value] * conf for t in self.treatments]
            score_matrix[i, :] = [prefs[t.name] * conf for t in self.treatments]
        self.df_scores = pd.DataFrame(score_matrix, index=[role.value for role in self.roles], columns=[t.value for t in self.treatments]).T
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
                    tie_correction += (c**3 - c)

        denominator_no_ties = self.m**2 * (self.n**3 - self.n)
        denominator = denominator_no_ties - self.m * tie_correction
        self.W = 12 * S / denominator if denominator > 0 else np.nan

        # 显著性检验
        chi2_stat = self.m * (self.n - 1) * self.W
        df = self.n - 1
        self.p_value = chi2.sf(chi2_stat, df)

        return self.W, self.p_value # 返回 W 协调系数与 p 值

    def summarize(self, consensus_threshold=0.8):
        """输出均值、标准差与是否达成共识"""
        if self.df_scores is None:
            self.build_weighted_matrix()
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        consensus = True if self.W and self.W > consensus_threshold else False
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

if __name__ == "__main__":
    # 示例数据（每个角色的偏好和置信度）
    role_type = list(RoleType)
    print(role_type)
    treatments = list(TreatmentOption)
    print(treatments)
    opinions_dict = {'oncologist': RoleOpinion(role='oncologist', treatment_preferences={'surgery': 0.9, 'chemotherapy': 0.7, 'radiotherapy': 0.8, 'immunotherapy': 0.3, 'palliative_care': -0.5, 'watchful_waiting': -0.8}, reasoning='手术为根治基石，多学科支持可降低风险，综合治疗提升生存率', confidence=0.95, concerns=['围术期心脏事件', '化疗耐受性', '血糖波动']), 'radiologist': RoleOpinion(role='radiologist', treatment_preferences={'surgery': 0.9, 'chemotherapy': 0.7, 'radiotherapy': 0.8, 'immunotherapy': 0.3, 'palliative_care': -0.5, 'watchful_waiting': -0.8}, reasoning='影像学支持根治性手术，多学科协同可降低围术期风险，术后放疗强化局部控制。', confidence=0.95, concerns=['围术期心血管事件', '放疗耐受性', '血糖波动影响愈合']), 'nurse': RoleOpinion(role='nurse', treatment_preferences={'surgery': 0.9, 'chemotherapy': 0.6, 'radiotherapy': 0.7, 'immunotherapy': 0.4, 'palliative_care': -0.3, 'watchful_waiting': -0.6}, reasoning='患者状态稳定，多学科支持下手术风险可控，术后恢复预期良好。', confidence=0.95, concerns=['围术期心功能波动', '血糖波动风险', '术后感染可能']), 'psychologist': RoleOpinion(role='psychologist', treatment_preferences={'surgery': 0.85, 'chemotherapy': 0.65, 'radiotherapy': 0.55, 'immunotherapy': 0.3, 'palliative_care': -0.2, 'watchful_waiting': -0.4}, reasoning='患者心理状态良好，手术可增强掌控感，多学科支持降低心理负担', confidence=0.9, concerns=['术后心理适应', '治疗依从性波动', '康复信心波动']), 'patient_advocate': RoleOpinion(role='patient_advocate', treatment_preferences={'surgery': 0.95, 'chemotherapy': 0.65, 'radiotherapy': 0.75, 'immunotherapy': 0.3, 'palliative_care': -0.1, 'watchful_waiting': -0.6}, reasoning='多学科支持手术，患者状态稳定，围术期管理可控，根治性治疗优先', confidence=0.9, concerns=['围术期并发症', '术后恢复挑战', '合并症叠加风险']), 'nutritionist': RoleOpinion(role='nutritionist', treatment_preferences={'surgery': 0.8, 'chemotherapy': 0.6, 'radiotherapy': 0.5, 'immunotherapy': 0.4, 'palliative_care': -0.3, 'watchful_waiting': -0.6}, reasoning='术前营养优化可提升手术耐受，术后肠内营养支持促进恢复，整体获益显著', confidence=0.9, concerns=['术后吸收障碍', '化疗食欲下降', '合并症恶化']), 'rehabilitation_therapist': RoleOpinion(role='rehabilitation_therapist', treatment_preferences={'surgery': 0.9, 'chemotherapy': 0.6, 'radiotherapy': 0.5, 'immunotherapy': 0.3, 'palliative_care': -0.2, 'watchful_waiting': -0.6}, reasoning='手术获益明确，多学科支持下风险可控，术前康复可提升耐受力', confidence=0.9, concerns=['术后功能障碍', '治疗依从性', '康复延迟'])}
    data = {}
    calc = CalculateConsensus()
    calc.set_treatments(treatments)
    calc.build_weighted_matrix(opinions_dict)
    print(calc.df_scores)
    calc.compute_kendalls_w()
    df, W, p_value, consensus = calc.summarize()
    print("W的类型", type(W))
    print(f"Kendall's W 协调系数: {W}, p 值: {p_value}, 共识: {consensus}")
    print(df['mean'])
    # 如何选出最优的治疗方案
    best_treatment = df['mean'].idxmax()
    print(f"最优治疗方案: {best_treatment}")
    print(f"最优治疗方案枚举值: {TreatmentOption(best_treatment).name}")
