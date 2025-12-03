import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from experiments.medqa_types import QuestionOption
from src.core.data_models import RoleType, TreatmentOption, RoleRegistry, RoleOpinion, QuestionOpinion


class CalculateConsensusByCosineSimilarity:
    def __init__(self):
        self.roles = list(RoleType)
        self.treatments: List[Union[TreatmentOption, QuestionOption]] = []
        self.m = len(self.roles)
        self.n = len(self.treatments)
        self.df_scores = None
        self.cos_matrix = None
        self.group_consensus = None

    def set_treatments(self, treatments: List[Union[TreatmentOption, QuestionOption]]):
        self.treatments = treatments
        self.n = len(treatments)

    def set_roles(self, roles: List[Union[RoleType, RoleRegistry]]):
        self.roles = roles
        self.m = len(roles)

    def build_weighted_matrix(self, opinions_dict: Dict[Union[RoleType, RoleRegistry], Union[RoleOpinion, QuestionOpinion]]):
        """构建支持度×证据强度×角色权重向量矩阵"""
        score_matrix = np.zeros((self.m, self.n))
        for i, role in enumerate(self.roles):
            prefs = opinions_dict[role].scores
            conf = opinions_dict[role].evidence_strength
            role_weight = role.weight
            # 直接用 -1~1 的打分 × 证据强度 × 角色权重
            score_matrix[i, :] = [prefs[t.name] * conf * role_weight for t in self.treatments]

        self.df_scores = pd.DataFrame(
            score_matrix,
            index=[role.value for role in self.roles],
            columns=[t.value for t in self.treatments]
        ).T
        return self.df_scores

    def compute_cosine_consensus(self):
        """计算基于余弦相似度的共识矩阵和整体共识"""
        if self.df_scores is None:
            raise ValueError("请先调用 build_weighted_matrix()")

        vectors = self.df_scores.T.values  # shape: (num_roles, num_treatments)
        m = vectors.shape[0]

        # 计算 Cosine similarity 矩阵
        cos_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                v_i = vectors[i]
                v_j = vectors[j]
                denom = np.linalg.norm(v_i) * np.linalg.norm(v_j)
                cos_matrix[i, j] = np.dot(v_i, v_j) / denom if denom > 0 else 0.0

        self.cos_matrix = pd.DataFrame(cos_matrix,
                                       index=[role.value for role in self.roles],
                                       columns=[role.value for role in self.roles])

        # 计算群体中心向量共识
        centroid = np.mean(vectors, axis=0)
        consensus_scores = []
        for v in vectors:
            denom = np.linalg.norm(v) * np.linalg.norm(centroid)
            cosine_sim = np.dot(v, centroid) / denom if denom > 0 else 0.0
            consensus_scores.append(cosine_sim)
        self.group_consensus = np.mean(consensus_scores)

        return self.cos_matrix, self.group_consensus

    def summarize(self, consensus_threshold=0.7, std_threshold=0.1):
        """输出均值、标准差、共识矩阵、整体共识"""
        if self.df_scores is None:
            raise ValueError("请先调用 build_weighted_matrix()")
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        max_std = df["std"].max()
        consensus_by_std = max_std <= std_threshold

        # 综合余弦相似度和标准差判断
        consensus = (self.group_consensus >= consensus_threshold) and consensus_by_std
        # consensus = True if self.group_consensus and self.group_consensus >= consensus_threshold else False
        print(f"Group consensus (Cosine similarity): {self.group_consensus:.3f}, 达成共识: {consensus}")
        return df, self.cos_matrix, self.group_consensus, consensus

    def select_final_answer(self, summary_df):
        """
        summary_df: DataFrame，包含 columns=['mean','std']，index=选项标签



        """
        summary_df = self.df_scores.copy()
        summary_df["mean"] = summary_df.mean(axis=1)
        summary_df["std"] = summary_df.std(axis=1)
        # 调整分数：惩罚分歧大的选项
        summary_df["adjusted_score"] = summary_df["mean"] * (1 - summary_df["std"])

        # 选择分数最高的选项
        final_option = summary_df["adjusted_score"].idxmax()

        # 可选：打印调试信息
        print("选项均值:\n", summary_df["mean"])
        print("选项标准差:\n", summary_df["std"])
        print("调整后分数:\n", summary_df["adjusted_score"])
        print("最终答案:", final_option)

        return final_option

    def plot_matrix(self):
        """绘制每个治疗项目的均值与方差"""
        if self.df_scores is None:
            raise ValueError("请先调用 build_weighted_matrix()")
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        df[["mean", "std"]].plot(kind="bar", figsize=(8, 4), title="Treatment Preferences (Weighted by Confidence & Role Weight)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()
