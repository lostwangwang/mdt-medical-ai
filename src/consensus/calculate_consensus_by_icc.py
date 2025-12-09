import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from typing import List, Union, Dict
from experiments.medqa_types import QuestionOption
from src.core.data_models import RoleType, TreatmentOption, RoleRegistry, RoleOpinion, QuestionOpinion


class CalculateConsensusByICC:
    def __init__(self):
        self.roles = list(RoleType)
        self.treatments: List[Union[TreatmentOption, QuestionOption]] = []
        self.m = len(self.roles)
        self.n = len(self.treatments)
        self.df_scores = None
        self.group_icc = None

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
            # conf = opinions_dict[role].evidence_strength
            role_weight = role.weight
            # score_matrix[i, :] = [prefs[t.name] * conf * role_weight for t in self.treatments]
            score_matrix[i, :] = [prefs[t.name] * role_weight for t in self.treatments]
        self.df_scores = pd.DataFrame(
            score_matrix,
            index=[role.value for role in self.roles],
            columns=[t.value for t in self.treatments]
        ).T
        return self.df_scores

    def compute_icc_consensus(self):
        """计算 ICC(3,k) 并作为群体一致性指标"""
        if self.df_scores is None:
            raise ValueError("请先调用 build_weighted_matrix()")

        # pingouin 需要 long format
        df_long = self.df_scores.reset_index().melt(id_vars=['index'], var_name='Rater', value_name='Score')
        df_long.rename(columns={'index': 'Item'}, inplace=True)

        icc_res = pg.intraclass_corr(data=df_long, targets='Item', raters='Rater', ratings='Score')
        icc_3k = icc_res[(icc_res['Type'] == 'ICC3k')]
        self.group_icc = icc_3k['ICC'].values[0]
        return self.group_icc

    def summarize(self, icc_threshold=0.7):
        """输出均值、标准差、ICC一致性以及是否达成共识"""
        if self.df_scores is None:
            raise ValueError("请先调用 build_weighted_matrix()")
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)

        if self.group_icc is None:
            self.compute_icc_consensus()

        consensus = self.group_icc >= icc_threshold
        print(f"Group consensus (ICC3,k): {self.group_icc:.3f}, 达成共识: {consensus}")
        return df, self.group_icc, consensus

    def select_final_answer(self):
        """基于均值和标准差调整分数选择最终答案"""
        df = self.df_scores.copy()
        df["mean"] = df.mean(axis=1)
        df["std"] = df.std(axis=1)
        df["adjusted_score"] = df["mean"] * (1 - df["std"])
        final_option = df["adjusted_score"].idxmax()

        print("选项均值:\n", df["mean"])
        print("选项标准差:\n", df["std"])
        print("调整后分数:\n", df["adjusted_score"])
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
