# MDT Memory & Roleplay Framework

> 一个能持续演化的医疗智能体系统，模拟多学科团队（MDT）会诊的决策协同

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Development-orange.svg)

## 📋 项目概览

本项目构建了一个创新的医疗AI系统，结合了：
- **动态记忆系统**：患者病情的时序演化建模
- **多智能体角色对话**：模拟肿瘤科医生、影像科医生、护士、心理师、患者代表的专业观点
- **共识矩阵**：量化团队决策的一致性与冲突
- **强化学习优化**：从历史会诊中学习最优决策策略

### 🎯 核心特性

- **🤖 多智能体协商**：5个专业角色的智能对话与决策
- **🧠 动态记忆演化**：患者状态的时间序列建模
- **📊 共识量化分析**：可解释的决策透明性
- **🚀 强化学习优化**：持续改进的决策质量
- **📈 丰富的可视化**：全面的分析仪表板

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Memory        │    │   Knowledge     │    │   Consensus     │
│   Controller    │───▶│   RAG System    │───▶│   Matrix        │
│   (杜军)        │    │   (共同维护)    │    │   (姚刚)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Multi-Agent   │    │   RL Training   │    │   Integrated    │
│   Dialogue      │───▶│   Environment   │───▶│   Workflow      │
│   (姚刚)        │    │   (姚刚)        │    │   (Tianyu)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 8GB+ RAM推荐
- CUDA支持（可选，用于大规模训练）

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-team/mdt-medical-ai.git
cd mdt-medical-ai

# 创建虚拟环境
conda create -n mdt-agent-ai python=3.12
conda activate mdt-agent-ai

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

#### 1. 演示模式（推荐首次使用）

```bash
python main_integrated.py --mode demo
```

这将分析3个预设的患者案例，展示系统的完整功能。

#### 2. 单患者分析

```bash
# 准备患者数据文件 patient.json
python main_integrated.py --mode patient --patient-file data/examples/patient.json
```

#### 3. 强化学习训练

```bash
python main_integrated.py --mode training --episodes 1000
```

#### 4. 基线模型对比

```bash
python main_integrated.py --mode comparison --num-patients 100 --num-trials 50
```

#### 5. 时序模拟

```bash
python main_integrated.py --mode simulation --simulation-days 30
```

## 📊 使用示例

### 基本用法

```python
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG
from src.core.data_models import PatientState

# 初始化系统
rag_system = MedicalKnowledgeRAG()
dialogue_manager = MultiAgentDialogueManager(rag_system)

# 创建患者状态
patient = PatientState(
    patient_id="DEMO_001",
    age=65,
    diagnosis="breast_cancer",
    stage="II",
    lab_results={"creatinine": 1.2, "hemoglobin": 11.5},
    vital_signs={"bp_systolic": 140, "heart_rate": 78},
    symptoms=["fatigue", "pain"],
    comorbidities=["diabetes", "hypertension"],
    psychological_status="anxious",
    quality_of_life_score=0.7,
    timestamp=datetime.now()
)

# 进行MDT讨论
result = dialogue_manager.conduct_mdt_discussion(patient)

# 获取推荐结果
recommended_treatment = max(result.aggregated_scores.items(), key=lambda x: x[1])
print(f"推荐治疗: {recommended_treatment[0].value}")
print(f"共识得分: {recommended_treatment[1]:.3f}")
```

### 高级集成使用

```python
from src.integration.workflow_manager import IntegratedWorkflowManager

# 创建工作流程管理器
workflow = IntegratedWorkflowManager()

# 注册患者
workflow.register_patient("P001", {"age": 65, "diagnosis": "breast_cancer"})

# 运行30天时序模拟
results = workflow.run_temporal_simulation("P001", days=30)

print(f"总决策次数: {results['performance_metrics']['total_mdt_discussions']}")
print(f"平均共识得分: {results['performance_metrics']['avg_consensus_score']:.3f}")
```

## 🧪 实验与评估

### 基线模型对比

系统与以下基线模型进行对比：

- **Random Baseline**: 随机决策
- **Rule-based Model**: 基于医学规则
- **Single Agent**: 单一专家（肿瘤科医生）
- **LangChain RAG**: 传统检索增强生成
- **Med-PaLM-like**: 医学大语言模型

### 评估指标

- **准确性**: 与专家标准答案的符合度
- **一致性**: 多次运行结果的稳定性
- **共识对齐**: 与团队共识的一致程度
- **解释质量**: 决策推理的可解释性
- **响应时间**: 系统决策速度

### 运行完整评估

```bash
# 运行所有实验
python experiments/baseline_comparison.py

# 查看结果
ls results/
# ├── figures/
# │   ├── baseline_comparison.png
# │   ├── consensus_heatmap.png
# │   └── learning_curves.png
# ├── comparison_results.json
# └── training_results.json
```

## 📁 项目结构

```
mdt_medical_ai/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── main_integrated.py          # 主程序入口（完整集成版）
├── src/                        # 源代码
│   ├── core/                   # 核心数据模型
│   │   └── data_models.py
│   ├── memory/                 # 记忆模块 (杜军负责)
│   │   ├── memory_controller.py
│   │   └── data_evolve.py
│   ├── consensus/              # 共识模块 (姚刚负责)
│   │   ├── consensus_matrix.py
│   │   ├── role_agents.py
│   │   └── dialogue_manager.py
│   ├── rl/                     # 强化学习 (姚刚负责)
│   │   ├── rl_environment.py
│   │   └── training.py
│   ├── knowledge/              # 知识系统
│   │   └── rag_system.py
│   ├── integration/            # 系统集成
│   │   └── workflow_manager.py
│   └── utils/                  # 工具函数
│       └── visualization.py
├── experiments/                # 实验脚本
│   └── baseline_comparison.py
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── examples/              # 示例数据
├── results/                    # 输出结果
├── tests/                      # 测试代码
├── docs/                       # 文档
└── notebooks/                  # Jupyter笔记本
```

## 👥 团队分工

| 成员       | 主要职责                           | 负责模块                         |
| ---------- | ---------------------------------- | -------------------------------- |
| **Tianyu** | 系统总体设计 / 技术把控 / 论文撰写 | 系统集成、知识管理               |
| **杜军**   | Memory Controller                  | 个体-群体记忆演化与数据管理      |
| **姚刚**   | Consensus Matrix & RL              | 多智能体对话、共识分析、强化学习 |

## 📈 性能基准

| 指标     | 我们的系统 | 最佳基线 | 提升   |
| -------- | ---------- | -------- | ------ |

项目提供两份顶会级论文稿件（Markdown 与 Typst）位于 `docs/`：

- `docs/paper_top_conference.md`
- `docs/paper_top_conference.typ`

一键编译 PDF：

```bash
python scripts/build_papers.py               # 同时编译 MD 与 Typst
python scripts/build_papers.py --md-only     # 仅编译 Markdown
python scripts/build_papers.py --typ-only    # 仅编译 Typst
```

生成结果：

- `docs/paper_top_conference_md.pdf`
- `docs/paper_top_conference_typst.pdf`

编译依赖：

- Markdown → PDF：`pandoc`（推荐）与 `xelatex` 或 `wkhtmltopdf`
- Typst → PDF：`typst`（已采用自包含样式，无需外部模板）

## ⭐ 如果本项目对您有帮助，请给个星星支持！

---

**联系我们**: [team@mdt-medical-ai.com](mailto:team@mdt-medical-ai.com)

*最后更新: 2025年10月*