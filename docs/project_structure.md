# MDT Memory & Roleplay Framework 项目结构

```
mdt_medical_ai/
│
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包
├── setup.py                          # 项目安装配置
├── .gitignore                        # Git忽略文件
├── pytest.ini                       # 测试配置
│
├── config/                           # 配置文件
│   ├── __init__.py
│   ├── model_config.yaml            # 模型配置
│   ├── medical_guidelines.yaml      # 医学指南配置
│   └── experiment_config.yaml       # 实验配置
│
├── data/                            # 数据目录
│   ├── raw/                         # 原始数据
│   │   ├── ehr_samples.xlsx         # EHR样例数据
│   │   └── medical_guidelines/      # 医学指南文档
│   ├── processed/                   # 处理后数据
│   │   ├── patient_simulated.csv    # 患者模拟数据
│   │   └── consensus_history.json   # 共识历史记录
│   └── knowledge_base/              # 知识库
│       ├── pubmed_embeddings/       # PubMed向量数据
│       └── clinical_guidelines/     # 临床指南
│
├── src/                             # 源代码目录
│   ├── __init__.py
│   │
│   ├── core/                        # 核心组件
│   │   ├── __init__.py
│   │   ├── data_models.py           # 数据模型定义
│   │   └── base_classes.py          # 基础类定义
│   │
│   ├── memory/                      # 记忆模块 (杜军负责)
│   │   ├── __init__.py
│   │   ├── memory_controller.py     # 记忆控制器
│   │   ├── data_evolve.py          # 数据演化逻辑
│   │   ├── individual_memory.py    # 个体记忆
│   │   └── group_memory.py         # 群体记忆
│   │
│   ├── consensus/                   # 共识模块 (姚刚负责)
│   │   ├── __init__.py
│   │   ├── consensus_matrix.py      # 共识矩阵系统
│   │   ├── role_agents.py          # 角色智能体
│   │   ├── dialogue_manager.py     # 对话管理器
│   │   └── conflict_resolution.py  # 冲突解决
│   │
│   ├── rl/                         # 强化学习模块 (姚刚负责)
│   │   ├── __init__.py
│   │   ├── rl_environment.py       # RL环境
│   │   ├── reward_functions.py     # 奖励函数
│   │   ├── training.py             # 训练逻辑
│   │   └── evaluation.py           # 评估指标
│   │
│   ├── knowledge/                   # 知识系统
│   │   ├── __init__.py
│   │   ├── rag_system.py           # 检索增强生成
│   │   ├── medical_kb.py           # 医学知识库
│   │   └── guideline_parser.py     # 指南解析器
│   │
│   ├── integration/                 # 系统集成
│   │   ├── __init__.py
│   │   ├── workflow_manager.py     # 工作流管理
│   │   ├── api_interfaces.py       # API接口
│   │   └── system_orchestrator.py  # 系统协调器
│   │
│   └── utils/                       # 工具函数
│       ├── __init__.py
│       ├── data_utils.py           # 数据工具
│       ├── visualization.py        # 可视化工具
│       └── metrics.py              # 评估指标
│
├── experiments/                     # 实验脚本 (姚刚负责)
│   ├── __init__.py
│   ├── baseline_comparison.py       # 基线对比实验
│   ├── ablation_studies.py         # 消融实验
│   ├── performance_evaluation.py   # 性能评估
│   └── case_studies/               # 案例研究
│       ├── breast_cancer_cases.py
│       └── lung_cancer_cases.py
│
├── tests/                          # 测试代码
│   ├── __init__.py
│   ├── test_memory/               # 记忆模块测试
│   ├── test_consensus/            # 共识模块测试
│   ├── test_rl/                   # RL模块测试
│   └── test_integration/          # 集成测试
│
├── docs/                          # 文档
│   ├── api_reference.md           # API参考文档
│   ├── user_guide.md              # 用户指南
│   ├── developer_guide.md         # 开发者指南
│   └── paper_drafts/              # 论文草稿
│       └── aaai_2024_draft.md
│
├── scripts/                       # 脚本文件
│   ├── setup_environment.sh       # 环境设置
│   ├── download_data.py           # 数据下载
│   ├── run_experiments.py         # 运行实验
│   └── generate_reports.py        # 生成报告
│
├── notebooks/                     # Jupyter笔记本
│   ├── data_exploration.ipynb     # 数据探索
│   ├── system_demo.ipynb          # 系统演示
│   └── result_analysis.ipynb      # 结果分析
│
├── results/                       # 结果输出
│   ├── figures/                   # 图表
│   ├── logs/                      # 日志文件
│   ├── models/                    # 保存的模型
│   └── reports/                   # 实验报告
│
└── deployment/                    # 部署相关
    ├── docker/                    # Docker配置
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── kubernetes/                # K8s配置
    └── cloud_configs/             # 云部署配置
```

## 文件分工说明

### 🧠 杜军负责的文件
```
src/memory/
├── memory_controller.py      # 主要负责
├── data_evolve.py           # 主要负责  
├── individual_memory.py     # 主要负责
└── group_memory.py          # 主要负责

data/processed/patient_simulated.csv  # 输出文件
```

### 🤝 姚刚负责的文件
```
src/consensus/
├── consensus_matrix.py      # 主要负责
├── role_agents.py          # 主要负责
├── dialogue_manager.py     # 主要负责
└── conflict_resolution.py  # 主要负责

src/rl/
├── rl_environment.py       # 主要负责
├── reward_functions.py     # 主要负责
├── training.py             # 主要负责
└── evaluation.py           # 主要负责

experiments/
├── baseline_comparison.py   # 主要负责
├── performance_evaluation.py # 主要负责
└── ablation_studies.py     # 主要负责
```

### 👨‍💼 Tianyu负责的文件
```
src/integration/           # 系统集成
src/knowledge/            # 知识系统整体设计
config/                   # 配置管理
docs/paper_drafts/        # 论文撰写
```

### 🤝 共同维护的文件
```
src/core/                 # 核心数据模型
src/utils/               # 公用工具
tests/                   # 测试代码
README.md                # 项目文档
```

## 开发工作流

1. **每周同步**：周五提交各自模块的进展
2. **接口定义**：优先定义模块间的接口规范
3. **测试驱动**：每个模块都要有对应的单元测试
4. **文档同步**：代码提交时同步更新API文档
5. **集成测试**：每周进行一次完整的系统集成测试

## Git分支策略

```
main                     # 主分支，稳定版本
├── develop             # 开发分支
├── feature/memory      # 杜军的功能分支
├── feature/consensus   # 姚刚的共识功能分支
├── feature/rl          # 姚刚的RL功能分支
└── feature/integration # Tianyu的集成分支
```