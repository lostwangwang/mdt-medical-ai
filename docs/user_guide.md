# 基线评测使用指南

## 1.**安装依赖, 可参考README**
### 环境要求

- Python 3.12+
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
### 2. 数据集
MedQA: https://github.com/jind11/MedQA

PubMedQA: https://github.com/pubmedqa/pubmedqa

DDXPlus: https://github.com/mila-iqia/ddxplus

SymCat: https://github.com/teliov/SymCat-to-synthea

JAMA & Medbullets: https://github.com/HanjieChen/ChallengeClinicalQA

### 3. 项目目录结构主要模块说明
以下是该项目的目录结构及注释说明：

```text
mdt_medical_ai/
├── data/                                    # 数据相关文件
│   └── examples/                            # 示例数据目录
│       ├── prompt.yaml                      # 提示词配置文件（目前为空）
│       └── ...                              # 其他数据集文件
├── experiments/                             # 实验相关代码和评估脚本
│   ├── medqa_types.py                       # MedQA数据类型定义和处理工具
│   ├── evaluation_ddxplus/                  # DDXPlus数据集评估脚本
│   │   ├── convert_evidences.py             # 证据解码工具
│   │   └── evalucation_ddxplus.py          # DDXPlus评估主脚本
│   ├── evaluation_medbullets/               # MedBullets数据集评估脚本
│   │   └── evalucation_medbullets.py       # MedBullets评估主脚本
│   ├── evaluation_medqa/                    # MedQA数据集评估脚本
│   │   └── evalucation_medqa.py            # MedQA评估主脚本
│   ├── evaluation_pubmedqa/                 # PubMedQA数据集评估脚本
│   │   └── evalucation_pubmedqa.py         # PubMedQA评估主脚本
│   ├── evaluation_symcat/                   # SymCat数据集评估脚本
│   │   └── evaluation_symcat.py            # SymCat评估主脚本
│   ├── medbullets/                          # MedBullets实验相关代码
│   │   └── one_agent_medbullets_evalucation.py  # 单智能体检评MedBullets数据集
│   └── one_agent_evaluation/                # 各数据集的单智能体检评实现
│       ├── ddxplus/                         # DDXPlus单智能体检评
│       │   ├── evidences.py                 # 证据处理工具
│       │   ├── one_agent_evaluation.py      # 主评估脚本
│       │   └── prompt.yaml                  # DDXPlus提示词模板
│       ├── llm/                             # LLM客户端封装
│       │   ├── llm_client.py                # LLM客户端实现
│       │   └── prompt.yaml                  # LLM通用提示词模板
│       ├── medqa/                           # MedQA单智能体检评
│       │   └── one_agent_evaluation.py      # 主评估脚本
│       ├── pubmedqa/                        # PubMedQA单智能体检评
│       │   └── one_agent_evaluation_pubmedqa.py  # 主评估脚本
│       └── symcat/                          # SymCat单智能体检评
│           ├── create_patients_data.py      # 创建患者数据脚本
│           ├── one_agent_evaluation.py      # 主评估脚本
│           └── prompt.yaml                  # SymCat提示词模板
└── src/                                     # 核心源代码（根据引用推断）
    ├── consensus/                           # 多智能体对话管理模块
    │   └── dialogue_manager.py              # 对话管理器实现
    ├── knowledge/                           # 医学知识库系统
    │   └── rag_system.py                    # RAG知识检索系统
    ├── tools/                               # 工具类模块
    │   ├── read_files.py                    # 文件读取工具
    │   └── list_to_options.py               # 列表转选项工具
    └── utils/                               # 工具函数和接口
        └── llm_interface.py                 # LLM接口封装
```


### 项目主要功能模块说明：

1. **数据处理模块 (`data/`)**
   - 存放各种医学问答数据集的示例文件
   - 包括 [prompt.yaml](file://E:\project\LLM\mdt_medical_ai\data\examples\prompt.yaml) 配置文件用于定义提示词模板

2. **实验评估模块 (`experiments/`)**
   - 包含多个医学问答数据集的评估脚本：
     - `ddxplus`: 差别诊断数据集
     - `medbullets`: 医学要点数据集
     - `medqa`: 医学问答数据集
     - `pubmedqa`: PubMed问答数据集
     - `symcat`: 症状检查数据集
   - 每个数据集都有对应的单智能体和多智能体检评脚本

3. **核心功能模块 (`src/`)**
   - `consensus/dialogue_manager.py`: 多智能体对话管理系统
   - `knowledge/rag_system.py`: 基于RAG的医学知识检索系统
   - `utils/llm_interface.py`: LLM接口封装
   - `tools/`: 各种辅助工具函数

4. **基线比较模块 ([baseline_comparison.py](file://E:\project\LLM\mdt_medical_ai\experiments\baseline_comparison.py))**
   - 实现多种基线模型的比较评估
   - 包括随机模型、规则模型、RAG模型等多种基线

这个项目的整体目标是构建一个多智能体医学问答系统，并在多个医学问答数据集上进行评估和比较。

## 4. 运行评测代码(举`evaluation_medqa.py`的代码的例子)

#### 1.首先在项目根目录创建 `.env`文件
> ![NOTE] 提交代码时，检查一下不要提交`.env`敏感信息
```shell
# 本地私有部署的qwen
QWEN_API_KEY=""
MODEL_NAME=""
BASE_URL=""
```

#### 2. 代码逻辑
##### evaluation_medqa.py 深层函数调用分析

##### 主要函数及其深层调用关系

##### read_jsonl
- **作用**: 读取并解析 JSONL 格式文件，支持随机采样
- **参数**:
  - `file_path`: JSONL 文件路径
  - `random_sample`: 随机采样数量
- **返回值**: 解析后的 JSON 对象列表
- **内部调用**:
  - `open`: 打开文件
  - `f.readlines()`: 读取所有行
  - `random.sample`: 随机采样（当指定采样数时）
  - `json.loads`: 解析每一行 JSON 字符串

##### 主程序逻辑 (`if __name__ == "__main__"`)

###### 数据准备阶段
1. **路径拼接**: 使用相对路径定位数据文件
2. **调用 [read_jsonl]: 读取 MedQA 数据集，随机采样1条记录

###### 核心处理循环
对每一条数据项进行以下操作：

##### 1. 创建 [MedicalQuestionState](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L6-L14)
- **作用**: 封装医学问题的各项属性
- **参数**: 
  - [patient_id](file://E:\project\LLM\mdt_medical_ai\src\core\data_models.py#L268-L268): 病人ID（索引）
  - [question](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L10-L10): 问题文本
  - [options](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L11-L11): 选项字典
  - [answer](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L12-L12): 答案文本
  - [meta_info](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L13-L13): 元信息
  - [answer_idx](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L14-L14): 正确答案索引

##### 2. 调用 `medqa_types.init_question_option`
- **作用**: 动态创建 [QuestionOption](file://E:\project\LLM\mdt_medical_ai\experiments\medqa_types.py#L98-L98) 枚举类
- **参数**: 选项字典
- **内部机制**: 
  - 使用 Python 的 `Enum` 构造函数动态创建枚举类
  - 将选项内容作为枚举成员值

##### 3. 初始化各组件
- **LLM 配置**:
  - 创建 [LLMConfig](file://E:\project\LLM\mdt_medical_ai\src\utils\llm_interface.py#L37-L54) 对象，封装模型配置
  - 创建 [LLMInterface](file://E:\project\LLM\mdt_medical_ai\src\utils\llm_interface.py#L57-L1417) 对象，封装 LLM 接口
- **知识系统**:
  - 创建 [MedicalKnowledgeRAG](file://E:\project\LLM\mdt_medical_ai\src\knowledge\rag_system.py#L19-L628) 对象，用于医学知识检索
- **对话管理**:
  - 创建 [MultiAgentDialogueManager](file://E:\project\LLM\mdt_medical_ai\src\consensus\dialogue_manager.py#L32-L1468) 对象，管理多智能体对话

##### 4. 调用 [MultiAgentDialogueManager.conduct_mdt_discussion_medqa](file://E:\project\LLM\mdt_medical_ai\src\consensus\dialogue_manager.py#L54-L118)
- **作用**: 执行多智能体医学问答讨论
- **参数**:
  - `question_state`: 医学问题状态
  - `question_options`: 问题选项枚举列表
- **深层调用链**:
  - 初始化各专业医生代理（CardiologistAgent, RespiratoryPhysicianAgent等）
  - 为每个代理构建角色提示词
  - 循环进行多轮讨论：
    - 调用各代理的 `generate_response` 方法生成回答
    - 更新共识矩阵
    - 检查是否达成共识或达到最大轮次
  - 生成最终共识结果

##### 5. 调用逻辑
###### 5.1 MultiAgentDialogueManager.conduct_mdt_discussion_medqa 
- **作用**: 进行多智能体讨论
- **参数**:
  - `question_state`:医学问题状态
  - `question_options`: 问题选项枚举列表
###### 5.2 MulTiAgentDialogueManager._initialize_discussion_medqa
- **作用**: 初始化多智能体生成初始意见
- **参数**: 
  - `question_state`: 医学问题状态
  - `question_options`: 问题选项枚举列表
- **内部调用**:
- `_generate_initial_opinion_medqa`: 生成初始意见
  - `_generate_reasoning_medqa`: 生成推理，返回一个json格式的数据
    - `self.llm_interface.generate_treatment_reasoning_medqa`: 调用LLM接口生成推理
      - `_build_treatment_reasoning_prompt_medqa`: 这里是写prompt的，就在这里修改
- `_create_initial_message_medqa`: 创建初始消息
    - `_generate_llm_initial_message_meqa`: 调用LLM进行生成初始消息
      - `agent.llm_interface.generate_focus_treatment_reasoning_meqa`: 基于初始化意见生成发言
        - `_build_focus_treatment_reasoning_prompt_meqa`: 在这里修改prompt

###### 5.3. 进行多轮对话
* 进行多轮对话:
* 判定条件：最少进行一次讨论，判断观点是否一致( `_check_discussion_convergence_medqa`)
* 循环进行多轮对话：
- 内部调用
  - 进行一轮多轮对话
    - `_conduct_dialogue_round_medqa`: 进行一轮多轮对话
      - `_conduct_sequential_presentation_medqa`: 进行顺序对话
        - `agent.generate_dialogue_response_medqa`: 生成对话
          - `_construct_response_medqa`: 构建回应内容
            - `self.llm_interface.generate_dialogue_response_medqa`
              - `_build_dialogue_response_prompt_medqa`: prompt
    -  `self._update_agent_opinions_medqa`: 更新智能体的观点
      - `agent._update_agent_opinions_and_preferences_medqa`: 根据当前轮次的对话内容,更新角色的医疗问题偏好和医疗问题意见以及置信度
        - `self._generate_update_agent_opinions_reasoning_medqa`: 根据当前轮次的对话内容,生成更新角色医疗问题意见的推理
          - `self.llm_interface.generate_update_agent_opinions_reasoning_medqa`: 生成更新角色有关医学问题的推理
            - `_build_update_agent_opinions_reasoning_prompt_medqa`: prompt

###### 5.4 生成共识矩阵以及判断收敛
- `_check_discussion_convergence_medqa`: 计算 Kendall’s W 协调系数