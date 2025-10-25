# 基线实验完整指南

## 概述

本指南详细说明如何为MDT医疗AI系统设计和运行基线对比实验，包括数据准备、实验运行和结果分析。

## 1. 数据准备

### 1.1 生成示例数据

首先运行数据准备脚本生成示例数据：

```bash
cd /mnt/e/project/LLM/mdt_medical_ai
python scripts/prepare_benchmark_data.py
```

这将在 `data/examples/` 目录下生成：
- `medqa_sample.jsonl`: MedQA格式的多选题数据
- `pubmedqa_sample.jsonl`: PubMedQA格式的是非题数据  
- `mdt_patients.json`: MDT系统专用的患者数据
- `dataset_info.json`: 数据集信息描述

### 1.2 准备真实数据集

如果你有真实的医学数据集，可以使用转换脚本：

```bash
# 转换MedQA-USMLE数据
python scripts/convert_datasets.py --dataset medqa --input /path/to/medqa.jsonl --output data/examples/medqa_real.jsonl

# 转换PubMedQA数据
python scripts/convert_datasets.py --dataset pubmedqa --input /path/to/pubmedqa.json --output data/examples/pubmedqa_real.jsonl

# 转换DDXPlus诊断数据
python scripts/convert_datasets.py --dataset ddxplus --input /path/to/ddxplus.csv --output data/examples/ddxplus.jsonl
```

### 1.3 数据格式说明

#### MedQA格式 (多选题)
```json
{
  "id": "medqa_001",
  "question": "A 65-year-old woman with...",
  "options": ["CT scan", "MRI", "Bone scan", "PET scan"],
  "answer_idx": 2
}
```

#### PubMedQA格式 (是非题)
```json
{
  "id": "pubmedqa_001", 
  "question": "Does chemotherapy improve survival?",
  "context": "Recent studies have shown...",
  "answer": "yes"
}
```

#### MDT患者格式
```json
{
  "patient_id": "MDT_PATIENT_001",
  "age": 65,
  "diagnosis": "breast_cancer",
  "stage": "II",
  "lab_results": {...},
  "expert_recommendation": {
    "primary_treatment": "surgery",
    "confidence": 0.85
  }
}
```

## 2. 基线实验类型

### 2.1 数据集基线

测试简单基线模型在标准医学数据集上的表现：

- **随机基线**: 随机选择答案
- **多数类基线**: 总是选择最常见的答案
- **关键词启发式**: 基于关键词匹配的简单规则

### 2.2 系统基线

测试不同医疗AI系统架构的表现：

- **随机模型**: 随机推荐治疗方案
- **规则模型**: 基于医学指南的规则系统
- **单智能体**: 单一专科医生模拟
- **RAG模型**: 检索增强生成模型
- **Med-PaLM类**: 大型医学语言模型模拟
- **MDT系统**: 我们的多智能体对话系统

## 3. 运行实验

### 3.1 数据集基线实验

```bash
# 运行单个基线
python experiments/dataset_benchmark.py \
  --dataset medqa \
  --data-path data/examples/medqa_sample.jsonl \
  --baseline random \
  --output results/medqa_random.json

# 运行所有基线
python experiments/dataset_benchmark.py \
  --dataset medqa \
  --data-path data/examples/medqa_sample.jsonl \
  --baseline all \
  --output results/medqa_all_baselines.json
```

### 3.2 系统基线实验

```bash
# 运行系统对比
python experiments/baseline_comparison.py
```

### 3.3 统一评估入口

```bash
# 只运行数据集基线
python experiments/performance_evaluation.py \
  --mode dataset \
  --dataset medqa \
  --data-path data/examples/medqa_sample.jsonl \
  --max-samples 500

# 只运行系统对比
python experiments/performance_evaluation.py \
  --mode system \
  --num-patients 50 \
  --num-trials 30

# 运行完整评估
python experiments/performance_evaluation.py \
  --mode both \
  --dataset pubmedqa \
  --data-path data/examples/pubmedqa_sample.jsonl \
  --num-patients 100 \
  --num-trials 50
```

## 4. 结果分析

### 4.1 输出文件

实验会生成以下文件：

- **JSON文件**: 详细的实验结果和配置
- **CSV文件**: 表格格式的结果，便于进一步分析
- **PNG图表**: 可视化对比结果
- **TXT报告**: 人类可读的详细报告
- **MD综合报告**: 包含所有实验的综合分析

### 4.2 关键指标

#### 数据集基线指标
- **准确率 (Accuracy)**: 正确答案的比例
- **响应时间**: 模型推理时间
- **一致性**: 多次运行的结果一致性

#### 系统基线指标
- **准确率**: 与专家推荐的一致性
- **共识对齐度**: 与多专家共识的匹配程度
- **响应时间**: 系统响应速度
- **一致性评分**: 多次运行的稳定性
- **解释质量**: 推理解释的质量评分
- **置信度校准**: 预测置信度与实际准确率的匹配度

### 4.3 结果解读

#### 良好表现的指标
- 准确率 > 0.7 (数据集基线)
- 准确率 > 0.8 (系统基线)
- 一致性 > 0.9
- 响应时间 < 2秒
- 置信度校准 > 0.8

#### 性能对比
- MDT系统应该在大多数指标上优于简单基线
- 特别关注解释质量和共识对齐度
- 响应时间可能比简单模型慢，但应在可接受范围内

## 5. 扩展和定制

### 5.1 添加新的基线模型

在 `experiments/baseline_comparison.py` 中添加新的基线类：

```python
class YourNewBaseline(BaselineModelInterface):
    def __init__(self):
        super().__init__("Your New Baseline")
    
    def predict(self, patient_state: PatientState) -> Dict[str, Any]:
        # 实现你的预测逻辑
        pass
```

### 5.2 添加新的数据集

1. 在 `experiments/dataset_benchmark.py` 中添加加载器：

```python
def load_your_dataset(path: str, max_samples: Optional[int] = None):
    # 实现数据加载逻辑
    pass
```

2. 在 `run_dataset_benchmark` 函数中添加支持

### 5.3 自定义评估指标

在相应的评估函数中添加新的指标计算：

```python
def eval_with_custom_metrics(baseline, examples):
    # 添加自定义指标
    custom_score = calculate_custom_metric(predictions, ground_truth)
    return {
        "accuracy": accuracy,
        "custom_metric": custom_score,
        # ... 其他指标
    }
```

## 6. 最佳实践

### 6.1 实验设计
- 使用固定随机种子确保可重现性
- 运行多次实验取平均值
- 记录实验配置和环境信息
- 使用统计显著性测试

### 6.2 数据管理
- 保持训练/验证/测试集分离
- 定期备份实验结果
- 使用版本控制管理数据和代码
- 记录数据来源和预处理步骤

### 6.3 结果报告
- 包含所有相关指标
- 提供误差条和置信区间
- 分析失败案例
- 讨论局限性和改进方向

## 7. 故障排除

### 7.1 常见问题

**数据加载失败**
- 检查文件路径和格式
- 确认数据编码为UTF-8
- 验证JSON/JSONL格式正确性

**内存不足**
- 减少 `--max-samples` 参数
- 分批处理大数据集
- 增加系统内存或使用更小的模型

**模型预测失败**
- 检查依赖包版本
- 确认模型文件完整性
- 查看详细错误日志

### 7.2 性能优化

- 使用GPU加速（如果可用）
- 并行处理多个样本
- 缓存中间结果
- 优化数据加载流程

## 8. 持续集成

建议设置定期的基线检查：

```bash
# 每日基线检查脚本
#!/bin/bash
cd /mnt/e/project/LLM/mdt_medical_ai

# 运行快速基线检查
python experiments/performance_evaluation.py \
  --mode both \
  --dataset medqa \
  --data-path data/examples/medqa_sample.jsonl \
  --max-samples 100 \
  --num-patients 20 \
  --num-trials 10 \
  --output-dir results/daily_check

# 发送结果通知（可选）
python scripts/send_results_notification.py results/daily_check
```

这样可以及时发现系统性能退化或改进。