# MDT医疗智能体系统 - 用户指南

> 详细的使用指南，帮助您快速上手MDT医疗智能决策系统

## 📚 目录

- [快速开始](#快速开始)
- [系统架构](#系统架构)
- [功能模块](#功能模块)
- [使用场景](#使用场景)
- [API参考](#api参考)
- [配置说明](#配置说明)
- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 环境准备

1. **系统要求**
   ```
   - Python 3.10+
   - 内存: 8GB+ 推荐
   - 存储: 10GB+ 可用空间
   - 操作系统: Windows/macOS/Linux
   ```

2. **安装依赖**
   ```bash
   # 克隆项目
   git clone https://github.com/your-team/mdt-medical-ai.git
   cd mdt-medical-ai
   
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   
   # 安装依赖
   pip install -r requirements.txt
   ```

3. **快速验证**
   ```bash
   # 运行快速启动脚本
   python scripts/quick_start.py --demo
   
   # 或运行系统检查
   python scripts/quick_start.py --check
   ```

### 5分钟上手

1. **运行演示模式**
   ```bash
   python main.py --mode demo
   ```
   这将分析3个预设的患者案例，展示完整的MDT决策过程。

2. **查看结果**
   ```
   results/
   ├── patient_DEMO_001_analysis.json
   ├── patient_DEMO_002_analysis.json
   └── patient_DEMO_003_analysis.json
   ```

3. **理解输出**
   - `recommended_treatment`: 推荐的治疗方案
   - `consensus_score`: 团队共识得分 (-1到+1)
   - `total_rounds`: MDT讨论轮数
   - `conflicts/agreements`: 团队分歧与一致意见

## 🏗️ 系统架构

### 核心组件

```
                    MDT医疗智能体系统
                           │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
  记忆控制器         共识矩阵系统        强化学习
  (Memory)         (Consensus)          (RL)
        │                 │                 │
    ┌───┴───┐         ┌───┴───┐         ┌───┴───┐
  个体记忆  群体记忆   角色对话  冲突解决   环境建模 策略优化
```

### 工作流程

1. **数据输入** → 患者基本信息、实验室结果、生命体征
2. **记忆检索** → 获取个体历史和相似病例
3. **知识检索** → RAG系统搜索医学指南和文献
4. **多智能体对话** → 5个角色进行专业讨论
5. **共识形成** → 量化一致性，识别冲突
6. **决策输出** → 推荐治疗方案及解释
7. **强化学习** → 从决策反馈中学习优化

## 🔧 功能模块

### 1. 患者数据管理

#### 创建患者状态
```python
from src.core.data_models import PatientState
from datetime import datetime

patient = PatientState(
    patient_id="P001",
    age=65,
    diagnosis="breast_cancer",
    stage="II",
    lab_results={
        "creatinine": 1.2,
        "hemoglobin": 11.5,
        "cea": 3.5
    },
    vital_signs={
        "bp_systolic": 140,
        "heart_rate": 78
    },
    symptoms=["fatigue", "pain"],
    comorbidities=["diabetes", "hypertension"],
    psychological_status="anxious",
    quality_of_life_score=0.7,
    timestamp=datetime.now()
)
```

#### 数据验证和清洗
```python
from src.utils.data_processor import DataPipeline

# 创建数据处理管道
pipeline = DataPipeline()

# 处理原始数据
success = pipeline.process_raw_data(
    input_file="data/raw/patients.csv",
    output_file="data/processed/clean_patients.json"
)
```

### 2. 多智能体对话系统

#### 基本使用
```python
from src.consensus.dialogue_manager import MultiAgentDialogueManager
from src.knowledge.rag_system import MedicalKnowledgeRAG

# 初始化系统
rag_system = MedicalKnowledgeRAG()
dialogue_manager = MultiAgentDialogueManager(rag_system)

# 进行MDT讨论
consensus_result = dialogue_manager.conduct_mdt_discussion(patient)

# 获取推荐结果
best_treatment = max(consensus_result.aggregated_scores.items(), key=lambda x: x[1])
print(f"推荐治疗: {best_treatment[0].value}")
print(f"共识得分: {best_treatment[1]:.3f}")
```

#### 查看对话记录
```python
# 获取完整对话记录
transcript = dialogue_manager.get_dialogue_transcript()
print(transcript)

# 对话摘要
if consensus_result.dialogue_summary:
    summary = consensus_result.dialogue_summary
    print(f"总消息数: {summary['total_messages']}")
    print(f"主要话题: {summary['key_topics']}")
```

### 3. 共识分析

#### 生成共识矩阵
```python
from src.consensus.consensus_matrix import ConsensusMatrix

consensus_system = ConsensusMatrix()
result = consensus_system.generate_consensus(patient)

# 查看共识矩阵
print("共识矩阵:")
print(result.consensus_matrix)

# 分析共识模式
analysis = consensus_system.analyze_consensus_patterns(result)
print(f"整体共识水平: {analysis['overall_consensus_level']:.3f}")
```

#### 冲突和一致性分析
```python
# 查看冲突
for conflict in result.conflicts:
    print(f"冲突治疗: {conflict['treatment'].value}")
    print(f"分歧程度: {conflict['variance']:.3f}")
    print(f"冲突角色: {conflict['conflicting_roles']}")

# 查看一致意见
for agreement in result.agreements:
    print(f"一致治疗: {agreement['treatment'].value}")
    print(f"共识得分: {agreement['consensus_score']:+.3f}")
    print(f"一致强度: {agreement['agreement_strength']:.3f}")
```

### 4. 强化学习训练

#### 基础训练
```python
from src.rl.rl_environment import MDTReinforcementLearning, RLTrainer

# 创建RL环境
rl_env = MDTReinforcementLearning(consensus_system)
trainer = RLTrainer(rl_env)

# 开始训练
training_results = trainer.train_dqn(episodes=1000, learning_rate=0.001)

print(f"最终平均奖励: {training_results['final_average_reward']:.3f}")
print(f"最佳奖励: {training_results['best_reward']:.3f}")
```

#### 策略评估
```python
# 评估训练后的策略
evaluation_results = trainer.evaluate_policy(num_episodes=100)

print(f"平均奖励: {evaluation_results['average_reward']:.3f}")
print(f"性能一致性: {evaluation_results['performance_consistency']:.3f}")
```

### 5. 可视化分析

#### 创建分析仪表板
```python
from src.utils.visualization import SystemVisualizer

visualizer = SystemVisualizer()

# 创建患者分析仪表板
dashboard = visualizer.create_patient_analysis_dashboard(patient, consensus_result)

# 保存所有图表
visualizer.save_all_figures(dashboard, "results/figures")
```

#### 训练结果可视化
```python
# 创建训练仪表板
training_dashboard = visualizer.create_training_dashboard(training_results)

# 创建综合报告
summary_report = visualizer.create_summary_report_figure(
    patient_analysis={"patient_info": patient, "consensus_result": consensus_result},
    training_results=training_results
)
```

## 📋 使用场景

### 场景1: 单个患者决策支持

```python
# 完整的单患者分析流程
def analyze_single_patient(patient_data):
    # 1. 创建患者状态
    patient = PatientState(**patient_data)
    
    # 2. 进行MDT讨论
    rag_system = MedicalKnowledgeRAG()
    dialogue_manager = MultiAgentDialogueManager(rag_system)
    consensus_result = dialogue_manager.conduct_mdt_discussion(patient)
    
    # 3. 生成可视化
    visualizer = SystemVisualizer()
    dashboard = visualizer.create_patient_analysis_dashboard(patient, consensus_result)
    
    # 4. 返回结果
    return {
        "recommendation": max(consensus_result.aggregated_scores.items(), key=lambda x: x[1]),
        "consensus_result": consensus_result,
        "visualizations": dashboard
    }

# 使用示例
patient_data = {
    "patient_id": "P001",
    "age": 65,
    "diagnosis": "breast_cancer",
    "stage": "II",
    # ... 其他字段
}

result = analyze_single_patient(patient_data)
print(f"推荐治疗: {result['recommendation'][0].value}")
```

### 场景2: 批量患者分析

```python
def batch_patient_analysis(patients_file, output_dir):
    """批量处理多个患者"""
    # 加载患者数据
    pipeline = DataPipeline()
    pipeline.process_raw_data(patients_file, f"{output_dir}/processed_patients.json")
    
    # 加载处理后的数据
    with open(f"{output_dir}/processed_patients.json", 'r') as f:
        patients_data = json.load(f)
    
    results = []
    for patient_data in patients_data:
        try:
            result = analyze_single_patient(patient_data)
            results.append({
                "patient_id": patient_data["patient_id"],
                "recommended_treatment": result["recommendation"][0].value,
                "consensus_score": result["recommendation"][1],
                "analysis_time": datetime.now()
            })
        except Exception as e:
            logger.error(f"Error analyzing patient {patient_data['patient_id']}: {e}")
    
    # 保存批量结果
    with open(f"{output_dir}/batch_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results
```

### 场景3: 模型对比评估

```python
def run_model_comparison():
    """运行模型对比评估"""
    from experiments.baseline_comparison import ComparisonExperiment
    
    # 创建对比实验
    experiment = ComparisonExperiment()
    
    # 生成测试数据
    test_patients = experiment.generate_test_patients(num_patients=100)
    
    # 运行对比
    results = experiment.run_comparison(num_trials=50)
    
    # 生成报告
    report = experiment.generate_comparison_report()
    
    # 可视化结果
    experiment.plot_comparison_results("results/comparison_plot.png")
    
    return results, report
```

### 场景4: 时序演化分析

```python
def run_temporal_analysis(patient_id, days=30):
    """运行患者时序演化分析"""
    from src.integration.workflow_manager import IntegratedWorkflowManager
    
    # 创建工作流管理器
    workflow = IntegratedWorkflowManager()
    
    # 注册患者
    workflow.register_patient(patient_id, {
        "age": 65,
        "diagnosis": "breast_cancer",
        "initial_stage": "II"
    })
    
    # 运行时序模拟
    simulation_results = workflow.run_temporal_simulation(patient_id, days)
    
    # 生成时序可视化
    visualizer = SystemVisualizer()
    temporal_dashboard = visualizer.create_temporal_analysis_dashboard(simulation_results)
    
    return simulation_results, temporal_dashboard
```

## 🔌 API参考

### 核心API端点

如果您启动了API服务器 (`python -m uvicorn src.api.main:app`):

#### 1. 患者分析
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "patient_id": "P001",
  "age": 65,
  "diagnosis": "breast_cancer",
  "stage": "II",
  "lab_results": {"creatinine": 1.2},
  "vital_signs": {"bp_systolic": 140}
}
```

#### 2. 批量分析
```http
POST /api/v1/batch_analyze
Content-Type: application/json

{
  "patients": [...],
  "options": {
    "include_dialogue": true,
    "generate_visualizations": false
  }
}
```

#### 3. 模型训练
```http
POST /api/v1/train
Content-Type: application/json

{
  "episodes": 1000,
  "learning_rate": 0.001,
  "save_model": true
}
```

### Python API

#### 主要类和方法

```python
# 患者状态
PatientState(patient_id, age, diagnosis, stage, ...)

# 对话管理
MultiAgentDialogueManager(rag_system)
    .conduct_mdt_discussion(patient_state) -> ConsensusResult

# 共识分析  
ConsensusMatrix()
    .generate_consensus(patient_state) -> ConsensusResult
    .analyze_consensus_patterns(consensus_result) -> Dict

# 强化学习
MDTReinforcementLearning(consensus_system)
    .reset(patient_state) -> state_vector
    .step(action) -> (next_state, reward, done, info)

# 可视化
SystemVisualizer()
    .create_patient_analysis_dashboard(patient, result) -> Dict
    .create_training_dashboard(training_results) -> Dict
```

## ⚙️ 配置说明

### 主配置文件 (config/model_config.yaml)

```yaml
# 系统配置
system:
  name: "MDT Memory & Roleplay Framework"
  environment: "development"  # development/testing/production
  debug: true

# 共识系统配置
consensus:
  role_weights:
    oncologist: 1.0
    nurse: 0.8
    psychologist: 0.7
  thresholds:
    conflict_variance: 0.5
    agreement_consensus: 0.3

# RL训练配置
reinforcement_learning:
  training:
    algorithm: "DQN"
    learning_rate: 0.001
    episodes: 1000
```

### 环境变量

```bash
# .env文件
MDT_ENV=development
MDT_LOG_LEVEL=INFO
DATABASE_URL=sqlite:///data/mdt.db
REDIS_URL=redis://localhost:6379/0
```

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **解决方案**: 确保在项目根目录运行，或设置PYTHONPATH
   ```bash
   export PYTHONPATH=/path/to/mdt-medical-ai
   ```

2. **内存不足**
   ```
   MemoryError: Unable to allocate array
   ```
   **解决方案**: 
   - 减少batch_size或num_patients参数
   - 增加系统虚拟内存
   - 使用数据分块处理

3. **GPU相关错误**
   ```
   CUDA out of memory
   ```
   **解决方案**:
   ```python
   # 在config中禁用GPU或减少模型大小
   device = "cpu"  # 强制使用CPU
   ```

4. **依赖包版本冲突**
   ```bash
   # 重新创建虚拟环境
   rm -rf venv
   python -m venv venv
   pip install -r requirements.txt
   ```

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **保存中间结果**
   ```yaml
   # config/model_config.yaml
   development:
     debug:
       save_intermediate_results: true
   ```

3. **性能分析**
   ```python
   import cProfile
   cProfile.run('your_function()', 'profile_output')
   ```

## 💡 最佳实践

### 数据准备

1. **数据质量检查**
   ```python
   # 使用数据验证器
   validator = DataValidator()
   is_valid, errors = validator.validate_patient_state(patient_data)
   
   if not is_valid:
       print("数据验证失败:")
       for error in errors:
           print(f"  - {error}")
   ```

2. **数据标准化**
   ```python
   # 统一数据格式
   pipeline = DataPipeline("config/data_config.yaml")
   cleaned_data = pipeline.process_raw_data(raw_file, output_file)
   ```

### 模型使用

1. **渐进式测试**
   ```python
   # 先测试小样本
   small_results = analyze_patients(patients[:10])
   
   # 验证结果合理性
   if validate_results(small_results):
       full_results = analyze_patients(all_patients)
   ```

2. **结果验证**
   ```python
   # 检查推荐的合理性
   def validate_recommendation(patient, recommendation):
       # 年龄检查
       if patient.age > 80 and recommendation == "surgery":
           return False, "高龄患者手术风险高"
       
       # 并发症检查  
       if "cardiac_dysfunction" in patient.comorbidities and \
          recommendation == "chemotherapy":
           return False, "心功能不全患者化疗需谨慎"
       
       return True, "推荐合理"
   ```

### 性能优化

1. **批处理**
   ```python
   # 批量处理而不是逐个处理
   def batch_process(patients, batch_size=10):
       for i in range(0, len(patients), batch_size):
           batch = patients[i:i+batch_size]
           process_patient_batch(batch)
   ```

2. **缓存机制**
   ```python
   # 缓存RAG检索结果
   @lru_cache(maxsize=1000)
   def cached_knowledge_retrieval(query_hash):
       return rag_system.retrieve(query)
   ```

### 生产环境部署

1. **使用Docker**
   ```bash
   # 构建镜像
   docker build -t mdt-medical-ai .
   
   # 运行容器
   docker run -p 8000:8000 -v /data:/app/data mdt-medical-ai
   ```

2. **监控和日志**
   ```python
   # 添加性能监控
   import time
   
   def timed_analysis(patient):
       start_time = time.time()
       result = analyze_patient(patient)
       duration = time.time() - start_time
       
       logger.info(f"Patient {patient.patient_id} analyzed in {duration:.2f}s")
       return result
   ```

3. **错误处理**
   ```python
   def robust_patient_analysis(patient):
       try:
           return analyze_patient(patient)
       except Exception as e:
           logger.error(f"Analysis failed for {patient.patient_id}: {e}")
           return get_fallback_recommendation(patient)
   ```

## 📞 获取帮助

- **文档**: [项目文档](https://docs.mdt-medical-ai.com)
- **API参考**: [API文档](https://api.mdt-medical-ai.com)  
- **问题反馈**: [GitHub Issues](https://github.com/your-team/mdt-medical-ai/issues)
- **讨论社区**: [GitHub Discussions](https://github.com/your-team/mdt-medical-ai/discussions)
- **邮件支持**: team@mdt-medical-ai.com

## 🎓 延伸阅读

- [开发者指南](developer_guide.md)
- [系统架构设计](architecture.md)
- [API完整参考](api_reference.md)
- [论文和研究](papers/)
- [更新日志](../CHANGELOG.md)

---

*最后更新: 2024年10月*