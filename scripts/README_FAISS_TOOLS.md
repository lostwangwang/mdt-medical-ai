# FAISS数据库查看工具使用说明

本目录包含了三个用于查看和分析FAISS数据库的Python脚本，可以帮助您查看`index.faiss`和`index.pkl`文件中的患者数据。

## 工具概览

### 1. `faiss_data_viewer.py` - 完整功能查看器
**功能最全面的工具，支持交互式操作**

**主要功能：**
- 检查FAISS文件状态
- 分析患者数据统计
- 交互式查看患者详细信息
- 向量数据分析
- 数据导出功能

**使用方法：**
```bash
# 交互模式（推荐）
python scripts/faiss_data_viewer.py

# 查看特定患者
python scripts/faiss_data_viewer.py --patient-id 11489167

# 导出数据
python scripts/faiss_data_viewer.py --export exported_data

# 非交互模式
python scripts/faiss_data_viewer.py --no-interactive

# 指定数据库路径
python scripts/faiss_data_viewer.py --db-path /path/to/clinical_memory_db
```

### 2. `quick_faiss_viewer.py` - 快速查看器
**轻量级工具，适合快速查看数据**

**主要功能：**
- 快速检查文件状态
- 显示患者列表
- 查看患者基本信息
- 简化的交互界面

**使用方法：**
```bash
python scripts/quick_faiss_viewer.py
```

### 3. `export_faiss_data.py` - 数据导出工具
**专门用于数据导出和格式转换**

**主要功能：**
- 导出为JSON格式
- 导出为CSV格式
- 生成汇总报告
- 支持多种导出选项

**使用方法：**
```bash
# 导出所有格式
python scripts/export_faiss_data.py

# 只导出JSON
python scripts/export_faiss_data.py --format json

# 只导出CSV
python scripts/export_faiss_data.py --format csv

# 指定输出目录
python scripts/export_faiss_data.py --output my_export_dir

# 不生成汇总报告
python scripts/export_faiss_data.py --no-report
```

## 数据结构说明

### 患者数据包含以下字段：

- **基本信息**
  - `subject_id`: 患者ID
  - `gender`: 性别
  - `anchor_age`: 年龄

- **医疗历史**
  - `allergies`: 过敏史列表
  - `cancer_history`: 癌症病史
  - `chronic_diseases`: 慢性疾病列表

- **检验数据**
  - `baseline_labs`: 基线检验结果
  - `baseline_vitals`: 基线生命体征

- **药物信息**
  - `discharge_medications`: 出院药物列表

- **病情总结**
  - `daily_summaries`: 包含病情趋势和总结

## 输出文件说明

### JSON导出文件：
- `all_patients_full.json`: 所有患者完整数据
- `individual_patients/patient_*.json`: 每个患者的单独文件
- `summary_report.json`: 数据库汇总报告

### CSV导出文件：
- `patients_basic_info.csv`: 患者基本信息
- `chronic_diseases.csv`: 慢性疾病详情
- `baseline_labs.csv`: 基线检验数据
- `medications.csv`: 药物信息

### 报告文件：
- `summary_report.txt`: 文本格式汇总报告
- `summary_report.json`: JSON格式汇总报告

## 使用建议

### 首次使用：
1. 先运行 `quick_faiss_viewer.py` 快速了解数据概况
2. 使用 `faiss_data_viewer.py` 进行详细分析
3. 根据需要使用 `export_faiss_data.py` 导出数据

### 常见用途：

**查看数据概况：**
```bash
python scripts/quick_faiss_viewer.py
```

**详细分析特定患者：**
```bash
python scripts/faiss_data_viewer.py --patient-id YOUR_PATIENT_ID
```

**导出数据进行进一步分析：**
```bash
python scripts/export_faiss_data.py --output analysis_data
```

**交互式探索：**
```bash
python scripts/faiss_data_viewer.py
```

## 依赖要求

### 必需依赖：
- `pickle` (Python标准库)
- `json` (Python标准库)
- `pathlib` (Python标准库)

### 可选依赖：
- `faiss`: 用于FAISS索引分析
- `pandas`: 用于CSV导出
- `numpy`: 用于数值计算

### 安装可选依赖：
```bash
pip install faiss-cpu pandas numpy
```

## 故障排除

### 常见问题：

1. **文件不存在错误**
   - 确保 `clinical_memory_db/index.pkl` 文件存在
   - 检查文件路径是否正确

2. **JSON解析错误**
   - 数据文件可能损坏，尝试重新生成

3. **FAISS库未安装**
   - 运行 `pip install faiss-cpu` 安装FAISS库
   - 或者忽略FAISS相关功能，只使用PKL数据

4. **内存不足**
   - 对于大型数据库，建议使用导出工具分批处理

### 获取帮助：
```bash
python scripts/faiss_data_viewer.py --help
python scripts/export_faiss_data.py --help
```

## 示例输出

### 患者列表示例：
```
序号 患者ID       性别   年龄   慢性疾病  检验项目  病情趋势
------------------------------------------------------------
1    11489167    M      65     3        15       improving
2    11489168    F      72     5        12       stable
3    11489169    M      58     2        18       worsening
```

### 汇总报告示例：
```
FAISS数据库汇总报告
==================
导出时间: 2024-01-15 10:30:00

患者统计:
- 总患者数: 150
- 性别分布: {'M': 85, 'F': 65}
- 年龄范围: 18 - 95 岁 (平均: 64.2)

医疗数据统计:
- 慢性疾病总数: 450 (平均每人: 3.0)
- 基线检验总数: 2250 (平均每人: 15.0)
- 出院药物总数: 750 (平均每人: 5.0)
```

### 数据记录
```shell
👥 患者列表:
--------------------------------------------------------------------------------
序号   患者ID         性别     年龄     慢性疾病     检验     药物     趋势        
--------------------------------------------------------------------------------
1    14840724     F      42     1        0      20     stable    
2    19747913     F      82     5        0      13     stable    
3    17749823     F      61     0        24     26     worsen    
4    19913456     F      54     0        41     87     worsen    
5    16082161     F      38     0        5      24     improve   
6    11489167     F      55     5        12     11     stable    
7    13166211     F      39     2        0      17     stable    
8    19737081     F      50     2        37     57     improve   
9    18206867     F      46     0        22     27     worsen    
10   18194799     F      40     0        0      8      stable    

============================================================
📋 患者详细信息示例 (患者ID: 14840724)
============================================================
👤 基本信息:
  患者ID: 14840724
  性别: F
  年龄: 42

🚫 过敏史 (1 项):
  - 

🎗️ 癌症病史 (2 项):
  - Secondary and unspecified malignant neoplasm of lymph nodes of axilla and upper limb (ICD9: 1963)
  - Malignant neoplasm of breast (female), unspecified (ICD9: 1749)

🏥 慢性疾病 (1 项):
  - Tobacco use disorder (ICD9: 3051)

💓 基线生命体征:
  - temp: 98.7
  - heart_rate: 56
  - sbp: 90
  - dbp: 60
  - resp_rate: 18
  - spo2: 99

💊 出院药物 (20 项):
  - OxycoDONE (Immediate Release)  (10-20, PO/NG)
  - TraMADOL (Ultram) (100, PO)
  - Lorazepam (0.5-1, PO/NG)
  - Acetaminophen (1000, PO/NG)
  - Diazepam (5, PO/NG)
  ... 还有 15 项

📊 病情总结:
  趋势: stable
  总结: 结构化记录较少，依据出院生命体征：体温98.7，心率56，血压90/60，呼吸18，血氧99%。整体趋于稳定。

============================================================
📋 有慢性疾病的患者示例 (患者ID: 14840724)
============================================================
👤 基本信息:
  患者ID: 14840724
  性别: F
  年龄: 42

🚫 过敏史 (1 项):
  - 

🎗️ 癌症病史 (2 项):
  - Secondary and unspecified malignant neoplasm of lymph nodes of axilla and upper limb (ICD9: 1963)
  - Malignant neoplasm of breast (female), unspecified (ICD9: 1749)

🏥 慢性疾病 (1 项):
  - Tobacco use disorder (ICD9: 3051)

💓 基线生命体征:
  - temp: 98.7
  - heart_rate: 56
  - sbp: 90
  - dbp: 60
  - resp_rate: 18
  - spo2: 99

💊 出院药物 (20 项):
  - OxycoDONE (Immediate Release)  (10-20, PO/NG)
  - TraMADOL (Ultram) (100, PO)
  - Lorazepam (0.5-1, PO/NG)
  - Acetaminophen (1000, PO/NG)
  - Diazepam (5, PO/NG)
  ... 还有 15 项

📊 病情总结:
  趋势: stable
  总结: 结构化记录较少，依据出院生命体征：体温98.7，心率56，血压90/60，呼吸18，血氧99%。整体趋于稳定。
```

### 所有数据
```shell
🏥 FAISS数据库患者数据演示
============================================================
✅ 成功加载数据
📄 文档数量: 10
🔗 索引映射: 10
✅ 解析出 10 位患者数据

📊 数据库统计:
  性别分布: {'F': 10}
  年龄范围: 38 - 82 岁 (平均: 50.7)
  慢性疾病总数: 15
  基线检验总数: 141
  出院药物总数: 290

👥 患者列表:
--------------------------------------------------------------------------------
序号   患者ID         性别     年龄     慢性疾病     检验     药物     趋势        
--------------------------------------------------------------------------------
1    14840724     F      42     1        0      20     stable    
2    19747913     F      82     5        0      13     stable    
3    17749823     F      61     0        24     26     worsen    
4    19913456     F      54     0        41     87     worsen    
5    16082161     F      38     0        5      24     improve   
6    11489167     F      55     5        12     11     stable    
7    13166211     F      39     2        0      17     stable    
8    19737081     F      50     2        37     57     improve   
9    18206867     F      46     0        22     27     worsen    
10   18194799     F      40     0        0      8      stable    
```