# 系统优化功能文档

## 概述

MDT医疗智能体系统集成了全面的系统优化功能，包括统一日志、全局错误处理、内存优化、性能监控和健康检查。这些功能旨在提高系统的可靠性、性能和可维护性。

## 核心组件

### 1. 系统优化器 (SystemOptimizer)

位置：`src/utils/system_optimizer.py`

系统优化器是所有优化功能的核心协调器，提供：
- 统一的组件管理
- 系统初始化和关闭
- 性能报告生成
- 组件间协调

#### 主要方法：
- `initialize()`: 初始化所有优化组件
- `get_logger(name)`: 获取统一配置的日志器
- `generate_report()`: 生成系统性能报告
- `shutdown()`: 优雅关闭所有组件

### 2. 统一日志系统 (UnifiedLogger)

位置：`src/utils/unified_logger.py`

提供结构化、多级别的日志记录功能：

#### 特性：
- 多种日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
- 结构化日志格式
- 文件和控制台双重输出
- 自动日志轮转
- 性能指标记录

#### 使用示例：
```python
from src.utils.system_optimizer import get_system_optimizer

system_optimizer = get_system_optimizer()
logger = system_optimizer.get_logger(__name__)

logger.info("系统启动", extra={"component": "main", "action": "startup"})
logger.error("处理错误", extra={"error_type": "validation", "patient_id": "P001"})
```

### 3. 全局错误处理 (GlobalErrorHandler)

位置：`src/utils/error_handler.py`

提供统一的错误处理和恢复机制：

#### 特性：
- 分类错误处理（系统、业务、网络等）
- 自动错误恢复策略
- 错误统计和分析
- 装饰器模式的错误处理

#### 使用示例：
```python
from src.utils.system_optimizer import get_system_optimizer

system_optimizer = get_system_optimizer()
error_handler = system_optimizer.error_handler

@error_handler.error_handler
def risky_function():
    # 可能出错的代码
    pass
```

### 4. 内存管理 (MemoryManager)

位置：`src/utils/memory_manager.py`

提供智能内存管理和优化：

#### 特性：
- 内存使用监控
- 自动垃圾回收
- 缓存管理
- 内存泄漏检测

#### 使用示例：
```python
from src.utils.system_optimizer import get_system_optimizer

system_optimizer = get_system_optimizer()
memory_manager = system_optimizer.memory_manager

@memory_manager.cached
def expensive_computation(data):
    # 计算密集型操作
    return result
```

### 5. 性能监控 (PerformanceMonitor)

位置：`src/utils/performance_monitor.py`

提供实时性能监控和分析：

#### 特性：
- 函数执行时间监控
- 系统资源使用跟踪
- 性能瓶颈识别
- 性能趋势分析

#### 使用示例：
```python
from src.utils.system_optimizer import optimized_function

@optimized_function
def monitored_function():
    # 需要监控的函数
    pass
```

### 6. 健康检查 (HealthChecker)

位置：`src/utils/health_checker.py`

提供系统健康状态监控：

#### 特性：
- 组件健康检查
- 系统资源监控
- 自动故障检测
- 健康报告生成

## 集成使用

### 在主应用中集成

```python
# main.py
from src.utils.system_optimizer import get_system_optimizer, optimized_function

# 获取系统优化器
system_optimizer = get_system_optimizer()
logger = system_optimizer.get_logger(__name__)

def main():
    try:
        # 初始化系统优化器
        system_optimizer.initialize()
        logger.info("系统优化器启动成功")
        
        # 应用逻辑
        # ...
        
    finally:
        # 生成性能报告
        try:
            report = system_optimizer.generate_report()
            logger.info("系统性能报告已生成", extra={"report_path": report.get("report_path")})
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
        
        # 关闭系统优化器
        system_optimizer.shutdown()
        logger.info("系统优化器已关闭")
```

### 在API服务中集成

```python
# backend/api/main.py
from fastapi import FastAPI
from src.utils.system_optimizer import get_system_optimizer, optimized_function

system_optimizer = get_system_optimizer()
logger = system_optimizer.get_logger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    system_optimizer.initialize()
    logger.info("API服务启动，系统优化器已初始化")

@app.post("/api/endpoint")
@optimized_function
async def api_endpoint():
    # API逻辑
    pass
```

## 配置选项

系统优化器支持通过环境变量或配置文件进行配置：

### 环境变量：
- `LOG_LEVEL`: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE_PATH`: 日志文件路径
- `ENABLE_PERFORMANCE_MONITORING`: 是否启用性能监控
- `MEMORY_THRESHOLD`: 内存使用阈值
- `HEALTH_CHECK_INTERVAL`: 健康检查间隔（秒）

### 配置示例：
```bash
export LOG_LEVEL=INFO
export LOG_FILE_PATH=/var/log/mdt_system.log
export ENABLE_PERFORMANCE_MONITORING=true
export MEMORY_THRESHOLD=80
export HEALTH_CHECK_INTERVAL=60
```

## 性能报告

系统优化器会生成详细的性能报告，包括：

- 系统运行时间
- 内存使用统计
- 函数执行性能
- 错误统计
- 健康检查结果

报告格式支持JSON和HTML，可用于系统监控和性能分析。

## 最佳实践

1. **统一日志使用**：所有组件都应使用系统优化器提供的日志器
2. **性能监控**：为关键函数添加`@optimized_function`装饰器
3. **错误处理**：使用全局错误处理器处理异常
4. **内存管理**：定期调用内存清理，使用缓存装饰器
5. **健康检查**：定期检查系统健康状态

## 故障排除

### 常见问题：

1. **日志文件权限问题**
   - 确保应用有写入日志目录的权限
   - 检查日志文件路径是否正确

2. **内存使用过高**
   - 检查缓存配置
   - 调用内存清理方法
   - 检查是否有内存泄漏

3. **性能监控数据缺失**
   - 确认已添加`@optimized_function`装饰器
   - 检查性能监控是否已启用

4. **健康检查失败**
   - 检查系统资源使用情况
   - 验证各组件是否正常运行

## 扩展开发

系统优化器采用模块化设计，支持扩展新的优化组件：

1. 创建新的优化组件类
2. 在`SystemOptimizer`中注册组件
3. 实现必要的接口方法
4. 更新配置和文档

详细的扩展开发指南请参考开发者文档。