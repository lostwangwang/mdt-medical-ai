"""
性能监控和指标收集系统
功能：实时性能监控、指标收集、性能分析和报告
作者：系统优化模块
"""

import time
import threading
import psutil
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import functools

from .logging_config import get_logger


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_threads: int
    open_files: int


@dataclass
class FunctionMetrics:
    """函数性能指标"""
    function_name: str
    module_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    error_count: int
    last_called: datetime


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.function_metrics: Dict[str, FunctionMetrics] = {}
        self.lock = threading.RLock()
        self.logger = get_logger('metrics_collector')
    
    def record_metric(self, metric: PerformanceMetric):
        """记录性能指标"""
        with self.lock:
            self.metrics[metric.name].append(metric)
    
    def record_function_call(self, function_name: str, module_name: str, 
                           execution_time: float, success: bool = True):
        """记录函数调用指标"""
        with self.lock:
            key = f"{module_name}.{function_name}"
            
            if key not in self.function_metrics:
                self.function_metrics[key] = FunctionMetrics(
                    function_name=function_name,
                    module_name=module_name,
                    call_count=0,
                    total_time=0.0,
                    avg_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    error_count=0,
                    last_called=datetime.now()
                )
            
            metrics = self.function_metrics[key]
            metrics.call_count += 1
            metrics.total_time += execution_time
            metrics.avg_time = metrics.total_time / metrics.call_count
            metrics.min_time = min(metrics.min_time, execution_time)
            metrics.max_time = max(metrics.max_time, execution_time)
            metrics.last_called = datetime.now()
            
            if not success:
                metrics.error_count += 1
    
    def get_metric_history(self, metric_name: str, 
                          time_window: Optional[timedelta] = None) -> List[PerformanceMetric]:
        """获取指标历史"""
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            metrics = list(self.metrics[metric_name])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp > cutoff_time]
            
            return metrics
    
    def get_metric_summary(self, metric_name: str, 
                          time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """获取指标摘要统计"""
        history = self.get_metric_history(metric_name, time_window)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'avg': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1]
        }
    
    def get_function_metrics(self) -> Dict[str, FunctionMetrics]:
        """获取函数性能指标"""
        with self.lock:
            return dict(self.function_metrics)
    
    def get_top_functions(self, sort_by: str = 'total_time', limit: int = 10) -> List[FunctionMetrics]:
        """获取性能排名前N的函数"""
        with self.lock:
            metrics_list = list(self.function_metrics.values())
            
            if sort_by == 'total_time':
                metrics_list.sort(key=lambda x: x.total_time, reverse=True)
            elif sort_by == 'avg_time':
                metrics_list.sort(key=lambda x: x.avg_time, reverse=True)
            elif sort_by == 'call_count':
                metrics_list.sort(key=lambda x: x.call_count, reverse=True)
            elif sort_by == 'error_count':
                metrics_list.sort(key=lambda x: x.error_count, reverse=True)
            
            return metrics_list[:limit]


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.monitoring_active = False
        self.monitor_thread = None
        self.logger = get_logger('system_monitor')
        self.last_network_stats = None
    
    def start_monitoring(self):
        """开始系统监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """停止系统监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(PerformanceMetric(
                name="system.cpu.percent",
                value=cpu_percent,
                unit="percent",
                timestamp=datetime.now(),
                tags={"type": "system"}
            ))
            
            # 内存使用
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(PerformanceMetric(
                name="system.memory.percent",
                value=memory.percent,
                unit="percent",
                timestamp=datetime.now(),
                tags={"type": "system"}
            ))
            
            self.metrics_collector.record_metric(PerformanceMetric(
                name="system.memory.used",
                value=memory.used / 1024 / 1024,  # MB
                unit="MB",
                timestamp=datetime.now(),
                tags={"type": "system"}
            ))
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric(PerformanceMetric(
                name="system.disk.percent",
                value=(disk.used / disk.total) * 100,
                unit="percent",
                timestamp=datetime.now(),
                tags={"type": "system"}
            ))
            
            # 网络IO
            network = psutil.net_io_counters()
            if self.last_network_stats:
                bytes_sent_rate = (network.bytes_sent - self.last_network_stats.bytes_sent) / self.collection_interval
                bytes_recv_rate = (network.bytes_recv - self.last_network_stats.bytes_recv) / self.collection_interval
                
                self.metrics_collector.record_metric(PerformanceMetric(
                    name="system.network.bytes_sent_rate",
                    value=bytes_sent_rate,
                    unit="bytes/sec",
                    timestamp=datetime.now(),
                    tags={"type": "system"}
                ))
                
                self.metrics_collector.record_metric(PerformanceMetric(
                    name="system.network.bytes_recv_rate",
                    value=bytes_recv_rate,
                    unit="bytes/sec",
                    timestamp=datetime.now(),
                    tags={"type": "system"}
                ))
            
            self.last_network_stats = network
            
            # 进程信息
            process = psutil.Process()
            self.metrics_collector.record_metric(PerformanceMetric(
                name="process.threads.count",
                value=process.num_threads(),
                unit="count",
                timestamp=datetime.now(),
                tags={"type": "process"}
            ))
            
            self.metrics_collector.record_metric(PerformanceMetric(
                name="process.files.open",
                value=len(process.open_files()),
                unit="count",
                timestamp=datetime.now(),
                tags={"type": "process"}
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.logger = get_logger('performance_profiler')
    
    def profile_function(self, tags: Optional[Dict[str, str]] = None):
        """函数性能分析装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    execution_time = time.time() - start_time
                    
                    # 记录函数调用指标
                    self.metrics_collector.record_function_call(
                        func.__name__,
                        func.__module__,
                        execution_time,
                        success
                    )
                    
                    # 记录详细性能指标
                    metric_tags = {"function": func.__name__, "module": func.__module__}
                    if tags:
                        metric_tags.update(tags)
                    
                    self.metrics_collector.record_metric(PerformanceMetric(
                        name="function.execution_time",
                        value=execution_time,
                        unit="seconds",
                        timestamp=datetime.now(),
                        tags=metric_tags,
                        metadata={
                            "args_count": len(args),
                            "kwargs_count": len(kwargs),
                            "success": success
                        }
                    ))
            
            return wrapper
        return decorator
    
    def get_performance_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """生成性能报告"""
        if time_window is None:
            time_window = timedelta(hours=1)
        
        # 系统指标摘要
        system_metrics = {}
        for metric_name in ["system.cpu.percent", "system.memory.percent", "system.disk.percent"]:
            summary = self.metrics_collector.get_metric_summary(metric_name, time_window)
            if summary:
                system_metrics[metric_name] = summary
        
        # 函数性能排名
        top_functions = {
            'by_total_time': self.metrics_collector.get_top_functions('total_time', 10),
            'by_avg_time': self.metrics_collector.get_top_functions('avg_time', 10),
            'by_call_count': self.metrics_collector.get_top_functions('call_count', 10),
            'by_error_count': self.metrics_collector.get_top_functions('error_count', 5)
        }
        
        # 函数执行时间分布
        execution_time_summary = self.metrics_collector.get_metric_summary(
            "function.execution_time", time_window
        )
        
        return {
            'report_time': datetime.now().isoformat(),
            'time_window_hours': time_window.total_seconds() / 3600,
            'system_metrics': system_metrics,
            'function_performance': top_functions,
            'execution_time_distribution': execution_time_summary
        }


class PerformanceMonitor:
    """性能监控主类"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.system_monitor = SystemMonitor()
        self.profiler = PerformanceProfiler()
        self.logger = get_logger('performance_monitor')
        
        # 自动启动系统监控
        self.system_monitor.start_monitoring()
    
    def profile(self, tags: Optional[Dict[str, str]] = None):
        """性能分析装饰器"""
        return self.profiler.profile_function(tags)
    
    def record_custom_metric(self, name: str, value: float, unit: str = "", 
                           tags: Optional[Dict[str, str]] = None):
        """记录自定义指标"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.system_monitor.metrics_collector.record_metric(metric)
    
    def get_performance_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取性能报告"""
        return self.profiler.get_performance_report(time_window)
    
    def export_metrics(self, file_path: str, time_window: Optional[timedelta] = None):
        """导出指标到文件"""
        report = self.get_performance_report(time_window)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        self.logger.info(f"Performance metrics exported to {file_path}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / 1024 / 1024,
                'memory_available_mb': memory.available / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting real-time metrics: {e}")
            return {}
    
    def shutdown(self):
        """关闭性能监控"""
        self.system_monitor.stop_monitoring()
        self.logger.info("Performance monitoring shutdown")


def get_performance_monitor() -> PerformanceMonitor:
    """获取性能监控器实例"""
    return PerformanceMonitor()


# 便捷装饰器
def monitor_performance(tags: Optional[Dict[str, str]] = None):
    """性能监控装饰器"""
    return get_performance_monitor().profile(tags)


# 导出主要接口
__all__ = [
    'PerformanceMonitor',
    'PerformanceMetric',
    'SystemMetrics',
    'FunctionMetrics',
    'MetricsCollector',
    'SystemMonitor',
    'PerformanceProfiler',
    'get_performance_monitor',
    'monitor_performance'
]