"""
统一日志记录系统
功能：提供结构化日志、性能监控、错误追踪等功能
作者：系统优化模块
"""

import logging
import logging.handlers
import json
import time
import traceback
import functools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import threading
import queue
import os


@dataclass
class LogEntry:
    """结构化日志条目"""
    timestamp: str
    level: str
    module: str
    function: str
    message: str
    extra_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    error_info: Optional[Dict[str, str]] = None


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """记录性能指标"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'tags': tags or {}
            }
            self.metrics[name].append(metric_entry)
            
            # 保持最近1000条记录
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-800:]
    
    def get_metrics_summary(self, name: str) -> Dict[str, float]:
        """获取指标摘要"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [entry['value'] for entry in self.metrics[name]]
            return {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1]
            }


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def format(self, record):
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            module=record.module if hasattr(record, 'module') else record.name,
            function=record.funcName,
            message=record.getMessage(),
            extra_data=getattr(record, 'extra_data', None),
            performance_metrics=getattr(record, 'performance_metrics', None),
            error_info=getattr(record, 'error_info', None)
        )
        
        return json.dumps(asdict(log_entry), ensure_ascii=False, default=str)


class MDTLogger:
    """MDT系统统一日志记录器"""
    
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
        self.performance_monitor = PerformanceMonitor()
        self.loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 根日志记录器配置
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器 - 一般日志
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mdt_system.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)
        
        # 结构化日志处理器
        structured_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mdt_structured.jsonl",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        structured_handler.setLevel(logging.DEBUG)
        structured_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(structured_handler)
        
        # 错误日志处理器
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mdt_errors.log",
            maxBytes=5*1024*1024,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        root_logger.addHandler(error_handler)
        
        # 性能日志处理器
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / "mdt_performance.log",
            maxBytes=5*1024*1024,
            backupCount=3
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(console_formatter)
        
        # 创建性能专用日志记录器
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.DEBUG)
        perf_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_performance(self, name: str, duration: float, **kwargs):
        """记录性能指标"""
        self.performance_monitor.record_metric(name, duration, kwargs)
        
        perf_logger = logging.getLogger('performance')
        perf_logger.info(
            f"Performance: {name} took {duration:.4f}s",
            extra={
                'performance_metrics': {
                    'name': name,
                    'duration': duration,
                    **kwargs
                }
            }
        )
    
    def log_error(self, logger_name: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误信息"""
        logger = self.get_logger(logger_name)
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        logger.error(
            f"Error occurred: {error}",
            extra={
                'error_info': error_info,
                'extra_data': context
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """获取性能摘要"""
        summary = {}
        for metric_name in self.performance_monitor.metrics:
            summary[metric_name] = self.performance_monitor.get_metrics_summary(metric_name)
        return summary


def performance_monitor(metric_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功执行的性能
                MDTLogger().log_performance(
                    name, 
                    duration,
                    status='success',
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败执行的性能
                MDTLogger().log_performance(
                    name, 
                    duration,
                    status='error',
                    error_type=type(e).__name__
                )
                
                # 记录错误
                MDTLogger().log_error(
                    func.__module__,
                    e,
                    {
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                )
                
                raise
        
        return wrapper
    return decorator


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return MDTLogger().get_logger(name)


def log_system_info():
    """记录系统信息"""
    logger = get_logger('system')
    
    import psutil
    import platform
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }
    
    logger.info(
        "System information logged",
        extra={'extra_data': system_info}
    )


# 初始化日志系统
_mdt_logger = MDTLogger()

# 导出主要接口
__all__ = [
    'MDTLogger',
    'get_logger', 
    'performance_monitor',
    'log_system_info',
    'PerformanceMonitor'
]