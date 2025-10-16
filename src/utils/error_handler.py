"""
全局错误处理机制
功能：统一异常处理、错误恢复、故障转移等
作者：系统优化模块
"""

import functools
import traceback
import time
from typing import Dict, Any, Optional, Callable, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime, timedelta

from .logging_config import get_logger


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 轻微错误，不影响主要功能
    MEDIUM = "medium"     # 中等错误，影响部分功能
    HIGH = "high"         # 严重错误，影响核心功能
    CRITICAL = "critical" # 致命错误，系统无法继续运行


class ErrorCategory(Enum):
    """错误类别"""
    VALIDATION = "validation"       # 数据验证错误
    COMPUTATION = "computation"     # 计算错误
    NETWORK = "network"            # 网络错误
    DATABASE = "database"          # 数据库错误
    MEMORY = "memory"              # 内存错误
    TIMEOUT = "timeout"            # 超时错误
    PERMISSION = "permission"      # 权限错误
    CONFIGURATION = "configuration" # 配置错误
    UNKNOWN = "unknown"            # 未知错误


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    traceback: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0


class ErrorRecoveryStrategy:
    """错误恢复策略"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def should_retry(self, error: Exception, retry_count: int) -> bool:
        """判断是否应该重试"""
        if retry_count >= self.max_retries:
            return False
        
        # 网络错误和超时错误通常可以重试
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        # 内存错误不建议重试
        if isinstance(error, MemoryError):
            return False
        
        # 其他错误根据具体情况判断
        return retry_count < 2
    
    def get_retry_delay(self, retry_count: int) -> float:
        """获取重试延迟时间（指数退避）"""
        return self.retry_delay * (2 ** retry_count)
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """尝试错误恢复"""
        # 默认恢复策略：返回None或空值
        if "default_value" in context:
            return context["default_value"]
        
        # 根据错误类型返回合适的默认值
        if isinstance(error, (ValueError, TypeError)):
            return None
        
        return None


class CircuitBreaker:
    """熔断器模式"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数"""
        with self.lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """判断是否应该尝试重置熔断器"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.timeout
    
    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class GlobalErrorHandler:
    """全局错误处理器"""
    
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
        self.logger = get_logger('error_handler')
        self.error_history: List[ErrorInfo] = []
        self.recovery_strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_count = 0
        self.lock = threading.Lock()
        
        # 注册默认恢复策略
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """注册默认恢复策略"""
        self.recovery_strategies[ConnectionError] = ErrorRecoveryStrategy(max_retries=3, retry_delay=2.0)
        self.recovery_strategies[TimeoutError] = ErrorRecoveryStrategy(max_retries=2, retry_delay=5.0)
        self.recovery_strategies[ValueError] = ErrorRecoveryStrategy(max_retries=1, retry_delay=0.5)
        self.recovery_strategies[MemoryError] = ErrorRecoveryStrategy(max_retries=0, retry_delay=0.0)
    
    def register_recovery_strategy(self, error_type: Type[Exception], strategy: ErrorRecoveryStrategy):
        """注册错误恢复策略"""
        self.recovery_strategies[error_type] = strategy
    
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """获取熔断器"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]
    
    def categorize_error(self, error: Exception) -> ErrorCategory:
        """错误分类"""
        if isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, MemoryError):
            return ErrorCategory.MEMORY
        elif isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION
        elif isinstance(error, (ZeroDivisionError, ArithmeticError)):
            return ErrorCategory.COMPUTATION
        else:
            return ErrorCategory.UNKNOWN
    
    def assess_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """评估错误严重程度"""
        # 根据错误类型和上下文评估严重程度
        if isinstance(error, MemoryError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ValueError, TypeError)):
            # 检查是否在关键路径中
            if context.get('critical_path', False):
                return ErrorSeverity.HIGH
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """处理错误"""
        context = context or {}
        
        with self.lock:
            self.error_count += 1
            error_id = f"ERR_{self.error_count:06d}"
        
        # 创建错误信息
        error_info = ErrorInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=self.assess_severity(error, context),
            category=self.categorize_error(error),
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context
        )
        
        # 记录错误
        self.logger.error(
            f"Error {error_id}: {error}",
            extra={
                'error_info': {
                    'error_id': error_id,
                    'severity': error_info.severity.value,
                    'category': error_info.category.value,
                    'error_type': error_info.error_type
                },
                'extra_data': context
            }
        )
        
        # 保存错误历史
        with self.lock:
            self.error_history.append(error_info)
            # 保持最近1000个错误记录
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-800:]
        
        return error_info
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any], retry_count: int = 0) -> Any:
        """尝试错误恢复"""
        error_type = type(error)
        
        # 查找匹配的恢复策略
        strategy = None
        for registered_type, registered_strategy in self.recovery_strategies.items():
            if issubclass(error_type, registered_type):
                strategy = registered_strategy
                break
        
        if strategy is None:
            strategy = ErrorRecoveryStrategy()  # 使用默认策略
        
        # 判断是否应该重试
        if strategy.should_retry(error, retry_count):
            delay = strategy.get_retry_delay(retry_count)
            self.logger.info(f"Retrying after {delay}s (attempt {retry_count + 1})")
            time.sleep(delay)
            return "retry"
        
        # 尝试恢复
        try:
            recovery_result = strategy.recover(error, context)
            self.logger.info(f"Error recovery successful: {recovery_result}")
            return recovery_result
        except Exception as recovery_error:
            self.logger.error(f"Error recovery failed: {recovery_error}")
            return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        with self.lock:
            if not self.error_history:
                return {}
            
            # 按类别统计
            category_counts = {}
            severity_counts = {}
            recent_errors = []
            
            # 最近24小时的错误
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for error_info in self.error_history:
                # 类别统计
                category = error_info.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # 严重程度统计
                severity = error_info.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                # 最近错误
                if error_info.timestamp > cutoff_time:
                    recent_errors.append({
                        'error_id': error_info.error_id,
                        'timestamp': error_info.timestamp.isoformat(),
                        'severity': severity,
                        'category': category,
                        'message': error_info.message
                    })
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors_24h': len(recent_errors),
                'category_distribution': category_counts,
                'severity_distribution': severity_counts,
                'recent_errors': recent_errors[-10:],  # 最近10个错误
                'error_rate': len(recent_errors) / 24.0  # 每小时错误率
            }
    
    def clear_old_errors(self, max_age_hours: int = 24):
        """清理旧的错误记录"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            old_count = len(self.error_history)
            
            # 保留最近的错误记录
            self.error_history = [
                error for error in self.error_history 
                if error.timestamp > cutoff_time
            ]
            
            cleared_count = old_count - len(self.error_history)
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} old error records")
            
            return cleared_count
    
    def error_handler(self, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.UNKNOWN,
                     recovery_strategy: Optional[ErrorRecoveryStrategy] = None):
        """错误处理装饰器方法"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 处理错误
                    context = {
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                    
                    # 将严重程度和类别添加到上下文中
                    context['severity'] = severity
                    context['category'] = category
                    self.handle_error(e, context)
                    
                    # 尝试恢复策略
                    if recovery_strategy:
                        try:
                            recovery_result = recovery_strategy.recover(e, context)
                            if recovery_result is not None:
                                self.logger.info(f"Error recovery successful for {func.__name__}")
                                return recovery_result
                        except Exception as recovery_error:
                            self.logger.error(f"Error recovery failed: {recovery_error}")
                    
                    raise
            
            return wrapper
        return decorator


def error_handler(
    default_value: Any = None,
    reraise: bool = False,
    circuit_breaker: Optional[str] = None,
    critical_path: bool = False
):
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = GlobalErrorHandler()
            retry_count = 0
            max_retries = 3
            
            while retry_count <= max_retries:
                try:
                    # 使用熔断器（如果指定）
                    if circuit_breaker:
                        breaker = handler.get_circuit_breaker(circuit_breaker)
                        return breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs),
                        'retry_count': retry_count,
                        'critical_path': critical_path,
                        'default_value': default_value
                    }
                    
                    # 处理错误
                    error_info = handler.handle_error(e, context)
                    
                    # 尝试恢复
                    recovery_result = handler.attempt_recovery(e, context, retry_count)
                    
                    if recovery_result == "retry" and retry_count < max_retries:
                        retry_count += 1
                        continue
                    elif recovery_result is not None and recovery_result != "retry":
                        return recovery_result
                    
                    # 如果需要重新抛出异常
                    if reraise:
                        raise
                    
                    # 返回默认值
                    return default_value
            
            # 超过最大重试次数
            if reraise:
                raise Exception(f"Function {func.__name__} failed after {max_retries} retries")
            return default_value
        
        return wrapper
    return decorator


def get_error_handler() -> GlobalErrorHandler:
    """获取全局错误处理器"""
    return GlobalErrorHandler()


# 导出主要接口
__all__ = [
    'GlobalErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorInfo',
    'ErrorRecoveryStrategy',
    'CircuitBreaker',
    'error_handler',
    'get_error_handler'
]