"""
系统健康检查和自动恢复机制
功能：健康状态监控、故障检测、自动恢复、服务可用性检查
作者：系统优化模块
"""

import time
import threading
import psutil
import requests
import sqlite3
import os
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import functools
import json

from .logging_config import get_logger
from .error_handler import GlobalErrorHandler, ErrorSeverity


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """组件类型枚举"""
    DATABASE = "database"
    API_SERVICE = "api_service"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HealthThreshold:
    """健康阈值配置"""
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


class HealthChecker:
    """健康检查器基类"""
    
    def __init__(self, name: str, component_type: ComponentType):
        self.name = name
        self.component_type = component_type
        self.logger = get_logger(f'health_checker.{name}')
    
    def check(self) -> HealthCheckResult:
        """执行健康检查"""
        raise NotImplementedError("Subclasses must implement check method")


class SystemResourceChecker(HealthChecker):
    """系统资源健康检查器"""
    
    def __init__(self):
        super().__init__("system_resources", ComponentType.CUSTOM)
        self.thresholds = {
            'cpu': HealthThreshold(70.0, 90.0, "percent", "CPU使用率"),
            'memory': HealthThreshold(80.0, 95.0, "percent", "内存使用率"),
            'disk': HealthThreshold(85.0, 95.0, "percent", "磁盘使用率")
        }
    
    def check(self) -> HealthCheckResult:
        """检查系统资源"""
        start_time = time.time()
        
        try:
            # CPU检查
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._evaluate_threshold(cpu_percent, self.thresholds['cpu'])
            
            # 内存检查
            memory = psutil.virtual_memory()
            memory_status = self._evaluate_threshold(memory.percent, self.thresholds['memory'])
            
            # 磁盘检查
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._evaluate_threshold(disk_percent, self.thresholds['disk'])
            
            # 综合状态评估
            statuses = [cpu_status, memory_status, disk_status]
            overall_status = max(statuses, key=lambda x: x.value)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=overall_status,
                message=f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk_percent:.1f}%",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3,
                    'disk_percent': disk_percent,
                    'disk_used_gb': disk.used / 1024**3
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"System resource check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def _evaluate_threshold(self, value: float, threshold: HealthThreshold) -> HealthStatus:
        """评估阈值状态"""
        if value >= threshold.critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= threshold.warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class DatabaseChecker(HealthChecker):
    """数据库健康检查器"""
    
    def __init__(self, db_path: str = "data/mdt_system.db"):
        super().__init__("database", ComponentType.DATABASE)
        self.db_path = db_path
    
    def check(self) -> HealthCheckResult:
        """检查数据库连接和基本操作"""
        start_time = time.time()
        
        try:
            # 检查数据库文件是否存在
            if not os.path.exists(self.db_path):
                return HealthCheckResult(
                    component_name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Database file not found: {self.db_path}",
                    timestamp=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # 尝试连接数据库
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                
                # 执行简单查询测试
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
                if result and result[0] == 1:
                    response_time = (time.time() - start_time) * 1000
                    
                    # 获取数据库统计信息
                    cursor.execute("PRAGMA database_list")
                    db_info = cursor.fetchall()
                    
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()[0]
                    
                    return HealthCheckResult(
                        component_name=self.name,
                        component_type=self.component_type,
                        status=HealthStatus.HEALTHY,
                        message=f"Database connection successful, {table_count} tables found",
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        metadata={
                            'db_path': self.db_path,
                            'table_count': table_count,
                            'db_size_mb': os.path.getsize(self.db_path) / 1024**2
                        }
                    )
                else:
                    return HealthCheckResult(
                        component_name=self.name,
                        component_type=self.component_type,
                        status=HealthStatus.CRITICAL,
                        message="Database query test failed",
                        timestamp=datetime.now(),
                        response_time_ms=(time.time() - start_time) * 1000
                    )
                    
        except sqlite3.OperationalError as e:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Database operational error: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )


class APIServiceChecker(HealthChecker):
    """API服务健康检查器"""
    
    def __init__(self, service_url: str, timeout: float = 5.0):
        super().__init__(f"api_service_{service_url}", ComponentType.API_SERVICE)
        self.service_url = service_url
        self.timeout = timeout
    
    def check(self) -> HealthCheckResult:
        """检查API服务可用性"""
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=self.timeout,
                headers={'User-Agent': 'MDT-HealthChecker/1.0'}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = f"API service responding normally (HTTP {response.status_code})"
            elif response.status_code < 500:
                status = HealthStatus.WARNING
                message = f"API service responding with warnings (HTTP {response.status_code})"
            else:
                status = HealthStatus.CRITICAL
                message = f"API service error (HTTP {response.status_code})"
            
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                metadata={
                    'status_code': response.status_code,
                    'response_size': len(response.content),
                    'service_url': self.service_url
                }
            )
            
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"API service timeout after {self.timeout}s",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message="API service connection failed",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                component_name=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"API service check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )


class RecoveryAction:
    """恢复动作"""
    
    def __init__(self, name: str, action: Callable[[], bool], 
                 description: str, max_attempts: int = 3):
        self.name = name
        self.action = action
        self.description = description
        self.max_attempts = max_attempts
        self.attempt_count = 0
        self.last_attempt = None
        self.logger = get_logger(f'recovery_action.{name}')
    
    def execute(self) -> bool:
        """执行恢复动作"""
        if self.attempt_count >= self.max_attempts:
            self.logger.warning(f"Recovery action {self.name} exceeded max attempts ({self.max_attempts})")
            return False
        
        self.attempt_count += 1
        self.last_attempt = datetime.now()
        
        try:
            self.logger.info(f"Executing recovery action: {self.name} (attempt {self.attempt_count})")
            success = self.action()
            
            if success:
                self.logger.info(f"Recovery action {self.name} succeeded")
                self.attempt_count = 0  # 重置计数器
            else:
                self.logger.warning(f"Recovery action {self.name} failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery action {self.name} raised exception: {e}")
            return False
    
    def reset(self):
        """重置恢复动作状态"""
        self.attempt_count = 0
        self.last_attempt = None


class HealthMonitor:
    """健康监控器"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.checkers: List[HealthChecker] = []
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.health_history: List[HealthCheckResult] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.logger = get_logger('health_monitor')
        self.error_handler = GlobalErrorHandler()
        
        # 默认恢复动作
        self._setup_default_recovery_actions()
    
    def add_checker(self, checker: HealthChecker):
        """添加健康检查器"""
        self.checkers.append(checker)
        self.logger.info(f"Added health checker: {checker.name}")
    
    def add_recovery_action(self, component_name: str, action: RecoveryAction):
        """为组件添加恢复动作"""
        if component_name not in self.recovery_actions:
            self.recovery_actions[component_name] = []
        
        self.recovery_actions[component_name].append(action)
        self.logger.info(f"Added recovery action '{action.name}' for component '{component_name}'")
    
    def start_monitoring(self):
        """开始健康监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """停止健康监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_checks(self):
        """执行健康检查"""
        for checker in self.checkers:
            try:
                result = checker.check()
                self.health_history.append(result)
                
                # 保持历史记录在合理范围内
                if len(self.health_history) > 10000:
                    self.health_history = self.health_history[-5000:]
                
                # 处理不健康状态
                if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    self._handle_unhealthy_component(result)
                
            except Exception as e:
                self.logger.error(f"Error checking {checker.name}: {e}")
    
    def _handle_unhealthy_component(self, result: HealthCheckResult):
        """处理不健康的组件"""
        self.logger.warning(f"Unhealthy component detected: {result.component_name} - {result.message}")
        
        # 记录错误
        severity = ErrorSeverity.HIGH if result.status == HealthStatus.CRITICAL else ErrorSeverity.MEDIUM
        self.error_handler.handle_error(
            Exception(f"Health check failed: {result.message}"),
            context={
                'component': result.component_name,
                'status': result.status.value,
                'response_time': result.response_time_ms,
                'severity': severity
            }
        )
        
        # 尝试自动恢复
        if result.component_name in self.recovery_actions:
            self._attempt_recovery(result.component_name)
    
    def _attempt_recovery(self, component_name: str):
        """尝试组件恢复"""
        actions = self.recovery_actions.get(component_name, [])
        
        for action in actions:
            if action.execute():
                self.logger.info(f"Successfully recovered component: {component_name}")
                return True
        
        self.logger.error(f"Failed to recover component: {component_name}")
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取当前健康状态"""
        if not self.health_history:
            return {'status': 'unknown', 'components': []}
        
        # 获取最新的健康检查结果
        latest_results = {}
        for result in reversed(self.health_history):
            if result.component_name not in latest_results:
                latest_results[result.component_name] = result
        
        # 计算整体状态
        component_statuses = [result.status for result in latest_results.values()]
        
        if any(status == HealthStatus.CRITICAL for status in component_statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in component_statuses):
            overall_status = HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in component_statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'components': [asdict(result) for result in latest_results.values()],
            'total_components': len(latest_results)
        }
    
    def get_health_history(self, component_name: Optional[str] = None, 
                          time_window: Optional[timedelta] = None) -> List[HealthCheckResult]:
        """获取健康检查历史"""
        history = self.health_history
        
        if component_name:
            history = [r for r in history if r.component_name == component_name]
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            history = [r for r in history if r.timestamp > cutoff_time]
        
        return history
    
    def _setup_default_recovery_actions(self):
        """设置默认恢复动作"""
        # 内存清理恢复动作
        def cleanup_memory():
            try:
                import gc
                gc.collect()
                return True
            except Exception:
                return False
        
        memory_cleanup = RecoveryAction(
            "memory_cleanup",
            cleanup_memory,
            "Force garbage collection to free memory"
        )
        self.add_recovery_action("system_resources", memory_cleanup)
        
        # 数据库连接重置恢复动作
        def reset_db_connections():
            try:
                # 这里可以添加数据库连接池重置逻辑
                return True
            except Exception:
                return False
        
        db_reset = RecoveryAction(
            "database_reset",
            reset_db_connections,
            "Reset database connections"
        )
        self.add_recovery_action("database", db_reset)


class HealthCheckManager:
    """健康检查管理器"""
    
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
        self.monitor = HealthMonitor()
        self.logger = get_logger('health_check_manager')
        
        # 添加默认检查器
        self._setup_default_checkers()
    
    def _setup_default_checkers(self):
        """设置默认健康检查器"""
        # 系统资源检查器
        self.monitor.add_checker(SystemResourceChecker())
        
        # 数据库检查器
        self.monitor.add_checker(DatabaseChecker())
        
        # 如果有API服务配置，添加API检查器
        # self.monitor.add_checker(APIServiceChecker("http://localhost:8000"))
    
    def start(self):
        """启动健康监控"""
        self.monitor.start_monitoring()
    
    def stop(self):
        """停止健康监控"""
        self.monitor.stop_monitoring()
    
    def add_custom_checker(self, checker: HealthChecker):
        """添加自定义检查器"""
        self.monitor.add_checker(checker)
    
    def add_custom_recovery_action(self, component_name: str, action: RecoveryAction):
        """添加自定义恢复动作"""
        self.monitor.add_recovery_action(component_name, action)
    
    def get_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self.monitor.get_health_status()
    
    def export_health_report(self, file_path: str, time_window: Optional[timedelta] = None):
        """导出健康报告"""
        status = self.get_status()
        history = self.monitor.get_health_history(time_window=time_window)
        
        report = {
            'current_status': status,
            'history_count': len(history),
            'time_window_hours': time_window.total_seconds() / 3600 if time_window else None,
            'history': [asdict(result) for result in history[-100:]]  # 最近100条记录
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        self.logger.info(f"Health report exported to {file_path}")


def get_health_manager() -> HealthCheckManager:
    """获取健康检查管理器实例"""
    return HealthCheckManager()


# 便捷装饰器
def health_check(component_name: str, component_type: ComponentType = ComponentType.CUSTOM):
    """健康检查装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 记录成功的健康检查
                health_result = HealthCheckResult(
                    component_name=component_name,
                    component_type=component_type,
                    status=HealthStatus.HEALTHY,
                    message=f"Function {func.__name__} executed successfully",
                    timestamp=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
                get_health_manager().monitor.health_history.append(health_result)
                return result
                
            except Exception as e:
                # 记录失败的健康检查
                health_result = HealthCheckResult(
                    component_name=component_name,
                    component_type=component_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Function {func.__name__} failed: {str(e)}",
                    timestamp=datetime.now(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
                
                get_health_manager().monitor.health_history.append(health_result)
                raise
        
        return wrapper
    return decorator


# 导出主要接口
__all__ = [
    'HealthStatus',
    'ComponentType',
    'HealthCheckResult',
    'HealthThreshold',
    'HealthChecker',
    'SystemResourceChecker',
    'DatabaseChecker',
    'APIServiceChecker',
    'RecoveryAction',
    'HealthMonitor',
    'HealthCheckManager',
    'get_health_manager',
    'health_check'
]