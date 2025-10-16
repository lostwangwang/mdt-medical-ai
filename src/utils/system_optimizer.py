"""
系统优化器主类
功能：整合日志、错误处理、内存管理、性能监控、健康检查等所有优化组件
作者：系统优化模块
"""

import os
import threading
import atexit
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .logging_config import MDTLogger, get_logger, log_system_info
from .error_handler import GlobalErrorHandler, get_error_handler
from .memory_manager import MemoryManager, get_memory_manager
from .performance_monitor import PerformanceMonitor, get_performance_monitor
from .health_checker import HealthCheckManager, get_health_manager


class SystemOptimizer:
    """系统优化器主类"""
    
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
        self.logger = get_logger('system_optimizer')
        self._components_initialized = False
        
        # 组件实例
        self.error_handler = None
        self.memory_manager = None
        self.performance_monitor = None
        self.health_manager = None
        
        # 注册退出处理
        atexit.register(self.shutdown)
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """初始化系统优化器"""
        if self._components_initialized:
            self.logger.info("System optimizer already initialized")
            return
        
        try:
            self.logger.info("Initializing system optimizer...")
            
            # 记录系统信息
            log_system_info()
            
            # 初始化各个组件
            self._initialize_error_handler(config)
            self._initialize_memory_manager(config)
            self._initialize_performance_monitor(config)
            self._initialize_health_manager(config)
            
            self._components_initialized = True
            self.logger.info("System optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system optimizer: {e}")
            raise
    
    def _initialize_error_handler(self, config: Optional[Dict[str, Any]]):
        """初始化错误处理器"""
        self.error_handler = get_error_handler()
        
        if config and 'error_handler' in config:
            error_config = config['error_handler']
            
            # 配置错误处理器
            if 'max_errors_per_minute' in error_config:
                self.error_handler.max_errors_per_minute = error_config['max_errors_per_minute']
            
            if 'circuit_breaker_threshold' in error_config:
                self.error_handler.circuit_breaker_threshold = error_config['circuit_breaker_threshold']
        
        self.logger.info("Error handler initialized")
    
    def _initialize_memory_manager(self, config: Optional[Dict[str, Any]]):
        """初始化内存管理器"""
        self.memory_manager = get_memory_manager()
        
        if config and 'memory_manager' in config:
            memory_config = config['memory_manager']
            
            # 配置内存管理器
            if 'cache_size_limit' in memory_config:
                self.memory_manager.cache_size_limit = memory_config['cache_size_limit']
            
            if 'gc_threshold' in memory_config:
                self.memory_manager.gc_threshold = memory_config['gc_threshold']
        
        self.logger.info("Memory manager initialized")
    
    def _initialize_performance_monitor(self, config: Optional[Dict[str, Any]]):
        """初始化性能监控器"""
        self.performance_monitor = get_performance_monitor()
        
        if config and 'performance_monitor' in config:
            perf_config = config['performance_monitor']
            
            # 配置性能监控器
            if 'collection_interval' in perf_config:
                self.performance_monitor.system_monitor.collection_interval = perf_config['collection_interval']
        
        self.logger.info("Performance monitor initialized")
    
    def _initialize_health_manager(self, config: Optional[Dict[str, Any]]):
        """初始化健康检查管理器"""
        self.health_manager = get_health_manager()
        
        if config and 'health_manager' in config:
            health_config = config['health_manager']
            
            # 配置健康检查管理器
            if 'check_interval' in health_config:
                self.health_manager.monitor.check_interval = health_config['check_interval']
        
        # 启动健康监控
        self.health_manager.start()
        self.logger.info("Health manager initialized and started")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self._components_initialized:
            return {'status': 'not_initialized'}
        
        try:
            # 获取各组件状态
            health_status = self.health_manager.get_status()
            performance_metrics = self.performance_monitor.get_real_time_metrics()
            memory_stats = self.memory_manager.get_memory_stats()
            error_stats = self.error_handler.get_error_statistics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'operational',
                'health': health_status,
                'performance': performance_metrics,
                'memory': memory_stats,
                'errors': error_stats,
                'components_initialized': self._components_initialized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_logger(self, name: str):
        """获取配置好的日志器"""
        return get_logger(name)
    
    def generate_report(self, output_dir: str = "reports") -> str:
        """生成系统报告（generate_system_report的别名）"""
        return self.generate_system_report(output_dir)
    
    def generate_system_report(self, output_dir: str = "reports", 
                             time_window: Optional[timedelta] = None) -> str:
        """生成系统报告"""
        if not self._components_initialized:
            raise RuntimeError("System optimizer not initialized")
        
        # 创建报告目录
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"system_report_{timestamp}.json")
        
        try:
            # 收集所有报告数据
            system_status = self.get_system_status()
            performance_report = self.performance_monitor.get_performance_report(time_window)
            
            # 导出各组件报告
            perf_file = os.path.join(output_dir, f"performance_metrics_{timestamp}.json")
            health_file = os.path.join(output_dir, f"health_report_{timestamp}.json")
            
            self.performance_monitor.export_metrics(perf_file, time_window)
            self.health_manager.export_health_report(health_file, time_window)
            
            # 生成综合报告
            comprehensive_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'time_window_hours': time_window.total_seconds() / 3600 if time_window else None,
                    'report_files': {
                        'performance': perf_file,
                        'health': health_file
                    }
                },
                'system_status': system_status,
                'performance_summary': performance_report,
                'recommendations': self._generate_recommendations(system_status, performance_report)
            }
            
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"System report generated: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating system report: {e}")
            raise
    
    def _generate_recommendations(self, system_status: Dict[str, Any], 
                                performance_report: Dict[str, Any]) -> List[str]:
        """生成系统优化建议"""
        recommendations = []
        
        try:
            # 健康状态建议
            if system_status.get('health', {}).get('overall_status') == 'critical':
                recommendations.append("系统健康状态严重，建议立即检查关键组件")
            elif system_status.get('health', {}).get('overall_status') == 'warning':
                recommendations.append("系统健康状态警告，建议关注相关组件")
            
            # 性能建议
            perf_metrics = system_status.get('performance', {})
            if perf_metrics.get('cpu_percent', 0) > 80:
                recommendations.append("CPU使用率过高，建议优化计算密集型操作")
            
            if perf_metrics.get('memory_percent', 0) > 85:
                recommendations.append("内存使用率过高，建议清理缓存或增加内存")
            
            # 错误统计建议
            error_stats = system_status.get('errors', {})
            if error_stats.get('total_errors', 0) > 100:
                recommendations.append("错误数量较多，建议检查错误日志并修复问题")
            
            # 函数性能建议
            if 'function_performance' in performance_report:
                slow_functions = performance_report['function_performance'].get('by_avg_time', [])
                if slow_functions and slow_functions[0].avg_time > 1.0:
                    recommendations.append(f"函数 {slow_functions[0].function_name} 执行时间较长，建议优化")
            
            if not recommendations:
                recommendations.append("系统运行正常，无特殊优化建议")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("生成建议时出错，请检查系统状态")
        
        return recommendations
    
    def optimize_system(self):
        """执行系统优化"""
        if not self._components_initialized:
            raise RuntimeError("System optimizer not initialized")
        
        self.logger.info("Starting system optimization...")
        
        try:
            # 内存优化
            self.memory_manager.optimize_memory()
            self.logger.info("Memory optimization completed")
            
            # 清理错误统计
            self.error_handler.clear_old_errors()
            self.logger.info("Error statistics cleanup completed")
            
            # 记录优化完成
            self.logger.info("System optimization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during system optimization: {e}")
            raise
    
    def shutdown(self):
        """关闭系统优化器"""
        if not self._components_initialized:
            return
        
        try:
            self.logger.info("Shutting down system optimizer...")
            
            # 停止健康监控
            if self.health_manager:
                self.health_manager.stop()
            
            # 关闭性能监控
            if self.performance_monitor:
                self.performance_monitor.shutdown()
            
            # 清理内存管理器
            if self.memory_manager:
                self.memory_manager.cleanup()
            
            self.logger.info("System optimizer shutdown completed")
            
        except Exception as e:
            print(f"Error during system optimizer shutdown: {e}")


# 全局系统优化器实例
_system_optimizer = None
_optimizer_lock = threading.Lock()


def get_system_optimizer() -> SystemOptimizer:
    """获取系统优化器实例"""
    global _system_optimizer
    
    if _system_optimizer is None:
        with _optimizer_lock:
            if _system_optimizer is None:
                _system_optimizer = SystemOptimizer()
    
    return _system_optimizer


def initialize_system_optimization(config: Optional[Dict[str, Any]] = None):
    """初始化系统优化"""
    optimizer = get_system_optimizer()
    optimizer.initialize(config)
    return optimizer


# 便捷装饰器
def optimized_function(func=None, *, monitor_performance: bool = True, 
                      handle_errors: bool = True,
                      use_cache: bool = False):
    """系统优化装饰器"""
    def decorator(f):
        # 应用性能监控
        if monitor_performance:
            f = get_performance_monitor().profile()(f)
        
        # 应用错误处理
        if handle_errors:
            f = get_error_handler().error_handler()(f)
        
        # 应用缓存
        if use_cache:
            f = get_memory_manager().cached()(f)
        
        return f
    
    if func is None:
        # 被调用为 @optimized_function(...)
        return decorator
    else:
        # 被调用为 @optimized_function
        return decorator(func)


# 导出主要接口
__all__ = [
    'SystemOptimizer',
    'get_system_optimizer',
    'initialize_system_optimization',
    'optimized_function'
]