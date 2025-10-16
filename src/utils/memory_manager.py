"""
内存管理和缓存系统
功能：内存优化、缓存管理、内存池等
作者：系统优化模块
"""

import gc
import weakref
import threading
import time
import psutil
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from functools import wraps
import functools
import pickle
import hashlib
import json
from datetime import datetime, timedelta

from .logging_config import get_logger, performance_monitor

T = TypeVar('T')


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class MemoryStats:
    """内存统计信息"""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    cache_memory: int
    gc_collections: Dict[int, int]


class LRUCache(Generic[T]):
    """LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self.stats = CacheStats(max_size=max_size)
        self.lock = threading.RLock()
        self.logger = get_logger('cache')
    
    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - timestamp > self.ttl
    
    def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                if self._is_expired(timestamp):
                    del self.cache[key]
                    self.stats.misses += 1
                    self.stats.size -= 1
                    return None
                
                # 移动到末尾（最近使用）
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return value
            
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: T) -> None:
        """存储缓存值"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # 更新现有值
                self.cache[key] = (value, current_time)
                self.cache.move_to_end(key)
            else:
                # 添加新值
                if len(self.cache) >= self.max_size:
                    # 移除最旧的项
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.stats.evictions += 1
                    self.stats.size -= 1
                
                self.cache[key] = (value, current_time)
                self.stats.size += 1
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats.size = 0
    
    def cleanup_expired(self) -> int:
        """清理过期项"""
        if self.ttl is None:
            return 0
        
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (value, timestamp) in self.cache.items():
                if current_time - timestamp > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.size -= 1
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        return self.stats


class MemoryPool:
    """内存池管理"""
    
    def __init__(self, initial_size: int = 100):
        self.pool: List[Any] = []
        self.lock = threading.Lock()
        self.logger = get_logger('memory_pool')
        self.created_objects = 0
        self.reused_objects = 0
        
        # 预分配对象
        for _ in range(initial_size):
            self.pool.append(self._create_object())
    
    def _create_object(self) -> Any:
        """创建新对象（子类实现）"""
        return {}
    
    def acquire(self) -> Any:
        """获取对象"""
        with self.lock:
            if self.pool:
                obj = self.pool.pop()
                self.reused_objects += 1
                return obj
            else:
                self.created_objects += 1
                return self._create_object()
    
    def release(self, obj: Any) -> None:
        """释放对象"""
        # 重置对象状态
        if isinstance(obj, dict):
            obj.clear()
        elif hasattr(obj, 'reset'):
            obj.reset()
        
        with self.lock:
            if len(self.pool) < 1000:  # 限制池大小
                self.pool.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'created_objects': self.created_objects,
                'reused_objects': self.reused_objects,
                'reuse_rate': self.reused_objects / (self.created_objects + self.reused_objects) if (self.created_objects + self.reused_objects) > 0 else 0.0
            }


class MemoryManager:
    """内存管理器"""
    
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
        self.logger = get_logger('memory_manager')
        self.caches: Dict[str, LRUCache] = {}
        self.memory_pools: Dict[str, MemoryPool] = {}
        self.weak_refs: weakref.WeakSet = weakref.WeakSet()
        self.monitoring_enabled = True
        self.last_gc_time = time.time()
        self.gc_threshold = 60  # 60秒
        
        # 启动内存监控线程
        self._start_monitoring()
    
    def _start_monitoring(self):
        """启动内存监控"""
        def monitor():
            while self.monitoring_enabled:
                try:
                    self._check_memory_usage()
                    self._cleanup_caches()
                    self._trigger_gc_if_needed()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def get_cache(self, name: str, max_size: int = 1000, ttl: Optional[float] = None) -> LRUCache:
        """获取或创建缓存"""
        if name not in self.caches:
            self.caches[name] = LRUCache(max_size=max_size, ttl=ttl)
        return self.caches[name]
    
    def get_memory_pool(self, name: str, initial_size: int = 100) -> MemoryPool:
        """获取或创建内存池"""
        if name not in self.memory_pools:
            self.memory_pools[name] = MemoryPool(initial_size=initial_size)
        return self.memory_pools[name]
    
    def register_object(self, obj: Any) -> None:
        """注册对象用于弱引用跟踪"""
        try:
            self.weak_refs.add(obj)
        except TypeError:
            # 某些对象不支持弱引用
            pass
    
    def get_memory_stats(self) -> MemoryStats:
        """获取内存统计信息"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        # 计算缓存占用的内存
        cache_memory = 0
        for cache in self.caches.values():
            cache_memory += cache.stats.size * 1024  # 估算每个缓存项1KB
        
        # GC统计
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        return MemoryStats(
            total_memory=memory.total,
            available_memory=memory.available,
            used_memory=process.memory_info().rss,
            memory_percent=memory.percent,
            cache_memory=cache_memory,
            gc_collections=gc_stats
        )
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        stats = self.get_memory_stats()
        
        # 记录内存统计
        self.logger.info(
            f"Memory usage: {stats.memory_percent:.1f}%, "
            f"Cache memory: {stats.cache_memory / 1024 / 1024:.1f}MB"
        )
        
        # 内存使用过高时的处理
        if stats.memory_percent > 85:
            self.logger.warning("High memory usage detected, triggering cleanup")
            self._emergency_cleanup()
    
    def _cleanup_caches(self):
        """清理缓存"""
        for name, cache in self.caches.items():
            expired_count = cache.cleanup_expired()
            if expired_count > 0:
                self.logger.debug(f"Cleaned {expired_count} expired items from cache '{name}'")
    
    def _trigger_gc_if_needed(self):
        """根据需要触发垃圾回收"""
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_threshold:
            collected = gc.collect()
            self.last_gc_time = current_time
            if collected > 0:
                self.logger.debug(f"Garbage collection freed {collected} objects")
    
    def _emergency_cleanup(self):
        """紧急内存清理"""
        # 清理所有缓存的一半内容
        for cache in self.caches.values():
            with cache.lock:
                items_to_remove = len(cache.cache) // 2
                for _ in range(items_to_remove):
                    if cache.cache:
                        oldest_key = next(iter(cache.cache))
                        del cache.cache[oldest_key]
                        cache.stats.size -= 1
                        cache.stats.evictions += 1
        
        # 强制垃圾回收
        collected = gc.collect()
        self.logger.info(f"Emergency cleanup completed, freed {collected} objects")
    
    def get_cache_stats(self) -> Dict[str, CacheStats]:
        """获取所有缓存统计"""
        return {name: cache.get_stats() for name, cache in self.caches.items()}
    
    def get_pool_stats(self) -> Dict[str, Dict[str, int]]:
        """获取所有内存池统计"""
        return {name: pool.get_stats() for name, pool in self.memory_pools.items()}
    
    def clear_all_caches(self):
        """清空所有缓存"""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("All caches cleared")
    
    def cached(self, ttl: int = 3600):
        """缓存装饰器"""
        def decorator(func: Callable) -> Callable:
            cache = self.get_cache(func.__name__, ttl=ttl)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                key = cache._generate_key(func.__name__, *args, **kwargs)
                
                # 尝试从缓存获取
                result = cache.get(key)
                if result is not None:
                    return result
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                cache.put(key, result)
                return result
            
            return wrapper
        return decorator
    
    def cleanup(self):
        """清理资源"""
        try:
            self.clear_all_caches()
            for pool in self.memory_pools.values():
                pool.pool.clear()
            self.logger.info("Memory manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during memory manager cleanup: {e}")
    
    def shutdown(self):
        """关闭内存管理器"""
        self.monitoring_enabled = False
        self.cleanup()


def cached(cache_name: str = "default", max_size: int = 1000, ttl: Optional[float] = None):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        manager = MemoryManager()
        cache = manager.get_cache(cache_name, max_size, ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = cache._generate_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            result = cache.get(key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.put(key, result)
            
            return result
        
        # 添加缓存管理方法
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    return decorator


@performance_monitor("memory_optimization")
def optimize_memory():
    """内存优化函数"""
    manager = MemoryManager()
    
    # 清理过期缓存
    manager._cleanup_caches()
    
    # 触发垃圾回收
    collected = gc.collect()
    
    # 获取内存统计
    stats = manager.get_memory_stats()
    
    logger = get_logger('memory_optimizer')
    logger.info(
        f"Memory optimization completed: "
        f"freed {collected} objects, "
        f"memory usage: {stats.memory_percent:.1f}%"
    )
    
    return {
        'freed_objects': collected,
        'memory_percent': stats.memory_percent,
        'cache_memory_mb': stats.cache_memory / 1024 / 1024
    }


def get_memory_manager() -> MemoryManager:
    """获取内存管理器实例"""
    return MemoryManager()


# 导出主要接口
__all__ = [
    'MemoryManager',
    'LRUCache',
    'MemoryPool',
    'MemoryStats',
    'CacheStats',
    'cached',
    'optimize_memory',
    'get_memory_manager'
]