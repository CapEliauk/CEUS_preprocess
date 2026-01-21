"""
内存管理模块 - 支持memmap和智能内存管理
"""
import gc
import os
import psutil
import threading
import time
import tempfile
from typing import Optional, Callable, Dict, Tuple
import numpy as np

from config import config
from utils.logger import get_logger

logger = get_logger('memory')


class MemmapManager:
    """内存映射管理器"""

    def __init__(self):
        self._temp_dir = tempfile.mkdtemp(prefix='ceus_memmap_')
        self._files: Dict[str, str] = {}
        self._arrays: Dict[str, np.memmap] = {}
        self._lock = threading.Lock()

    def create_memmap(self, name: str, shape: Tuple, dtype=np.float32) -> np.memmap:
        """创建内存映射数组"""
        with self._lock:
            filepath = os.path.join(self._temp_dir, f'{name}.dat')

            arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape)

            self._files[name] = filepath
            self._arrays[name] = arr

            logger.debug(f"创建memmap: {name}, shape={shape}, dtype={dtype}")
            return arr

    def get_memmap(self, name: str) -> Optional[np.memmap]:
        """获取已有的memmap"""
        return self._arrays.get(name)

    def delete_memmap(self, name: str):
        """删除memmap"""
        with self._lock:
            if name in self._arrays:
                del self._arrays[name]
            if name in self._files:
                try:
                    os.remove(self._files[name])
                except:
                    pass
                del self._files[name]

    def cleanup(self):
        """清理所有memmap"""
        with self._lock:
            self._arrays.clear()
            for filepath in self._files.values():
                try:
                    os.remove(filepath)
                except:
                    pass
            self._files.clear()

            try:
                os.rmdir(self._temp_dir)
            except:
                pass

        logger.debug("Memmap清理完成")


class FrameBuffer:
    """帧缓冲区 - LRU策略"""

    def __init__(self, max_frames: int = None):
        self.max_frames = max_frames or config.memory.FRAME_BUFFER_SIZE
        self._frames: Dict[int, np.ndarray] = {}
        self._access_order: list = []
        self._lock = threading.Lock()

    def add(self, frame_idx: int, frame: np.ndarray):
        """添加帧"""
        with self._lock:
            if frame_idx in self._frames:
                self._access_order.remove(frame_idx)
            elif len(self._frames) >= self.max_frames:
                oldest = self._access_order.pop(0)
                del self._frames[oldest]

            self._frames[frame_idx] = frame.copy()
            self._access_order.append(frame_idx)

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取帧"""
        with self._lock:
            if frame_idx in self._frames:
                self._access_order.remove(frame_idx)
                self._access_order.append(frame_idx)
                return self._frames[frame_idx].copy()
        return None

    def clear(self):
        """清空"""
        with self._lock:
            self._frames.clear()
            self._access_order.clear()
            gc.collect()


class MemoryManager:
    """内存管理器单例"""

    _instance: Optional['MemoryManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.max_memory_percent = config.memory.MAX_MEMORY_PERCENT
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: list = []
        self._memmap_manager = MemmapManager()

        logger.info(f"内存管理器初始化，最大内存使用: {self.max_memory_percent}%")

    @property
    def memmap(self) -> MemmapManager:
        return self._memmap_manager

    def get_memory_usage(self) -> float:
        """获取当前进程内存使用百分比"""
        return psutil.Process().memory_percent()

    def get_system_memory_percent(self) -> float:
        """获取系统内存使用百分比"""
        return psutil.virtual_memory().percent

    def get_available_memory_gb(self) -> float:
        """获取可用内存（GB）"""
        return psutil.virtual_memory().available / (1024 ** 3)

    def is_memory_critical(self) -> bool:
        """检查内存是否临界"""
        return self.get_system_memory_percent() > self.max_memory_percent

    def force_gc(self):
        """强制垃圾回收"""
        gc.collect()
        logger.debug(f"执行GC，当前内存: {self.get_system_memory_percent():.1f}%")

    def wait_for_memory(self, timeout: float = 30.0) -> bool:
        """等待内存释放"""
        start = time.time()
        while self.is_memory_critical():
            self.force_gc()
            time.sleep(0.5)
            if time.time() - start > timeout:
                logger.warning("等待内存释放超时")
                return False
        return True

    def register_callback(self, callback: Callable):
        """注册内存警告回调"""
        self._callbacks.append(callback)

    def start_monitoring(self, interval: float = 1.0):
        """启动监控"""
        if self._monitoring:
            return

        self._monitoring = True

        def monitor():
            while self._monitoring:
                usage = self.get_system_memory_percent()
                if usage > self.max_memory_percent:
                    logger.warning(f"内存使用过高: {usage:.1f}%")
                    self.force_gc()
                for cb in self._callbacks:
                    try:
                        cb(usage)
                    except Exception as e:
                        logger.error(f"内存回调错误: {e}")
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        logger.info("内存监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("内存监控已停止")

    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self._memmap_manager.cleanup()
        self.force_gc()