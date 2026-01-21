"""
视频加载模块 - 使用生成器模式和生产者-消费者模式
封装OpenCV的VideoCapture，提供统一的接口给上层调用
"""
import cv2
import numpy as np
from typing import Optional, Generator, Tuple, List
from pathlib import Path
import threading
from queue import Queue, Empty
from dataclasses import dataclass

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from config import config
from utils.memory_manager import MemoryManager, FrameBuffer
from utils.file_utils import is_dicom_file
from utils.logger import get_logger

logger = get_logger('video_loader')


@dataclass
class FrameData:
    """帧数据"""
    index: int
    frame: np.ndarray
    timestamp: float


class VideoLoader:
    """视频加载器基类"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.frame_count = 0
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self._lock = threading.Lock()
        self._buffer = FrameBuffer()
        self._memory = MemoryManager()

    def get_frame(self, frame_idx: int) -> np.ndarray | None:
        """获取指定帧"""
        raise NotImplementedError

    def iter_frames(self, start: int = 0, end: int = None,
                    step: int = 1) -> Generator[FrameData, None, None]:
        """帧迭代器（生成器模式）"""
        raise NotImplementedError

    def iter_frames_batch(self, batch_size: int = None) -> Generator[List[FrameData], None, None]:
        """批量帧迭代器"""
        batch_size = batch_size or config.memory.BATCH_SIZE
        batch = []

        for frame_data in self.iter_frames():
            batch.append(frame_data)
            if len(batch) >= batch_size:
                yield batch
                batch = []

                # 检查内存
                if self._memory.is_memory_critical():
                    self._memory.wait_for_memory()

        if batch:
            yield batch

    def release(self):
        """释放资源"""
        self._buffer.clear()


class StandardVideoLoader(VideoLoader):
    """标准视频加载器"""

    def __init__(self, filepath: str):
        super().__init__(filepath)

        self._cap = cv2.VideoCapture(filepath)
        if not self._cap.isOpened():
            raise ValueError(f"无法打开视频: {filepath}")

        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"加载视频: {filepath}, {self.frame_count}帧, {self.fps:.1f}fps, {self.width}x{self.height}")

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None

        # 检查缓存
        cached = self._buffer.get(frame_idx)
        if cached is not None:
            return cached

        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret:
                self._buffer.add(frame_idx, frame)
                return frame.copy()
        return None

    def iter_frames(self, start: int = 0, end: int = None,
                    step: int = 1) -> Generator[FrameData, None, None]:
        """帧迭代器"""
        end = end or self.frame_count

        with self._lock:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            current = start
            while current < end:
                if self._memory.is_memory_critical():
                    self._memory.force_gc()

                ret, frame = self._cap.read()
                if not ret:
                    break

                if (current - start) % step == 0:
                    yield FrameData(
                        index=current,
                        frame=frame,
                        timestamp=current / self.fps
                    )

                current += 1

    def release(self):
        """释放资源"""
        super().release()
        with self._lock:
            if self._cap:
                self._cap.release()
        logger.debug(f"视频资源已释放: {self.filepath}")


class DicomVideoLoader(VideoLoader):
    """DICOM视频加载器"""

    def __init__(self, filepath: str):
        if not PYDICOM_AVAILABLE:
            raise ImportError("需要安装pydicom: pip install pydicom")

        super().__init__(filepath)

        self._ds = pydicom.dcmread(filepath)
        self._pixel_array = self._ds.pixel_array

        # 解析帧
        self._frames = self._parse_frames()
        self.frame_count = len(self._frames)

        if self.frame_count == 0:
            raise ValueError(f"DICOM无有效帧: {filepath}")

        self.height, self.width = self._frames[0].shape[:2]

        # 获取帧率
        if hasattr(self._ds, 'CineRate'):
            self.fps = float(self._ds.CineRate)
        elif hasattr(self._ds, 'FrameTime'):
            self.fps = 1000.0 / float(self._ds.FrameTime)
        else:
            self.fps = 30.0

        logger.info(f"加载DICOM: {filepath}, {self.frame_count}帧, {self.fps:.1f}fps")

    def _parse_frames(self) -> List[np.ndarray]:
        """解析DICOM帧"""
        frames = []
        pixel_array = self._pixel_array

        if pixel_array.ndim == 4:  # (N, H, W, C)
            for i in range(pixel_array.shape[0]):
                frame = self._normalize(pixel_array[i])
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frames.append(frame)
        elif pixel_array.ndim == 3:
            if pixel_array.shape[2] == 3:  # 单帧彩色
                frames.append(self._normalize(pixel_array))
            else:  # 多帧灰度
                for i in range(pixel_array.shape[0]):
                    frame = cv2.cvtColor(self._normalize(pixel_array[i]), cv2.COLOR_GRAY2BGR)
                    frames.append(frame)
        elif pixel_array.ndim == 2:  # 单帧灰度
            frame = cv2.cvtColor(self._normalize(pixel_array), cv2.COLOR_GRAY2BGR)
            frames.append(frame)

        return frames

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化到uint8"""
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return arr.astype(np.uint8)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if 0 <= frame_idx < self.frame_count:
            return self._frames[frame_idx].copy()
        return None

    def iter_frames(self, start: int = 0, end: int = None,
                    step: int = 1) -> Generator[FrameData, None, None]:
        end = end or self.frame_count

        for i in range(start, end, step):
            if i < self.frame_count:
                yield FrameData(
                    index=i,
                    frame=self._frames[i].copy(),
                    timestamp=i / self.fps
                )

    def release(self):
        super().release()
        self._frames.clear()
        self._pixel_array = None


class FrameProducer(threading.Thread):
    """帧生产者线程"""

    def __init__(self, loader: VideoLoader, queue: Queue,
                 start: int = 0, end: int = None, step: int = 1):
        super().__init__(daemon=True)
        self.loader = loader
        self.queue = queue
        self.start_idx = start
        self.end_idx = end or loader.frame_count
        self.step = step
        self._stop_event = threading.Event()

    def run(self):
        logger.debug(f"帧生产者启动: {self.start_idx} -> {self.end_idx}")

        for frame_data in self.loader.iter_frames(self.start_idx, self.end_idx, self.step):
            if self._stop_event.is_set():
                break

            self.queue.put(frame_data)

        self.queue.put(None)  # 结束信号
        logger.debug("帧生产者结束")

    def stop(self):
        self._stop_event.set()


def create_video_loader(filepath: str) -> VideoLoader:
    """工厂函数"""
    if is_dicom_file(filepath) or Path(filepath).suffix.lower() in ('.dcm', '.dicom'):
        return DicomVideoLoader(filepath)
    return StandardVideoLoader(filepath)


def create_frame_producer(loader: VideoLoader, queue_size: int = 30,
                          **kwargs) -> Tuple[Queue, FrameProducer]:
    """创建帧生产者"""
    queue = Queue(maxsize=queue_size)
    producer = FrameProducer(loader, queue, **kwargs)
    return queue, producer