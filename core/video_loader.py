"""
视频加载模块 - 使用生成器模式和生产者-消费者模式
封装OpenCV的VideoCapture，提供统一的接口给上层调用
"""
import cv2
import numpy as np
from typing import Optional, Generator, Tuple, List
from pathlib import Path
import threading
import gc
import uuid
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
        self.fps = config.video.DEFAULT_FPS
        self.width = 0
        self.height = 0
        self._lock = threading.Lock()
        self._buffer = FrameBuffer()
        self._memory = MemoryManager()

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        raise NotImplementedError

    def iter_frames(self, start: int = 0, end: int = None,
                    step: int = 1) -> Generator[FrameData, None, None]:
        """帧迭代器（生成器模式）"""
        raise NotImplementedError

    def iter_frames_batch(self, batch_size: int = None) -> Generator[List[FrameData], None, None]:
        """批量帧迭代器"""
        batch_size = batch_size or config.memory.BATCH_SIZE
        batch: list[FrameData] = []

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
        raise NotImplementedError


class StandardVideoLoader(VideoLoader):
    """标准视频加载器"""

    def __init__(self, filepath: str):
        super().__init__(filepath)

        self._cap = cv2.VideoCapture(filepath)
        if not self._cap.isOpened():
            raise ValueError(f"无法打开视频: {filepath}")

        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or config.video.DEFAULT_FPS
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
        with self._lock:
            self._buffer.clear()
            if self._cap:
                self._cap.release()
        logger.debug(f"视频资源已释放: {self.filepath}")


class DicomVideoLoader(VideoLoader):
    """DICOM视频加载器"""

    def __init__(self, filepath: str):
        if not PYDICOM_AVAILABLE:
            raise ImportError("需要安装pydicom: pip install pydicom")

        super().__init__(filepath)

        # 读取 Header (不立刻读取 pixel_data)
        # stop_before_pixels=True 可以避免立刻加载大文件
        try:
            self._ds = pydicom.dcmread(filepath, stop_before_pixels=False)
        except Exception as e:
            logger.error(f"DICOM读取失败: {e}")
            raise

        # 获取基本维度信息
        self.height = int(self._ds.Rows)
        self.width = int(self._ds.Columns)

        # 尝试获取帧数
        if hasattr(self._ds, "NumberOfFrames"):
            self.frame_count = int(self._ds.NumberOfFrames)
        else:
            self.frame_count = 1

        # 创建 Memmap (磁盘映射内存)
        # 生成唯一ID，防止多文件冲突
        self._memmap_key = f"dicom_{uuid.uuid4().hex}"

        # 存储为 BGR 格式 (H, W, 3)，统一 uint8
        shape = (self.frame_count, self.height, self.width, 3)

        # 调用 MemoryManager 创建磁盘文件
        self._frames = self._memory.memmap.create_memmap(
            name=self._memmap_key,
            shape=shape,
            dtype=np.uint8
        )

        # 逐帧处理并写入磁盘 (核心防OOM逻辑)
        logger.info(f"开始处理DICOM帧到磁盘缓存: {shape}")
        self._load_to_memmap()

        # 解析 FPS
        self._parse_fps()

        logger.info(f"加载DICOM完成(Memmap): {filepath}, {self.frame_count}帧, {self.fps:.1f}fps, {self.width}x{self.height}")

    def _parse_fps(self):
        """解析FPS"""
        if hasattr(self._ds, 'FrameRate') and self._ds.FrameRate is not None and self._ds.FrameRate > 0:
            self.fps = float(self._ds.FrameRate)
        elif hasattr(self._ds, 'FrameTime') and self._ds.FrameTime:
            self.fps = 1000.0 / float(self._ds.FrameTime)
        elif hasattr(self._ds, 'FrameTimeVector') and len(self._ds.FrameTimeVector) > 0:
            frame_intervals = [float(t) for t in self._ds.FrameTimeVector if t > 0]
            if frame_intervals:
                self.fps = 1000.0 / float(np.mean(frame_intervals))
        else:
            self.fps = config.video.DEFAULT_FPS_DICOM

    def _get_pixel_generator(self) -> Generator[np.ndarray, None, None]:
        """
        创建一个生成器，逐帧产出像素数据，严格避免全量解压。
        """
        # 获取图像的基本信息
        samples_per_pixel = getattr(self._ds, 'SamplesPerPixel', 1)
        bits_allocated = getattr(self._ds, 'BitsAllocated', 16)

        # 判断是否为压缩格式 (TransferSyntaxUID)
        # 显式检查 file_meta，如果缺失则尝试从 dataset 获取或默认为非压缩
        is_compressed = False
        try:
            is_compressed = self._ds.file_meta.TransferSyntaxUID.is_compressed
        except (AttributeError, ValueError):
            # 如果无法判断，通常默认为未压缩，除非 PixelData 是封装格式
            pass

        # === 分支 1: 处理压缩数据 (JPEG, RLE 等) ===
        if is_compressed:
            import pydicom.encaps

            # 获取帧数
            num_frames = getattr(self._ds, "NumberOfFrames", 1)

            # 使用 pydicom 的生成器获取每一帧的压缩字节流
            # 这不会解压数据，内存占用极低
            if 'PixelData' in self._ds:
                frame_gen = pydicom.encaps.generate_pixel_data_frame(self._ds.PixelData, num_frames)
            else:
                logger.error("DICOM 文件缺少 PixelData")
                return

            for frame_bytes in frame_gen:
                # 使用 OpenCV 直接解码内存中的压缩字节流
                # 这比 pydicom 的解压机制更轻量，且天然支持单帧解码
                arr = np.frombuffer(frame_bytes, np.uint8)

                # cv2.IMREAD_UNCHANGED 会根据数据深度自动处理 (如 16位图像)
                # cv2 解码默认为 BGR 顺序，这正好符合 VideoLoader 的需求
                frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

                if frame is not None:
                    yield frame
                else:
                    # 如果 cv2 解码失败 (例如特殊的 RLE 格式)，这里可以记录警告
                    # 为了防止崩溃，返回一个黑帧
                    logger.warning("单帧解码失败，跳过该帧")
                    yield np.zeros((self.height, self.width, 3 if samples_per_pixel == 3 else 1), dtype=np.uint8)

        # === 分支 2: 处理未压缩数据 (Raw Data) ===
        else:
            # 确定数据类型
            if bits_allocated == 8:
                dtype = np.uint8
            elif bits_allocated == 16:
                dtype = np.uint16
            else:
                logger.error(f"不支持的位深: {bits_allocated}")
                return

            # 使用 frombuffer 创建内存视图，这不会复制数据，只是创建了一个指向它的指针
            try:
                # 注意：pydicom 读取时如果是 Implicit VR，PixelData 可能是 bytes
                pixel_data = self._ds.PixelData
                full_arr = np.frombuffer(pixel_data, dtype=dtype)
            except Exception as e:
                logger.error(f"无法创建内存视图: {e}")
                return

            # 计算单帧的大小 (元素个数)
            frame_size = self.height * self.width * samples_per_pixel
            num_frames = getattr(self._ds, "NumberOfFrames", 1)

            # 确保数据长度足够
            if full_arr.size < frame_size * num_frames:
                logger.warning("PixelData 数据长度不足，可能文件损坏")
                num_frames = full_arr.size // frame_size

            # 逐帧切片 yield
            for i in range(num_frames):
                start = i * frame_size
                end = (i + 1) * frame_size

                # 切片操作也是视图，不占用额外内存
                frame_flat = full_arr[start:end]

                # Reshape 为图像矩阵
                if samples_per_pixel == 3:
                    # DICOM Raw 通常是 RGB，后续可能需要转 BGR
                    frame = frame_flat.reshape((self.height, self.width, 3))
                else:
                    frame = frame_flat.reshape((self.height, self.width))

                yield frame

    def _load_to_memmap(self):
        """将数据处理后写入 Memmap，并及时释放内存"""

        # 获取生成器
        pixel_gen = self._get_pixel_generator()

        for idx, raw_frame in enumerate(pixel_gen):
            if idx >= self.frame_count:
                break

            # 归一化
            normalized = self._normalize(raw_frame)

            # 颜色空间转换 (转为 BGR)
            if normalized.ndim == 2:
                # 灰度转彩色
                frame = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            elif normalized.ndim == 3 and normalized.shape[2] == 3:
                # RGB 转 BGR (上面cv2.imdecode已经转成BGR格式)
                frame = normalized
            elif normalized.ndim == 3 and normalized.shape[2] == 4:
                frame = cv2.cvtColor(normalized, cv2.COLOR_RGBA2BGR)
            else:
                frame = normalized

            # 写入 Memmap (写入硬盘)
            self._frames[idx] = frame

            # 强制内存清理
            del raw_frame
            del normalized
            del frame

            # 每处理一定数量帧，检查内存
            if idx % 10 == 0 and self._memory.is_memory_critical():
                self._memory.force_gc()

        # 写入完成后，刷新到磁盘
        self._frames.flush()

        # 删除 ds 中的 pixel_array 缓存（如果有）
        if hasattr(self._ds, "_pixel_array"):
            del self._ds._pixel_array
        self._memory.force_gc()

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化到uint8"""
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return arr.astype(np.uint8)

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """从 Memmap 读取帧"""
        if 0 <= frame_idx < self.frame_count:
            # 返回副本， 防止外部修改影响 Mammap 文件
            return self._frames[frame_idx].copy()
        return None

    def iter_frames(self, start: int = 0, end: int = None, step: int = 1) -> Generator[FrameData, None, None]:
        end = end or self.frame_count
        with self._lock:
            for i in range(start, end, step):
                if self._memory.is_memory_critical():
                    self._memory.force_gc()

                yield FrameData(
                    index=i,
                    # 直接切片 Memmap，速度极快，不占用额外 RAM
                    frame=self._frames[i].copy(),
                    timestamp=i / self.fps
                )

    def release(self):
        """释放资源"""
        self._memory.memmap.cleanup()
        self._ds = None
        gc.collect()


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
    if is_dicom_file(filepath) or Path(filepath).suffix.lower() in config.video.DICOM_EXTENSIONS:
        return DicomVideoLoader(filepath)
    return StandardVideoLoader(filepath)


def create_frame_producer(loader: VideoLoader, queue_size: int = 30) -> Tuple[Queue, FrameProducer]:
    """创建帧生产者"""
    queue = Queue(maxsize=queue_size)
    producer = FrameProducer(loader, queue)
    return queue, producer