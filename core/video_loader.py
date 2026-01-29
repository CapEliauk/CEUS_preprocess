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
            if hasattr(self, "_buffer"):
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
            self._ds = pydicom.dcmread(filepath, stop_before_pixels=True)
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
        创建一个生成器，逐帧产出像素数据。
        如果因为内存优化导致 PixelData 未加载，这里会临时加载。
        """
        import pydicom.encaps
        # 1. 确定数据源
        # 如果 __init__ 中使用了 stop_before_pixels=True，这里 self._ds 可能没有 PixelData
        # 我们需要一个临时的 dataset 来读取像素
        temp_ds = None
        if not hasattr(self._ds, 'PixelData'):
            try:
                # 临时读取完整文件 (会产生瞬间内存峰值，但随着生成器结束会释放)
                temp_ds = pydicom.dcmread(self.filepath)
                ds_source = temp_ds
            except Exception as e:
                logger.error(f"临时读取DICOM像素失败: {e}")
                return
        else:
            ds_source = self._ds

        # 2. 获取图像的基本信息 (从数据源获取)
        samples_per_pixel = getattr(ds_source, 'SamplesPerPixel', 1)
        bits_allocated = getattr(ds_source, 'BitsAllocated', 16)
        num_frames = getattr(ds_source, "NumberOfFrames", 1)

        # 3. 判断压缩格式
        is_compressed = False
        try:
            # 优先看 file_meta
            if hasattr(ds_source, 'file_meta'):
                is_compressed = ds_source.file_meta.TransferSyntaxUID.is_compressed
            # 如果没有，尝试推断 (只要 PixelData 是封装序列通常就是压缩的)
        except (AttributeError, ValueError):
            pass

        try:
            # === 分支 1: 处理压缩数据 (JPEG, RLE 等) ===
            if is_compressed:
                if hasattr(ds_source, 'PixelData'):
                    frame_gen = pydicom.encaps.generate_pixel_data_frame(ds_source.PixelData, num_frames)

                    for frame_bytes in frame_gen:
                        arr = np.frombuffer(frame_bytes, np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

                        if frame is not None:
                            yield frame
                        else:
                            logger.warning("单帧解码失败，返回黑帧")
                            yield np.zeros((self.height, self.width, 3 if samples_per_pixel == 3 else 1),
                                           dtype=np.uint8)
                else:
                    logger.error("无法获取 PixelData (压缩格式)")

            # === 分支 2: 处理未压缩数据 (Raw Data) ===
            else:
                if bits_allocated == 8:
                    dtype = np.uint8
                elif bits_allocated == 16:
                    dtype = np.uint16
                else:
                    logger.error(f"不支持的位深: {bits_allocated}")
                    return

                pixel_data = ds_source.PixelData
                full_arr = np.frombuffer(pixel_data, dtype=dtype)

                frame_size = self.height * self.width * samples_per_pixel

                # 长度校验
                if full_arr.size < frame_size * num_frames:
                    num_frames = full_arr.size // frame_size

                for i in range(num_frames):
                    start = i * frame_size
                    end = (i + 1) * frame_size
                    frame_flat = full_arr[start:end]

                    if samples_per_pixel == 3:
                        frame = frame_flat.reshape((self.height, self.width, 3))
                    else:
                        frame = frame_flat.reshape((self.height, self.width))

                    yield frame

        finally:
            # 生成器结束或中断时，确保释放临时对象
            if temp_ds is not None:
                del temp_ds
            # 强制回收一次，确保大数组内存被标记为可回收
            # (注意：频繁调用gc有性能开销，但在这种IO密集型操作结束时调用是划算的)
            pass

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
        if hasattr(self._ds, 'PixelData'):
            del self._ds.PixelData
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
        if hasattr(self, '_memmap_key'):
            self._memory.memmap.delete_memmap(self._memmap_key)
        else:
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