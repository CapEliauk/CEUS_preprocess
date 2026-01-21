"""
Clip生成模块 - 基于ROI最小外接矩形裁剪到224x224
"""
import numpy as np
import cv2
from typing import List, Tuple, Generator, Optional
from dataclasses import dataclass

from config import config
from core.tic_analyzer import TICAnalyzer
from utils.memory_manager import MemoryManager
from utils.logger import get_logger

logger = get_logger('clip_generator')


@dataclass
class Clip:
    """Clip数据"""
    frames: List[np.ndarray]
    phase: str
    start_frame: int
    end_frame: int


@dataclass
class ROIBoundingBox:
    """ROI边界框"""
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 10) -> 'ROIBoundingBox':
        """从mask计算最小外接矩形"""
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return cls(0, 0, mask.shape[1], mask.shape[0])

        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()

        # 添加padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask.shape[1], x_max + padding)
        y_max = min(mask.shape[0], y_max + padding)

        return cls(x_min, y_min, x_max - x_min, y_max - y_min)

    @classmethod
    def from_contours(cls, mask: np.ndarray, padding: int = 10) -> 'ROIBoundingBox':
        """从mask轮廓计算最小外接矩形"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return cls(0, 0, mask.shape[1], mask.shape[0])

        # 合并所有轮廓点
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # 添加padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(mask.shape[1] - x, w + 2 * padding)
        h = min(mask.shape[0] - y, h + 2 * padding)

        return cls(x, y, w, h)

    def make_square(self, img_shape: Tuple[int, int]) -> 'ROIBoundingBox':
        """将边界框调整为正方形（保持中心点）"""
        size = max(self.width, self.height)

        # 中心点
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2

        # 新的左上角
        new_x = cx - size // 2
        new_y = cy - size // 2

        # 确保不超出图像边界
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        if new_x + size > img_shape[1]:
            new_x = max(0, img_shape[1] - size)
        if new_y + size > img_shape[0]:
            new_y = max(0, img_shape[0] - size)

        # 如果图像太小，使用实际可用大小
        actual_size = min(size, img_shape[1] - new_x, img_shape[0] - new_y)

        return ROIBoundingBox(int(new_x), int(new_y), int(actual_size), int(actual_size))

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return f"ROIBoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class ClipGenerator:
    """Clip生成器 - 基于ROI最小外接矩形裁剪到224x224"""

    def __init__(self):
        self.clip_length = config.video.CLIP_LENGTH
        self.stride = config.video.SLIDING_WINDOW_STRIDE
        self.target_size = config.video.CLIP_SIZE  # (224, 224)
        self._memory = MemoryManager()

        logger.debug(f"ClipGenerator初始化: clip_length={self.clip_length}, "
                    f"stride={self.stride}, target_size={self.target_size}")

    def compute_roi_bbox(self, mask: np.ndarray, padding: int = 10) -> ROIBoundingBox:
        """
        从mask计算ROI的最小外接矩形

        Args:
            mask: ROI mask
            padding: 边缘填充像素

        Returns:
            ROIBoundingBox
        """
        bbox = ROIBoundingBox.from_contours(mask, padding)
        logger.debug(f"计算ROI bbox: {bbox}")
        return bbox

    def crop_and_resize(self, frame: np.ndarray, bbox: ROIBoundingBox) -> np.ndarray:
        """
        根据bbox裁剪并缩放到目标尺寸(224x224)

        Args:
            frame: 输入帧
            bbox: 裁剪边界框

        Returns:
            裁剪并缩放后的帧
        """
        h, w = frame.shape[:2]

        # 确保bbox在有效范围内
        x1 = max(0, min(bbox.x, w - 1))
        y1 = max(0, min(bbox.y, h - 1))
        x2 = max(x1 + 1, min(bbox.x + bbox.width, w))
        y2 = max(y1 + 1, min(bbox.y + bbox.height, h))

        # 裁剪
        cropped = frame[y1:y2, x1:x2]

        # 如果裁剪区域太小，添加padding
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            logger.warning(f"裁剪区域太小: {cropped.shape}, 使用整帧")
            cropped = frame

        # 缩放到目标尺寸
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def crop_and_resize_square(self, frame: np.ndarray, bbox: ROIBoundingBox) -> np.ndarray:
        """
        将ROI区域裁剪为正方形，然后缩放到224x224

        Args:
            frame: 输入帧
            bbox: 原始bbox

        Returns:
            224x224的帧
        """
        h, w = frame.shape[:2]

        # 转换为正方形bbox
        square_bbox = bbox.make_square((h, w))

        return self.crop_and_resize(frame, square_bbox)

    def generate_clips(self, frames: List[np.ndarray],
                       indices: List[int],
                       roi_mask: np.ndarray,
                       analyzer: Optional[TICAnalyzer] = None,
                       padding: int = 10,
                       make_square: bool = True) -> Generator[Clip, None, None]:
        """
        生成clips - 基于ROI最小外接矩形裁剪

        Args:
            frames: 帧列表（已经是CEUS区域）
            indices: 帧索引列表
            roi_mask: CEUS区域内的ROI mask
            analyzer: TIC分析器（用于确定相位）
            padding: bbox padding
            make_square: 是否将bbox转为正方形

        Yields:
            Clip对象
        """
        if len(frames) < self.clip_length:
            logger.warning(f"帧数不足: {len(frames)} < {self.clip_length}")
            return

        # 计算ROI的最小外接矩形
        bbox = self.compute_roi_bbox(roi_mask, padding)

        if make_square and len(frames) > 0:
            frame_shape = frames[0].shape[:2]
            bbox = bbox.make_square(frame_shape)

        logger.info(f"使用bbox裁剪: {bbox}, 目标尺寸: {self.target_size}")

        # 滑动窗口生成clips
        num_clips = 0
        for start in range(0, len(frames) - self.clip_length + 1, self.stride):
            if self._memory.is_memory_critical():
                self._memory.wait_for_memory()

            # 裁剪并缩放每一帧
            clip_frames = []
            for i in range(start, start + self.clip_length):
                if make_square:
                    cropped = self.crop_and_resize_square(frames[i],
                                                          self.compute_roi_bbox(roi_mask, padding))
                else:
                    cropped = self.crop_and_resize(frames[i], bbox)
                clip_frames.append(cropped)

            # 确定相位
            frame_idx = indices[start] if start < len(indices) else start
            if analyzer is not None:
                phase = analyzer.get_frame_phase(frame_idx)
            else:
                phase = "Unknown"

            num_clips += 1
            yield Clip(
                frames=clip_frames,
                phase=phase,
                start_frame=frame_idx,
                end_frame=frame_idx + self.clip_length
            )

        logger.info(f"生成 {num_clips} 个clips")

    def generate_clips_with_bbox(self, frames: List[np.ndarray],
                                  indices: List[int],
                                  bbox: ROIBoundingBox,
                                  analyzer: Optional[TICAnalyzer] = None) -> Generator[Clip, None, None]:
        """
        使用预计算的bbox生成clips

        Args:
            frames: 帧列表
            indices: 帧索引列表
            bbox: 预计算的边界框
            analyzer: TIC分析器

        Yields:
            Clip对象
        """
        if len(frames) < self.clip_length:
            return

        # 转换为正方形
        if len(frames) > 0:
            frame_shape = frames[0].shape[:2]
            square_bbox = bbox.make_square(frame_shape)
        else:
            square_bbox = bbox

        logger.info(f"使用预计算bbox: {square_bbox}")

        for start in range(0, len(frames) - self.clip_length + 1, self.stride):
            if self._memory.is_memory_critical():
                self._memory.wait_for_memory()

            clip_frames = [self.crop_and_resize(frames[i], square_bbox)
                          for i in range(start, start + self.clip_length)]

            frame_idx = indices[start] if start < len(indices) else start
            phase = analyzer.get_frame_phase(frame_idx) if analyzer else "Unknown"

            yield Clip(
                frames=clip_frames,
                phase=phase,
                start_frame=frame_idx,
                end_frame=frame_idx + self.clip_length
            )

    def save_clip(self, clip: Clip, output_path: str, fps: float = None):
        """保存clip为视频文件"""
        fps = fps or config.video.OUTPUT_FPS

        # 确保输出目录存在
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.target_size)

        if not out.isOpened():
            logger.error(f"无法创建视频文件: {output_path}")
            return False

        try:
            for frame in clip.frames:
                # 确保帧尺寸正确
                if frame.shape[:2] != self.target_size[::-1]:
                    frame = cv2.resize(frame, self.target_size)
                out.write(frame)
            logger.debug(f"保存clip: {output_path}, {len(clip.frames)}帧")
            return True
        except Exception as e:
            logger.error(f"保存clip失败: {e}")
            return False
        finally:
            out.release()