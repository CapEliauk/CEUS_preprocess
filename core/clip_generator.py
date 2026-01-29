"""
Clip生成模块 (重构版) - 专用于 VideoMAE 样本生成
"""
import os
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from config import config
from utils.logger import get_logger

logger = get_logger('clip_generator')

@dataclass
class ROIBoundingBox:
    """ROI边界框 (保留原有逻辑)"""
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 10) -> 'ROIBoundingBox':
        points = np.where(mask > 0)
        if len(points[0]) == 0:
            return cls(0, 0, mask.shape[1], mask.shape[0])
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask.shape[1], x_max + padding)
        y_max = min(mask.shape[0], y_max + padding)
        return cls(x_min, y_min, x_max - x_min, y_max - y_min)

    def make_square(self, img_shape: Tuple[int, int]) -> 'ROIBoundingBox':
        size = max(self.width, self.height)
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        new_x = int(max(0, cx - size // 2))
        new_y = int(max(0, cy - size // 2))

        # 边界检查
        if new_x + size > img_shape[1]: new_x = max(0, img_shape[1] - size)
        if new_y + size > img_shape[0]: new_y = max(0, img_shape[0] - size)

        actual_size = int(min(size, img_shape[1] - new_x, img_shape[0] - new_y))
        return ROIBoundingBox(new_x, new_y, actual_size, actual_size)

class ClipGenerator:
    """VideoMAE 样本生成器"""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.target_size = (224, 224)

        # 创建子文件夹
        self.train_dir = os.path.join(self.output_dir, "train_npz")
        self.vis_dir = os.path.join(self.output_dir, "vis_mp4")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def _process_frame(self, frame: np.ndarray, bbox: ROIBoundingBox) -> np.ndarray:
        """辅助：裁剪并缩放单帧"""
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0: return cv2.resize(frame, self.target_size)
        return cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)

    def save_training_sample(self,
                             frames: List[np.ndarray],
                             timestamps: List[float],
                             roi_mask: np.ndarray,
                             case_id: str,
                             phase_name: str) -> bool:
        """
        核心方法：保存处理后的样本
        """
        if len(frames) == 0: return False

        # 1. 计算所有帧通用的 Bbox (正方形)
        full_bbox = ROIBoundingBox.from_mask(roi_mask, padding=15)
        square_bbox = full_bbox.make_square(frames[0].shape[:2])

        # 2. 批量裁剪和缩放
        processed_frames = [self._process_frame(f, square_bbox) for f in frames]

        # 3. 堆叠成 Tensor (T, H, W, C)
        try:
            video_tensor = np.stack(processed_frames, axis=0)
        except ValueError:
            return False

        # 4. 保存为 .npz (训练用)
        save_name = f"{case_id}_{phase_name}"
        npz_path = os.path.join(self.train_dir, f"{save_name}.npz")
        np.savez_compressed(npz_path,
                            video=video_tensor,
                            timestamps=np.array(timestamps))

        # 5. 保存为 .mp4 (可视化检查用)
        self._save_video(processed_frames, os.path.join(self.vis_dir, f"{save_name}.mp4"), fps=4.0)

        logger.info(f"样本已保存: {save_name} (Frames: {len(frames)})")
        return True

    def _save_video(self, frames, path, fps):
        if not frames: return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for f in frames:
            img = f.astype(np.uint8)
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.write(img)
        out.release()