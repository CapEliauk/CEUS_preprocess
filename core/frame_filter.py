"""
帧筛选模块 - 内存优化版 (支持流式计算)
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim
from utils.logger import get_logger

logger = get_logger('frame_filter')


class FrameFilter:
    """呼吸门控筛选器"""

    def __init__(self):
        self.peak_distance = 5
        self.target_frames = 16

    def _get_roi_gray(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """辅助：提取ROI并转灰度"""
        # 1. 转灰度
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 2. 如果没有Mask，直接返回
        if mask is None:
            return gray

        # 3. 利用Mask裁剪最小区域 (加速SSIM)
        y, x = np.where(mask > 0)
        if len(y) == 0:
            return gray

        y1, y2, x1, x2 = y.min(), y.max(), x.min(), x.max()
        crop = gray[y1:y2 + 1, x1:x2 + 1]

        return crop

    def prepare_anchor(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """准备基准帧数据 (供 Worker 调用)"""
        return self._get_roi_gray(frame, mask)

    def compute_single_ssim(self, frame: np.ndarray, anchor_gray: np.ndarray, mask: np.ndarray) -> float:
        """计算单帧 SSIM (流式处理用)"""
        curr_gray = self._get_roi_gray(frame, mask)

        # 尺寸保护 (防止裁剪后尺寸不一致报错)
        if curr_gray.shape != anchor_gray.shape:
            # 如果尺寸不一致，尝试 resize
            curr_gray = cv2.resize(curr_gray, (anchor_gray.shape[1], anchor_gray.shape[0]))

        try:
            return ssim(curr_gray, anchor_gray, data_range=255)
        except Exception:
            return 0.0

    def find_peaks_from_scores(self, scores: List[float]) -> List[int]:
        """从分数列表中寻找波峰索引"""
        if not scores: return []

        peaks, _ = find_peaks(scores, distance=self.peak_distance, height=0.3)

        # 兜底
        if len(peaks) < 5:
            return np.linspace(0, len(scores) - 1, len(scores), dtype=int).tolist()

        return peaks.tolist()

    def sample_for_videomae(self, indices: List[int]) -> List[int]:
        """VideoMAE 均匀采样"""
        if len(indices) <= self.target_frames: return indices
        selected = np.linspace(0, len(indices) - 1, self.target_frames, dtype=int)
        return [indices[i] for i in selected]