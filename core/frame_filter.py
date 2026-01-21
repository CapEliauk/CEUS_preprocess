"""
帧筛选模块 - 内存优化版本 (迭代式处理)
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from config import config
from utils.logger import get_logger

logger = get_logger('frame_filter')


class FrameFilter:
    """帧筛选器 - 内存优化版"""

    def __init__(self):
        self.ssim_threshold = config.filter.SSIM_THRESHOLD
        self.motion_sigma = config.filter.MOTION_THRESHOLD_SIGMA

    def _get_gray_metric_data(self, frame: np.ndarray, mask_bool: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取用于计算指标的灰度数据 (内存优化)

        策略:
        1. 如果有Mask，直接提取Mask内的像素点 (1D数组)，极大节省内存。
        2. 如果无Mask，先降采样再转灰度，避免全分辨率处理。
        """
        # 策略1: ROI Mask 模式 (返回 1D float32)
        if mask_bool is not None:
            if frame.ndim == 3:
                # 只提取 ROI 区域的 RGB 像素
                pixels = frame[mask_bool] # (N_pixels, 3)
                # BGR -> Gray: 手动点积
                return np.dot(pixels.astype(np.float32), [0.114, 0.587, 0.299])
            else:
                return frame[mask_bool].astype(np.float32)

        # 策略2: 全图模式 - 降采样 (返回 2D float32)
        # 运动检测不需要 1080p 分辨率，缩小到宽度 480 足够准确且极快
        target_width = 480
        h, w = frame.shape[:2]
        if w > target_width:
            scale = target_width / w
            new_h = int(h * scale)
            # 使用 INTER_NEAREST 或 LINEAR 均可，速度优先
            small_frame = cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame

        if small_frame.ndim == 3:
            return cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return small_frame.astype(np.float32)

    def filter_by_motion(self, frames: np.ndarray,
                         roi_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于运动的快速筛选 - 迭代式

        Args:
            frames: (N, H, W, C) 帧数组
            roi_mask: (H, W) 掩膜
        """
        n_frames = len(frames)
        if n_frames < 2:
            return np.arange(n_frames), np.array([0.0] * n_frames)

        # 准备 Mask
        mask_bool = None
        if roi_mask is not None:
            # 确保 mask 非空
            if np.any(roi_mask):
                mask_bool = roi_mask > 0

        # 初始化分数数组
        motion_scores = np.zeros(n_frames, dtype=np.float32)

        # 预处理第一帧
        prev_data = self._get_gray_metric_data(frames[0], mask_bool)

        # 逐帧迭代 (避免一次性转换所有帧)
        for i in range(1, n_frames):
            curr_data = self._get_gray_metric_data(frames[i], mask_bool)

            # 计算差异 (L1 Norm)
            # 无论是 1D (Masked) 还是 2D (Resized)，mean() 都能正常工作
            score = np.mean(np.abs(curr_data - prev_data))
            motion_scores[i] = score

            prev_data = curr_data

        # --- 以下逻辑保持不变 ---

        # 自适应阈值 (忽略第一帧的0分)
        valid_motion = motion_scores[1:]
        if len(valid_motion) == 0:
             return np.arange(n_frames), motion_scores

        mean_motion = np.mean(valid_motion)
        std_motion = np.std(valid_motion)
        threshold = mean_motion + self.motion_sigma * std_motion

        # 有效帧判断
        valid_mask = motion_scores <= threshold

        # 第一帧通常保留（或者根据第二帧判断，这里默认保留）
        valid_mask[0] = True

        # 保证最少帧数
        min_frames = max(1, int(n_frames * config.filter.MIN_VALID_FRAME_RATIO))
        current_valid_count = np.sum(valid_mask)

        if current_valid_count < min_frames:
            logger.info(f"有效帧过少 ({current_valid_count} < {min_frames})，放宽阈值")
            # 排序后取前 min_frames 个最小运动的帧
            sorted_indices = np.argsort(motion_scores)
            # 确保前 min_frames 个被选中，其他的根据阈值
            top_k_indices = sorted_indices[:min_frames]
            valid_mask[top_k_indices] = True

        return np.where(valid_mask)[0], motion_scores

    def filter_by_ssim(self, frames: np.ndarray,
                       valid_indices: np.ndarray,
                       roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        基于SSIM的精细筛选 - 迭代式
        """
        if not SKIMAGE_AVAILABLE:
            logger.warning("skimage不可用，跳过SSIM筛选")
            return valid_indices

        if len(valid_indices) <= 1:
            return valid_indices

        refined_indices = [valid_indices[0]]

        # 准备 Mask (SSIM 需要保持 2D 结构，不能用 1D 像素提取)
        # 如果有 Mask，我们将其应用于灰度图，背景置 0
        mask_bool = roi_mask > 0 if roi_mask is not None else None

        # 获取第一帧灰度 (SSIM通常不需要降采样太厉害，或者保持原样)
        # 这里我们手动转灰度，保持原分辨率以获得准确 SSIM
        def get_gray_for_ssim(idx):
            frame = frames[idx]
            if frame.ndim == 3:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                g = frame

            if mask_bool is not None:
                # 应用 Mask，背景变黑
                g = g.copy() # 避免修改原图
                g[~mask_bool] = 0
            return g

        prev_gray = get_gray_for_ssim(valid_indices[0])

        for idx in valid_indices[1:]:
            curr_gray = get_gray_for_ssim(idx)

            try:
                # 计算 SSIM
                score, _ = ssim(curr_gray, prev_gray, full=True)

                if score >= self.ssim_threshold:
                    refined_indices.append(idx)
                    prev_gray = curr_gray # 更新基准帧
                else:
                    # 如果相似度低（比如突然模糊），丢弃该帧
                    # prev_gray 不更新，继续用清晰的帧与下一帧对比
                    pass
            except Exception as e:
                logger.warning(f"SSIM计算出错: {e}")
                refined_indices.append(idx)
                prev_gray = curr_gray

        return np.array(refined_indices)

    def compute_quality_scores(self, frames: np.ndarray,
                               roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算帧质量分数 - 迭代式
        """
        n = len(frames)
        scores = np.zeros(n)

        # Laplacian 核
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        mask_bool = roi_mask > 0 if roi_mask is not None else None

        # 逐帧计算清晰度
        sharpness = np.zeros(n)
        for i in range(n):
            frame = frames[i]

            # 转灰度
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = frame.astype(np.float32)

            # 如果有 Mask，只在 Mask 内计算方差
            # 但卷积需要空间信息，先卷积再 Mask
            lap = ndimage.convolve(gray, kernel)

            if mask_bool is not None:
                lap_vals = lap[mask_bool]
                sharpness[i] = np.var(lap_vals)
            else:
                sharpness[i] = np.var(lap)

        # 归一化
        max_sharp = np.max(sharpness)
        if max_sharp > 0:
            sharpness_norm = sharpness / max_sharp
        else:
            sharpness_norm = np.zeros(n)

        # 这里的“稳定性”可以用 filter_by_motion 里的逻辑算，暂时简化为 1.0
        # 或者为了完整性，这里可以再跑一次简单的 diff

        return sharpness_norm  # 目前主要返回清晰度

    def filter_frames(self, frames: List[np.ndarray],
                      roi_mask: Optional[np.ndarray] = None,
                      use_ssim: bool = None) -> Tuple[List[np.ndarray], List[int]]:
        """
        综合帧筛选 (入口)
        """
        if not frames:
            return [], []

        use_ssim = use_ssim if use_ssim is not None else not config.filter.USE_FAST_FILTER

        # 注意：这里 stack 可能会消耗一些内存 (uint8)，但在 16GB 机器上通常没问题
        # 如果 frames 列表非常巨大 (如 >1000 帧 4K)，这里也可能 OOM
        # 但通常 clip 处理是分段的。
        try:
            frames_array = np.stack(frames, axis=0)
        except Exception as e:
            logger.error(f"内存不足，无法堆叠帧数组: {e}")
            # 如果堆叠失败，可能需要降级处理或分批，这里直接返回原列表
            return frames, list(range(len(frames)))

        # 第一步：运动筛选
        valid_indices, _ = self.filter_by_motion(frames_array, roi_mask)

        logger.debug(f"运动筛选: {len(frames)} -> {len(valid_indices)}")

        # 第二步：SSIM精细筛选（可选）
        if use_ssim and len(valid_indices) > config.video.CLIP_LENGTH:
            valid_indices = self.filter_by_ssim(frames_array, valid_indices, roi_mask)
            logger.debug(f"SSIM筛选: -> {len(valid_indices)}")

        valid_frames = [frames[i] for i in valid_indices]
        return valid_frames, valid_indices.tolist()