"""
处理流水线 - 分离计算逻辑
"""
import numpy as np
from typing import Optional, List, Dict, Generator, Callable
from dataclasses import dataclass
from queue import Queue
import threading

from PyQt5.QtCore import QObject, pyqtSignal

from config import config
from core.video_loader import VideoLoader, FrameData, create_frame_producer
from core.tic_analyzer import TICAnalyzer, TICFitResult
from core.frame_filter import FrameFilter
from core.clip_generator import ClipGenerator, ROIBoundingBox
from core.roi_manager import DualRegionROIManager
from utils.memory_manager import MemoryManager
from utils.logger import get_logger

logger = get_logger('pipeline')


@dataclass
class ProcessingResult:
    """处理结果"""
    video_path: str
    clips_count: int
    phase_clips: Dict[str, int]
    tic_result: Optional[TICFitResult]
    success: bool
    error_message: str = ""


class ProcessingSignals(QObject):
    """处理信号"""
    progress = pyqtSignal(str, int, str)  # video_path, percent, message
    tic_computed = pyqtSignal(object)  # TICFitResult
    phase_detected = pyqtSignal(dict)  # phase_intervals
    clip_generated = pyqtSignal(str, str, int)  # video_path, phase, clip_idx
    completed = pyqtSignal(object)  # ProcessingResult
    error = pyqtSignal(str, str)  # video_path, error


class VideoPipeline:
    """视频处理流水线"""

    def __init__(self, signals: Optional[ProcessingSignals] = None):
        self.signals = signals or ProcessingSignals()

        self.analyzer = TICAnalyzer()
        self.filter = FrameFilter()
        self.generator = ClipGenerator()
        self.roi_manager = DualRegionROIManager()
        self.memory = MemoryManager()

        self._stop_flag = threading.Event()

        self.manual_anchor_frame = None

    def set_anchor_frame(self, frame: np.ndarray):
        """接受GUI传来的B-mode基准帧"""
        self.manual_anchor_frame = frame

    def stop(self):
        """停止处理"""
        self._stop_flag.set()

    def reset(self):
        """重置"""
        self._stop_flag.clear()

    def set_roi(self, mask: np.ndarray, bbox: tuple):
        """设置ROI"""
        self.roi_manager.set_mask(mask)

    def process_video(self, loader: VideoLoader, output_dir: str,
                      enable_motion_compensation: bool = True) -> ProcessingResult:
        """
        处理单个视频
        """
        video_path = loader.filepath
        logger.info(f"开始处理: {video_path}")

        try:
            # 阶段1: 采样帧计算TIC
            self._emit_progress(video_path, 5, "采样帧...")
            sample_frames = self._sample_frames(loader)

            if self._stop_flag.is_set():
                return self._make_result(video_path, False, "已取消")

            # 阶段2: 计算TIC
            self._emit_progress(video_path, 15, "计算TIC曲线...")
            self.analyzer.fps = loader.fps
            frames_array = np.stack(sample_frames, axis=0)
            self.analyzer.tic_data = self.analyzer.extract_roi_intensities_batch(
                frames_array, self.roi_manager.mask
            )
            self.analyzer.time_axis = np.arange(len(sample_frames)) / loader.fps

            # 阶段3: 拟合TIC
            self._emit_progress(video_path, 25, "拟合TIC模型...")
            fit_result = self.analyzer.fit_gamma_model()
            self.signals.tic_computed.emit(fit_result)

            # 阶段4: 获取相位
            phase_intervals = self.analyzer.get_phase_intervals()
            self.signals.phase_detected.emit(phase_intervals)

            if self._stop_flag.is_set():
                return self._make_result(video_path, False, "已取消")

            # 阶段5: 启用运动补偿
            if enable_motion_compensation and config.registration.ENABLE_REGISTRATION:
                self._emit_progress(video_path, 30, "初始化运动补偿...")
                ref_frame = loader.get_frame(0)
                if ref_frame is not None:
                    self.roi_manager.enable_compensation(ref_frame)

            # 阶段6: 按相位处理
            self._emit_progress(video_path, 35, "处理相位...")

            bbox = ROIBoundingBox(*self.roi_manager.get_bbox())
            phase_clips = {'Arterial': 0, 'Portal': 0, 'Delay': 0}
            total_clips = 0

            for phase_idx, (phase, (start_t, end_t)) in enumerate(phase_intervals.items()):
                if self._stop_flag.is_set():
                    break

                progress = 35 + phase_idx * 20
                self._emit_progress(video_path, progress, f"处理{phase}期...")

                clips = self._process_phase(
                    loader, phase, start_t, end_t, bbox, output_dir
                )

                phase_clips[phase] = clips
                total_clips += clips

            if self._stop_flag.is_set():
                return self._make_result(video_path, False, "已取消")

            self._emit_progress(video_path, 100, f"完成! {total_clips}个clips")

            return ProcessingResult(
                video_path=video_path,
                clips_count=total_clips,
                phase_clips=phase_clips,
                tic_result=fit_result,
                success=True
            )

        except Exception as e:
            logger.exception(f"处理失败: {video_path}")
            self.signals.error.emit(video_path, str(e))
            return self._make_result(video_path, False, str(e))

    def _sample_frames(self, loader: VideoLoader,
                       max_samples: int = 200) -> List[np.ndarray]:
        """采样帧"""
        step = max(1, loader.frame_count // max_samples)
        frames = []

        for frame_data in loader.iter_frames(step=step):
            if self._stop_flag.is_set():
                break
            if self.memory.is_memory_critical():
                self.memory.wait_for_memory()
            frames.append(frame_data.frame)

        return frames

    def _process_phase(self, loader: VideoLoader, phase: str,
                       start_t: float, end_t: float,
                       bbox: ROIBoundingBox, output_dir: str) -> int:

        # 1. 提取该相位的所有帧
        start_f = int(start_t * loader.fps)
        end_f = int(end_t * loader.fps)

        # 注意：这里需要同时拿 B-mode (用来筛选) 和 CEUS (用来训练)
        # 假设 loader.get_dual_frames(i) 能返回 (b_frame, c_frame)
        # 如果你现在的 loader 只能返回一种，你需要去修改 video_loader 让它支持返回双幅
        # 这里假设你已经有办法拿到两个列表：
        b_frames_list = []
        c_frames_list = []
        original_indices = []

        for i in range(start_f, end_f, 2):  # step=2 降采样一点没关系
            frame_data = loader.get_frame(i)  # 假设这里能拿到双幅
            if frame_data is None: continue

            # 这里你需要自己实现一下 split_dual_view
            # 通常是 frame[:, :width//2] 和 frame[:, width//2:]
            b_img, c_img = self.split_dual_view(frame_data)

            b_frames_list.append(b_img)
            c_frames_list.append(c_img)
            original_indices.append(i)

        if len(b_frames_list) < 16:
            return 0

        # 2. 呼吸门控筛选 (在 B-mode 上做)
        # 这里的 mask 应该是 B-mode 侧的 mask
        peak_indices = self.filter.filter_by_respiration(b_frames_list, self.roi_manager.mask, self.manual_anchor_frame)

        # 3. VideoMAE 采样 (取16帧)
        final_indices_local = self.filter.sample_for_videomae(peak_indices)

        # 4. 提取最终数据
        final_c_frames = [c_frames_list[i] for i in final_indices_local]
        final_timestamps = [original_indices[i] / loader.fps for i in final_indices_local]

        # 5. 保存
        success = self.generator.save_training_sample(
            frames=final_c_frames,
            timestamps=final_timestamps,
            roi_mask=self.roi_manager.mask,  # 注意：这是B-mode mask，如果左右对称可以直接用
            case_id=os.path.basename(loader.filepath).split('.')[0],
            phase_name=phase
        )

        return 1 if success else 0

    def split_dual_view(self, frame):
        # 简单的左右分割辅助函数
        h, w = frame.shape[:2]
        mid = w // 2
        return frame[:, :mid], frame[:, mid:]

    def _emit_progress(self, video_path: str, percent: int, message: str):
        """发送进度信号"""
        self.signals.progress.emit(video_path, percent, message)
        logger.debug(f"[{percent}%] {message}")

    def _make_result(self, video_path: str, success: bool,
                     message: str = "") -> ProcessingResult:
        """创建结果对象"""
        return ProcessingResult(
            video_path=video_path,
            clips_count=0,
            phase_clips={},
            tic_result=None,
            success=success,
            error_message=message
        )


# 需要导入os
import os