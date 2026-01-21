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
from core.clip_generator import ClipGenerator, Clip, ROIBoundingBox
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
        """处理单个相位"""
        start_f = int(start_t * loader.fps)
        end_f = min(int(end_t * loader.fps), loader.frame_count)

        # 收集相位帧
        phase_frames = []
        phase_indices = []

        for frame_data in loader.iter_frames(start_f, end_f, step=2):
            if self._stop_flag.is_set():
                break

            # 运动补偿
            frame, mask = self.roi_manager.process_frame(frame_data.frame)
            phase_frames.append(frame)
            phase_indices.append(frame_data.index)

            if self.memory.is_memory_critical():
                self.memory.wait_for_memory()

        if len(phase_frames) < config.video.CLIP_LENGTH:
            return 0

        # 筛选帧
        filtered, valid_idx = self.filter.filter_frames(
            phase_frames, self.roi_manager.mask
        )

        if len(filtered) < config.video.CLIP_LENGTH:
            return 0

        filtered_indices = [phase_indices[i] for i in valid_idx]

        # 生成clips
        clip_count = 0
        for clip in self.generator.generate_clips(
                filtered, filtered_indices, bbox, self.analyzer
        ):
            if self._stop_flag.is_set():
                break

            clip.phase = phase
            filename = f"{phase}_clip{clip_count:04d}{config.video.OUTPUT_FORMAT}"
            output_path = os.path.join(output_dir, filename)

            self.generator.save_clip(clip, output_path, loader.fps)
            self.signals.clip_generated.emit(loader.filepath, phase, clip_count)

            clip_count += 1

        # 清理
        del phase_frames, filtered
        self.memory.force_gc()

        return clip_count

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