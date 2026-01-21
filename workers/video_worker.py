"""
视频处理工作器 - 支持双区域处理，基于ROI裁剪
"""
import os
import threading
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from config import config
from core.video_loader import create_video_loader
from core.tic_analyzer import TICAnalyzer
from core.frame_filter import FrameFilter
from core.clip_generator import ClipGenerator, ROIBoundingBox
from core.roi_manager import DualRegionROIManager, RegionRect
from workers.task_manager import TaskManager, VideoTask, VideoStatus
from utils.memory_manager import MemoryManager
from utils.file_utils import create_output_structure, get_output_filename
from utils.logger import get_logger

logger = get_logger('video_worker')


class VideoProcessorWorker(QObject):
    """视频处理工作器 - 双区域版本"""

    progress_updated = pyqtSignal(str, int, str)
    video_completed = pyqtSignal(str, bool, int, str)
    tic_computed = pyqtSignal(str, object)

    def __init__(self, task_manager: TaskManager):
        super().__init__()
        self.task_manager = task_manager

        self._running = False
        self._paused = False
        self._pause_condition = threading.Condition()

        self._memory = MemoryManager()

    def start(self):
        self._running = True
        logger.info("处理工作器启动")
        self._process_loop()

    def pause(self):
        with self._pause_condition:
            self._paused = True
        logger.info("处理已暂停")

    def resume(self):
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        logger.info("处理已恢复")

    def stop(self):
        self._running = False
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        logger.info("处理工作器停止")

    def _process_loop(self):
        while self._running:
            with self._pause_condition:
                while self._paused and self._running:
                    self._pause_condition.wait()

            if not self._running:
                break

            task = self.task_manager.get_next_process_task()

            if task is None:
                import time
                time.sleep(0.5)
                continue

            try:
                clips_count = self._process_video(task)
                self.task_manager.complete_process(task.video_path, True, clips_count)
                self.video_completed.emit(task.video_path, True, clips_count, "")
            except Exception as e:
                logger.exception(f"处理失败: {task.video_path}")
                self.task_manager.complete_process(task.video_path, False, 0, str(e))
                self.video_completed.emit(task.video_path, False, 0, str(e))

    def _process_video(self, task: VideoTask) -> int:
        """处理视频 - 双区域版本"""
        video_path = task.video_path

        self.progress_updated.emit(video_path, 0, "Loading video...")
        loader = create_video_loader(video_path)

        # 区域配置
        bmode_rect = RegionRect.from_tuple(task.bmode_rect)
        ceus_rect = RegionRect.from_tuple(task.ceus_rect)

        # 获取CEUS区域内的ROI mask（局部坐标）
        ceus_roi_mask = task.ceus_roi_mask

        # 计算CEUS区域内ROI的最小外接矩形
        ceus_local_mask = self._extract_ceus_local_mask(ceus_roi_mask, ceus_rect, task.frame_shape)
        roi_bbox = ROIBoundingBox.from_contours(ceus_local_mask, padding=10)

        logger.info(f"CEUS区域: {ceus_rect.to_tuple()}, ROI bbox: {roi_bbox}")

        self.progress_updated.emit(video_path, 10, "Sampling frames...")

        # 采样帧用于TIC计算（从CEUS区域）
        sample_frames = []
        sample_step = max(1, loader.frame_count // 200)

        for frame_data in loader.iter_frames(step=sample_step):
            if self._memory.is_memory_critical():
                self._memory.wait_for_memory()
            # 提取CEUS区域
            ceus_region = ceus_rect.crop(frame_data.frame)
            sample_frames.append(ceus_region)
            if len(sample_frames) >= 200:
                break

        self.progress_updated.emit(video_path, 20, "Computing TIC curve...")

        # 使用CEUS区域计算TIC
        analyzer = TICAnalyzer(loader.fps)

        if ceus_local_mask is not None and sample_frames:
            frames_array = np.stack(sample_frames, axis=0)
            analyzer.tic_data = analyzer.extract_roi_intensities_batch(frames_array, ceus_local_mask)
            analyzer.time_axis = np.arange(len(sample_frames)) * sample_step / loader.fps

        # 拟合TIC
        self.progress_updated.emit(video_path, 30, "Fitting TIC model...")
        fit_result = analyzer.fit_gamma_model()
        self.tic_computed.emit(video_path, fit_result)

        # 获取相位区间
        phase_intervals = analyzer.get_phase_intervals()

        self.progress_updated.emit(video_path, 40, "Filtering frames...")

        # 帧筛选和clip生成
        filter_ = FrameFilter()
        generator = ClipGenerator()

        output_dir = create_output_structure(
            os.path.dirname(video_path), task.output_dir, task.relative_path
        )

        total_clips = 0
        phase_idx = 0

        for phase, (start_t, end_t) in phase_intervals.items():
            if not self._running:
                break

            phase_idx += 1
            progress = 40 + phase_idx * 18
            self.progress_updated.emit(video_path, progress, f"Processing {phase} phase...")

            start_f = int(start_t * loader.fps)
            end_f = min(int(end_t * loader.fps), loader.frame_count)

            # 收集帧
            phase_bmode_frames = []
            phase_ceus_frames = []
            phase_indices = []

            for frame_data in loader.iter_frames(start_f, end_f, step=2):
                if self._memory.is_memory_critical():
                    self._memory.wait_for_memory()

                # 分别提取两个区域
                bmode_frame = bmode_rect.crop(frame_data.frame)
                ceus_frame = ceus_rect.crop(frame_data.frame)

                phase_bmode_frames.append(bmode_frame)
                phase_ceus_frames.append(ceus_frame)
                phase_indices.append(frame_data.index)

            if len(phase_bmode_frames) < config.video.CLIP_LENGTH:
                continue

            # 在B-mode区域使用SSIM筛选
            bmode_mask = task.bmode_roi_mask
            _, valid_idx = filter_.filter_frames(phase_bmode_frames, bmode_mask)

            if len(valid_idx) < config.video.CLIP_LENGTH:
                continue

            # 使用相同索引获取CEUS帧
            filtered_ceus = [phase_ceus_frames[i] for i in valid_idx]
            filtered_indices = [phase_indices[i] for i in valid_idx]

            # 在CEUS区域生成clips - 使用ROI最小外接矩形裁剪到224x224
            clip_count = 0
            for clip in generator.generate_clips(
                frames=filtered_ceus,
                indices=filtered_indices,
                roi_mask=ceus_local_mask,
                analyzer=analyzer,
                padding=10,
                make_square=True
            ):
                if not self._running:
                    break

                clip.phase = phase
                filename = get_output_filename(os.path.basename(video_path), phase, clip_count)
                output_path = os.path.join(output_dir, filename)

                if generator.save_clip(clip, output_path, loader.fps):
                    clip_count += 1
                    total_clips += 1

            logger.info(f"{phase}期: 生成 {clip_count} 个clips")

            # 清理内存
            del phase_bmode_frames, phase_ceus_frames, filtered_ceus
            self._memory.force_gc()

        loader.release()

        self.progress_updated.emit(video_path, 100, f"Done! {total_clips} clips")
        self.task_manager.update_progress(video_path, 100)

        return total_clips

    def _extract_ceus_local_mask(self, ceus_roi_mask: np.ndarray,
                                  ceus_rect: RegionRect,
                                  frame_shape: tuple) -> np.ndarray:
        """
        从完整帧的CEUS mask中提取CEUS区域内的局部mask
        """
        if ceus_roi_mask is None:
            # 返回全1 mask
            return np.ones((ceus_rect.height, ceus_rect.width), dtype=np.uint8) * 255

        # 如果mask尺寸与frame_shape相同，需要裁剪
        if ceus_roi_mask.shape[0] == frame_shape[0] and ceus_roi_mask.shape[1] == frame_shape[1]:
            local_mask = ceus_roi_mask[
                ceus_rect.y:ceus_rect.y + ceus_rect.height,
                ceus_rect.x:ceus_rect.x + ceus_rect.width
            ]
            return local_mask.copy()

        # 如果mask尺寸与CEUS区域相同，直接返回
        if (ceus_roi_mask.shape[0] == ceus_rect.height and
            ceus_roi_mask.shape[1] == ceus_rect.width):
            return ceus_roi_mask.copy()

        # 其他情况，缩放到CEUS区域大小
        return cv2.resize(ceus_roi_mask, (ceus_rect.width, ceus_rect.height),
                         interpolation=cv2.INTER_NEAREST)


class VideoProcessorThread(QThread):
    def __init__(self, worker: VideoProcessorWorker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.start()