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

        self.manual_anchor_frame = None


    def set_anchor_frame(self, frame: np.ndarray):
        """设置人工呼吸基准帧"""
        self.manual_anchor_frame = frame
        logger.info("Worker 已接收人工基准帧")

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

        anchor_idx = getattr(task, "selected_frame_idx", 0)
        manual_anchor_frame = loader.get_frame(anchor_idx)

        if manual_anchor_frame is not None:
            logger.info(f"成功加载基准帧 (Index: {anchor_idx})")
        else:
            logger.warning(f"无法加载基准帧 (Index: {anchor_idx})，将导致处理跳过")

        # 区域配置
        bmode_rect = RegionRect.from_tuple(task.bmode_rect)
        ceus_rect = RegionRect.from_tuple(task.ceus_rect)

        # 获取CEUS区域内的ROI mask（局部坐标）
        ceus_roi_mask = task.ceus_roi_mask

        # 计算CEUS区域内ROI的最小外接矩形
        ceus_local_mask = self._extract_ceus_local_mask(ceus_roi_mask, ceus_rect, task.frame_shape)
        roi_bbox = ROIBoundingBox.from_mask(ceus_local_mask, padding=10)

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
        generator = ClipGenerator(task.output_dir)

        # output_dir = create_output_structure(
        #     os.path.dirname(video_path), task.output_dir, task.relative_path
        # )

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

            # === 第一遍扫描 (Pass 1): 只算分，不存图 ===
            logger.info(f"{phase}期: Pass 1 - 计算呼吸曲线...")

            ssim_scores = []
            valid_global_indices = []  # 记录实际的帧号

            # 1. 准备基准帧数据 (预处理一次，重复使用)
            if manual_anchor_frame is not None:
                # 裁剪 B-mode 区域作为基准
                anchor_frame_bmode = bmode_rect.crop(manual_anchor_frame)
                anchor_processed = filter_.prepare_anchor(anchor_frame_bmode, task.bmode_roi_mask)
            else:
                # 如果没有基准帧，随便拿第一帧顶替 (或者跳过)
                logger.warning("未设置人工基准帧，跳过呼吸筛选")
                continue

            # 2. 流式遍历 (内存占用极低)
            for frame_data in loader.iter_frames(start_f, end_f, step=2):
                if self._memory.is_memory_critical():
                    self._memory.wait_for_memory()

                # 提取 B-mode
                bmode_frame = bmode_rect.crop(frame_data.frame)

                # 计算 SSIM 并立即丢弃图像
                score = filter_.compute_single_ssim(bmode_frame, anchor_processed, task.bmode_roi_mask)

                ssim_scores.append(score)
                valid_global_indices.append(frame_data.index)

                # 显式删除引用，加速回收
                del bmode_frame

            if len(ssim_scores) < config.video.CLIP_LENGTH:
                continue

            # === 索引选择 ===
            # 1. 找波峰 (在 scores 列表里找，返回的是局部索引 0, 1, 2...)
            peak_local_indices = filter_.find_peaks_from_scores(ssim_scores)

            # 2. 映射回全局帧号
            peak_global_indices = [valid_global_indices[i] for i in peak_local_indices]

            # 3. VideoMAE 采样 (选出最终的 16 个帧号)
            final_global_indices = filter_.sample_for_videomae(peak_global_indices)

            if len(final_global_indices) == 0:
                continue

            # === 第二遍扫描 (Pass 2): 精准读取 (只读16帧) ===
            logger.info(f"{phase}期: Pass 2 - 提取样本帧...")

            final_ceus_frames = []
            final_timestamps = []

            # 使用随机访问接口 get_frame
            for idx in final_global_indices:
                frame = loader.get_frame(idx)
                if frame is None: continue

                # 提取 CEUS
                ceus_frame = ceus_rect.crop(frame)
                final_ceus_frames.append(ceus_frame)

                # 记录时间戳
                final_timestamps.append(idx / loader.fps)

            # === 保存 ===
            ceus_local_mask = self._extract_ceus_local_mask(task.ceus_roi_mask, ceus_rect, task.frame_shape)

            success = generator.save_training_sample(
                frames=final_ceus_frames,
                timestamps=final_timestamps,
                roi_mask=ceus_local_mask,
                case_id=os.path.basename(video_path).split('.')[0],
                phase_name=phase
            )

            if success:
                total_clips += 1

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