"""
视频处理工作器
"""
import os
import threading
import time
import gc
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from config import config
from core.video_loader import create_video_loader
from core.tic_analyzer import TICAnalyzer
from core.frame_filter import FrameFilter
from core.clip_generator import ClipGenerator, ROIBoundingBox
from core.roi_manager import RegionRect
from workers.task_manager import TaskManager, VideoTask
from utils.memory_manager import MemoryManager
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
        if frame is not None:
            self.manual_anchor_frame = frame.copy() # 存副本，防止引用外部变量
        else:
            self.manual_anchor_frame = None
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
        """主处理循环 - 负责任务调度和顶级内存回收"""
        while self._running:
            with self._pause_condition:
                while self._paused and self._running:
                    self._pause_condition.wait()

            if not self._running:
                break

            task = self.task_manager.get_next_process_task()

            if task is None:
                time.sleep(0.5)
                continue

            # === 任务处理沙盒 ===
            try:
                clips_count = self._process_video(task)
                self.task_manager.complete_process(task.video_path, True, clips_count)
                self.video_completed.emit(task.video_path, True, clips_count, "")
            except Exception as e:
                logger.exception(f"处理失败: {task.video_path}")
                self.task_manager.complete_process(task.video_path, False, 0, str(e))
                self.video_completed.emit(task.video_path, False, 0, str(e))
            finally:
                # === 【核心修改】任务级强制清理 ===
                # 无论成功还是失败，只要跳出 _process_video，立即进行全量回收
                # 这会清除上一个视频产生的所有 Python 对象
                gc.collect()
                logger.info(f"任务 {task.relative_path} 结束，已强制执行GC")

    def _process_video(self, task: VideoTask) -> int:
        """
        处理单个视频 - 流式计算优化版
        """
        video_path = task.video_path
        self.progress_updated.emit(video_path, 0, "Loading video...")

        # 定义变量以便 finally 块清理
        loader = None
        analyzer = None
        filter_ = None
        generator = None
        manual_anchor_frame = None

        try:
            # 1. 创建加载器
            loader = create_video_loader(video_path)

            # 2. 加载基准帧
            anchor_idx = getattr(task, "selected_frame_idx", 0)
            manual_anchor_frame = loader.get_frame(anchor_idx)

            # 3. 准备 ROI
            bmode_rect = RegionRect.from_tuple(task.bmode_rect)
            ceus_rect = RegionRect.from_tuple(task.ceus_rect)
            ceus_roi_mask = task.ceus_roi_mask

            # 提取局部 Mask
            ceus_local_mask = self._extract_ceus_local_mask(ceus_roi_mask, ceus_rect, task.frame_shape)
            roi_bbox = ROIBoundingBox.from_mask(ceus_local_mask, padding=10)

            # === [核心优化] 4. TIC 流式计算 (不存帧，直接算) ===
            self.progress_updated.emit(video_path, 15, "Computing TIC curve (Streaming)...")

            analyzer = TICAnalyzer(loader.fps)

            tic_intensities = []
            tic_times = []

            # 采样步长
            sample_step = max(1, loader.frame_count // 200)

            # 逐帧迭代
            count = 0
            for frame_data in loader.iter_frames(step=sample_step):
                QThread.msleep(1)

                # 内存保护
                if self._memory.is_memory_critical():
                    self._memory.wait_for_memory()

                # 裁剪 CEUS 区域
                ceus_region = ceus_rect.crop(frame_data.frame)

                # [重点] 立即计算强度，不保存 frame 到列表！
                # 这一步计算完，ceus_region 就会被释放
                intensity = analyzer.extract_roi_intensity_single(ceus_region, ceus_local_mask)

                tic_intensities.append(intensity)
                tic_times.append(frame_data.index / loader.fps)

                # 及时清理当前帧变量
                del ceus_region

                count += 1
                if count >= 200:
                    break

            # 赋值给分析器
            analyzer.tic_data = np.array(tic_intensities)
            analyzer.time_axis = np.array(tic_times)

            # 清理加载器缓存（防止 VideoLoader 内部缓存了这 200 帧）
            if hasattr(loader, '_buffer'):
                loader._buffer.clear()
            gc.collect()

            self.progress_updated.emit(video_path, 30, "Fitting TIC model...")
            fit_result = analyzer.fit_gamma_model()
            self.tic_computed.emit(video_path, fit_result)

            phase_intervals = analyzer.get_phase_intervals()

            # 5. 初始化后续工具
            filter_ = FrameFilter()
            generator = ClipGenerator(task.output_dir)

            total_clips = 0
            phase_idx = 0

            # 6. 相位处理循环
            for phase, (start_t, end_t) in phase_intervals.items():
                if not self._running: break
                phase_idx += 1

                # 变量初始化
                ssim_scores = []
                valid_global_indices = []
                anchor_processed = None

                self.progress_updated.emit(video_path, 40 + phase_idx * 15, f"Processing {phase}...")

                start_f = int(start_t * loader.fps)
                end_f = min(int(end_t * loader.fps), loader.frame_count)

                # 准备基准图
                if manual_anchor_frame is not None:
                    anchor_frame_bmode = bmode_rect.crop(manual_anchor_frame)
                    anchor_processed = filter_.prepare_anchor(anchor_frame_bmode, task.bmode_roi_mask)
                    del anchor_frame_bmode
                else:
                    continue

                # Pass 1: 计算分数
                for frame_data in loader.iter_frames(start_f, end_f, step=2):
                    QThread.msleep(1)
                    if self._memory.is_memory_critical():
                        self._memory.wait_for_memory()

                    bmode_frame = bmode_rect.crop(frame_data.frame)
                    score = filter_.compute_single_ssim(bmode_frame, anchor_processed, task.bmode_roi_mask)

                    ssim_scores.append(score)
                    valid_global_indices.append(frame_data.index)
                    del bmode_frame

                # 筛选
                if len(ssim_scores) < config.video.CLIP_LENGTH:
                    continue
                peak_local_indices = filter_.find_peaks_from_scores(ssim_scores)
                peak_global_indices = [valid_global_indices[i] for i in peak_local_indices]
                final_global_indices = filter_.sample_for_videomae(peak_global_indices)

                # 清理中间变量
                del ssim_scores
                del valid_global_indices
                del anchor_processed

                if not final_global_indices: continue

                # Pass 2: 提取保存
                final_ceus_frames = []
                final_timestamps = []

                for idx in final_global_indices:
                    frame = loader.get_frame(idx)
                    if frame is None: continue
                    final_ceus_frames.append(ceus_rect.crop(frame))
                    final_timestamps.append(idx / loader.fps)
                    del frame

                success = generator.save_training_sample(
                    frames=final_ceus_frames,
                    timestamps=final_timestamps,
                    roi_mask=ceus_local_mask,
                    case_id=os.path.basename(video_path).split('.')[0],
                    phase_name=phase
                )

                if success: total_clips += 1

                # 每一轮相位结束后清理
                del final_ceus_frames
                gc.collect()

            self.progress_updated.emit(video_path, 100, f"Done! {total_clips} clips")
            return total_clips

        finally:
            logger.debug(f"清理资源: {video_path}")
            if loader:
                try:
                    loader.release()  # 必须调用，否则无法删除临时文件
                except:
                    pass
            del loader
            del analyzer
            del filter_
            del generator
            del manual_anchor_frame
            gc.collect()

    def _extract_ceus_local_mask(self, ceus_roi_mask: np.ndarray,
                                  ceus_rect: RegionRect,
                                  frame_shape: tuple) -> np.ndarray:
        if ceus_roi_mask is None:
            return np.ones((ceus_rect.height, ceus_rect.width), dtype=np.uint8) * 255
        if ceus_roi_mask.shape[0] == frame_shape[0] and ceus_roi_mask.shape[1] == frame_shape[1]:
            local_mask = ceus_roi_mask[
                ceus_rect.y:ceus_rect.y + ceus_rect.height,
                ceus_rect.x:ceus_rect.x + ceus_rect.width
            ]
            return local_mask.copy()
        if (ceus_roi_mask.shape[0] == ceus_rect.height and
            ceus_roi_mask.shape[1] == ceus_rect.width):
            return ceus_roi_mask.copy()
        return cv2.resize(ceus_roi_mask, (ceus_rect.width, ceus_rect.height),
                         interpolation=cv2.INTER_NEAREST)


class VideoProcessorThread(QThread):
    def __init__(self, worker: VideoProcessorWorker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.start()