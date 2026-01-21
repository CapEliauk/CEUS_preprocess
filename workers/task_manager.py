"""
任务管理器 - 支持双区域（B-mode/CEUS）
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import threading

from utils.logger import get_logger

logger = get_logger('task_manager')


class VideoStatus(Enum):
    PENDING_ANNOTATION = "待标注"
    ANNOTATING = "标注中"
    PENDING_PROCESS = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "失败"


@dataclass
class VideoTask:
    """视频任务 - 支持双区域"""
    video_path: str
    relative_path: str
    output_dir: str = ""
    status: VideoStatus = VideoStatus.PENDING_ANNOTATION

    # 帧选择
    selected_frame_idx: int = 0

    # 双区域配置
    bmode_rect: Optional[Tuple[int, int, int, int]] = None
    ceus_rect: Optional[Tuple[int, int, int, int]] = None
    frame_shape: Optional[Tuple[int, int]] = None  # (H, W)

    # B-mode上的ROI（用于SSIM计算）
    bmode_roi_mask: Optional[np.ndarray] = None
    bmode_roi_points: Optional[List[List[float]]] = None
    bmode_roi_shape_type: str = "polygon"

    # CEUS上的映射ROI（用于TIC计算）
    ceus_roi_mask: Optional[np.ndarray] = None

    # 处理结果
    clips_generated: int = 0
    error_message: str = ""
    progress: int = 0


class TaskManager(QObject):
    """任务管理器"""

    task_added = pyqtSignal(str)
    task_status_changed = pyqtSignal(str, str)
    annotation_ready = pyqtSignal(str)
    all_annotated = pyqtSignal()
    all_completed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._tasks: Dict[str, VideoTask] = {}
        self._annotation_queue: List[str] = []
        self._process_queue: List[str] = []
        self._lock = threading.Lock()

    def add_video(self, video_path: str, relative_path: str, output_dir: str = ""):
        """添加视频"""
        with self._lock:
            if video_path not in self._tasks:
                task = VideoTask(
                    video_path=video_path,
                    relative_path=relative_path,
                    output_dir=output_dir
                )
                self._tasks[video_path] = task
                self._annotation_queue.append(video_path)
                self.task_added.emit(video_path)
                logger.debug(f"添加任务: {relative_path}")

    def set_output_dir(self, output_dir: str):
        with self._lock:
            for task in self._tasks.values():
                task.output_dir = output_dir

    def get_next_annotation_task(self) -> Optional[VideoTask]:
        with self._lock:
            if self._annotation_queue:
                path = self._annotation_queue[0]
                task = self._tasks.get(path)
                if task:
                    task.status = VideoStatus.ANNOTATING
                    self.task_status_changed.emit(path, task.status.value)
                return task
        return None

    def complete_annotation(self, video_path: str,
                           bmode_rect: Tuple[int, int, int, int],
                           ceus_rect: Tuple[int, int, int, int],
                           frame_shape: Tuple[int, int],
                           bmode_roi_mask: np.ndarray,
                           bmode_roi_points: List[List[float]],
                           bmode_roi_shape_type: str,
                           ceus_roi_mask: np.ndarray,
                           selected_frame_idx: int):
        """完成标注"""
        with self._lock:
            if video_path in self._tasks:
                task = self._tasks[video_path]

                task.bmode_rect = bmode_rect
                task.ceus_rect = ceus_rect
                task.frame_shape = frame_shape
                task.bmode_roi_mask = bmode_roi_mask.copy()
                task.bmode_roi_points = bmode_roi_points
                task.bmode_roi_shape_type = bmode_roi_shape_type
                task.ceus_roi_mask = ceus_roi_mask.copy()
                task.selected_frame_idx = selected_frame_idx
                task.status = VideoStatus.PENDING_PROCESS

                if video_path in self._annotation_queue:
                    self._annotation_queue.remove(video_path)
                self._process_queue.append(video_path)

                self.task_status_changed.emit(video_path, task.status.value)
                self.annotation_ready.emit(video_path)

                logger.info(f"标注完成: {task.relative_path}")

                if not self._annotation_queue:
                    self.all_annotated.emit()

    def skip_annotation(self, video_path: str):
        with self._lock:
            if video_path in self._annotation_queue:
                self._annotation_queue.remove(video_path)
                self._annotation_queue.append(video_path)
                if video_path in self._tasks:
                    self._tasks[video_path].status = VideoStatus.PENDING_ANNOTATION

    def get_next_process_task(self) -> Optional[VideoTask]:
        with self._lock:
            for path in self._process_queue:
                task = self._tasks.get(path)
                if task and task.status == VideoStatus.PENDING_PROCESS:
                    task.status = VideoStatus.PROCESSING
                    self.task_status_changed.emit(path, task.status.value)
                    return task
        return None

    def complete_process(self, video_path: str, success: bool,
                        clips_count: int = 0, error: str = ""):
        with self._lock:
            if video_path in self._tasks:
                task = self._tasks[video_path]
                task.status = VideoStatus.COMPLETED if success else VideoStatus.FAILED
                task.clips_generated = clips_count
                task.error_message = error
                task.progress = 100 if success else task.progress

                if video_path in self._process_queue:
                    self._process_queue.remove(video_path)

                self.task_status_changed.emit(video_path, task.status.value)

                logger.info(f"处理完成: {task.relative_path}, {'成功' if success else '失败'}")

                if self._all_done():
                    self.all_completed.emit()

    def update_progress(self, video_path: str, progress: int):
        with self._lock:
            if video_path in self._tasks:
                self._tasks[video_path].progress = progress

    def _all_done(self) -> bool:
        for task in self._tasks.values():
            if task.status not in [VideoStatus.COMPLETED, VideoStatus.FAILED]:
                return False
        return True

    def get_task(self, video_path: str) -> Optional[VideoTask]:
        return self._tasks.get(video_path)

    def get_all_tasks(self) -> List[VideoTask]:
        return list(self._tasks.values())

    def get_pending_annotation_count(self) -> int:
        return len(self._annotation_queue)

    def get_stats(self) -> Dict[str, int]:
        stats = {s.value: 0 for s in VideoStatus}
        for task in self._tasks.values():
            stats[task.status.value] += 1
        return stats

    def clear(self):
        with self._lock:
            self._tasks.clear()
            self._annotation_queue.clear()
            self._process_queue.clear()