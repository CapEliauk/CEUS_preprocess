"""
主窗口 - 双区域标注工作流程
"""
import os
from typing import Optional, Dict
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QPushButton, QLabel, QFileDialog,
                             QListWidget, QListWidgetItem, QGroupBox,
                             QTextEdit, QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QBrush, QColor

from config import config
from gui.frame_selector import FrameSelector
from gui.region_selector import DualRegionSelector
from gui.roi_annotator import BmodeROIAnnotator
from core.video_loader import create_video_loader, VideoLoader
from core.roi_manager import DualRegionROIManager, ROICoordinateMapper, RegionRect
from core.tic_analyzer import TICFitResult
from workers.task_manager import TaskManager, VideoTask, VideoStatus
from workers.video_worker import VideoProcessorWorker, VideoProcessorThread
from utils.file_utils import find_video_files
from utils.memory_manager import MemoryManager
from utils.logger import get_logger

logger = get_logger('main_window')


class VideoListItem(QListWidgetItem):
    STATUS_COLORS = {
        VideoStatus.PENDING_ANNOTATION: "#FFE082",
        VideoStatus.ANNOTATING: "#81D4FA",
        VideoStatus.PENDING_PROCESS: "#CE93D8",
        VideoStatus.PROCESSING: "#90CAF9",
        VideoStatus.COMPLETED: "#A5D6A7",
        VideoStatus.FAILED: "#EF9A9A",
    }

    def __init__(self, task: VideoTask):
        super().__init__()
        self.video_path = task.video_path
        self.update_display(task)

    def update_display(self, task: VideoTask):
        progress_str = f" [{task.progress}%]" if task.status == VideoStatus.PROCESSING else ""
        clips_str = f" ({task.clips_generated}clips)" if task.status == VideoStatus.COMPLETED else ""
        self.setText(f"{task.relative_path} - {task.status.value}{progress_str}{clips_str}")
        color = self.STATUS_COLORS.get(task.status, "#FFFFFF")
        self.setBackground(QBrush(QColor(color)))


class MainWindow(QMainWindow):
    """主窗口 - 双区域标注流程"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CEUS视频处理系统 - 双区域模式")
        self.setMinimumSize(1600, 1000)

        # 状态
        self.video_loader: Optional[VideoLoader] = None
        self.current_task: Optional[VideoTask] = None
        self.output_dir: Optional[str] = None
        self.list_items: Dict[str, VideoListItem] = {}

        # 当前标注状态
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_idx: int = 0
        self.bmode_rect: Optional[tuple] = None
        self.ceus_rect: Optional[tuple] = None

        # ROI管理
        self.roi_manager = DualRegionROIManager()

        # 任务管理
        self.task_manager = TaskManager()

        # 后台处理
        self.worker = VideoProcessorWorker(self.task_manager)
        self.worker_thread: Optional[VideoProcessorThread] = None

        # 内存管理
        self.memory = MemoryManager()
        self.memory.start_monitoring()

        self._setup_ui()
        self._connect_signals()

        # 刷新定时器
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_progress)
        self.refresh_timer.start(500)

        logger.info("主窗口初始化完成")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # ===== 左侧面板 =====
        left_panel = QWidget()
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(left_panel)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)

        self.open_folder_btn = QPushButton("📂 打开文件夹")
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        file_layout.addWidget(self.open_folder_btn)

        out_row = QHBoxLayout()
        self.output_label = QLabel("输出: 未设置")
        out_row.addWidget(self.output_label, 1)
        self.output_btn = QPushButton("设置")
        self.output_btn.clicked.connect(self._on_set_output)
        out_row.addWidget(self.output_btn)
        file_layout.addLayout(out_row)

        left_layout.addWidget(file_group)

        # 视频列表
        list_group = QGroupBox("视频列表")
        list_layout = QVBoxLayout(list_group)
        self.video_list = QListWidget()
        self.video_list.itemDoubleClicked.connect(self._on_video_double_click)
        list_layout.addWidget(self.video_list)
        self.stats_label = QLabel("待标注: 0 | 待处理: 0 | 完成: 0")
        list_layout.addWidget(self.stats_label)
        left_layout.addWidget(list_group, 1)

        # 控制
        ctrl_group = QGroupBox("控制")
        ctrl_layout = QHBoxLayout(ctrl_group)
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        ctrl_layout.addWidget(self.start_btn)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.clicked.connect(self._on_toggle_pause)
        self.pause_btn.setEnabled(False)
        ctrl_layout.addWidget(self.pause_btn)
        left_layout.addWidget(ctrl_group)

        main_layout.addWidget(left_panel)

        # ===== 右侧面板 =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 工作流程标签页
        self.tabs = QTabWidget()

        # Tab 1: 帧选择
        self.frame_selector = FrameSelector()
        self.frame_selector.frame_selected.connect(self._on_frame_selected)
        self.tabs.addTab(self.frame_selector, "1️⃣ 选择帧")

        # Tab 2: 区域选择
        self.region_selector = DualRegionSelector()
        self.region_selector.regions_confirmed.connect(self._on_regions_confirmed)
        self.tabs.addTab(self.region_selector, "2️⃣ 选择区域")

        # Tab 3: ROI标注
        self.roi_annotator = BmodeROIAnnotator()
        self.roi_annotator.roi_annotated.connect(self._on_roi_annotated)
        self.roi_annotator.annotation_cancelled.connect(self._on_annotation_skipped)
        self.tabs.addTab(self.roi_annotator, "3️⃣ 标注ROI")

        right_layout.addWidget(self.tabs)

        # 进度信息
        info_group = QGroupBox("当前状态")
        info_layout = QVBoxLayout(info_group)
        self.current_video_label = QLabel("当前视频: 无")
        self.current_video_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.current_video_label)
        self.step_label = QLabel("步骤: 等待开始")
        info_layout.addWidget(self.step_label)
        self.tic_info = QLabel("TIC: 未计算")
        self.tic_info.setStyleSheet("color: #2196F3;")
        info_layout.addWidget(self.tic_info)
        right_layout.addWidget(info_group)

        # 日志
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        log_layout.addWidget(self.log)
        right_layout.addWidget(log_group)

        main_layout.addWidget(right_panel, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.mem_label = QLabel("内存: 0%")
        self.status_bar.addPermanentWidget(self.mem_label)

    def _connect_signals(self):
        self.task_manager.task_status_changed.connect(self._on_task_status_changed)
        self.task_manager.annotation_ready.connect(self._on_annotation_ready)
        self.task_manager.all_annotated.connect(self._on_all_annotated)
        self.task_manager.all_completed.connect(self._on_all_completed)

        self.worker.progress_updated.connect(self._on_process_progress)
        self.worker.video_completed.connect(self._on_video_completed)
        self.worker.tic_computed.connect(self._on_tic_computed)

        self.memory.register_callback(self._on_memory_update)

    def _log(self, msg: str):
        self.log.append(msg)
        logger.info(msg)

    def _on_memory_update(self, usage: float):
        self.mem_label.setText(f"内存: {usage:.1f}%")
        self.mem_label.setStyleSheet("color: red;" if usage > 70 else "")

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if not folder:
            return

        self.task_manager.clear()
        self.video_list.clear()
        self.list_items.clear()

        videos = list(find_video_files(folder))
        if not videos:
            QMessageBox.warning(self, "警告", "未找到视频")
            return

        for path, rel in videos:
            self.task_manager.add_video(path, rel, self.output_dir or "")
            task = self.task_manager.get_task(path)
            item = VideoListItem(task)
            self.list_items[path] = item
            self.video_list.addItem(item)

        self._log(f"找到 {len(videos)} 个视频")
        self._update_stats()

    def _on_set_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if folder:
            self.output_dir = folder
            self.output_label.setText(f"输出: {os.path.basename(folder)}")
            self.task_manager.set_output_dir(folder)

    def _on_start(self):
        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先设置输出目录")
            return
        if self.task_manager.get_pending_annotation_count() == 0:
            QMessageBox.warning(self, "警告", "没有待处理的视频")
            return

        # 启动后台线程
        if self.worker_thread is None or not self.worker_thread.isRunning():
            self.worker_thread = VideoProcessorThread(self.worker)
            self.worker_thread.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)

        self._load_next_task()

    def _load_next_task(self):
        task = self.task_manager.get_next_annotation_task()
        if task is None:
            self.current_task = None
            self.current_video_label.setText("当前视频: 全部标注完成")
            self.step_label.setText("步骤: 等待处理完成")
            return

        self.current_task = task
        self._load_video(task)

    def _load_video(self, task: VideoTask):
        try:
            if self.video_loader:
                self.video_loader.release()

            self.video_loader = create_video_loader(task.video_path)
            self.frame_selector.set_video(self.video_loader)
            self.tabs.setCurrentIndex(0)

            self.current_video_label.setText(f"当前视频: {task.relative_path}")
            self.step_label.setText("步骤: 1/3 选择最亮帧")
            self._log(f"加载: {task.relative_path}")

            self._update_list_item(task.video_path)
            self._update_stats()

        except Exception as e:
            self._log(f"加载失败: {e}")
            self.task_manager.skip_annotation(task.video_path)
            self._load_next_task()

    @pyqtSlot(int, np.ndarray)
    def _on_frame_selected(self, frame_idx: int, frame: np.ndarray):
        """帧选择完成 -> 进入区域选择"""
        self.current_frame = frame.copy()
        self.current_frame_idx = frame_idx

        self.region_selector.set_frame(frame)
        self.tabs.setCurrentIndex(1)

        self.step_label.setText("步骤: 2/3 选择B-mode和CEUS区域")
        self._log(f"选择帧 {frame_idx}")

    @pyqtSlot(tuple, tuple)
    def _on_regions_confirmed(self, bmode_rect: tuple, ceus_rect: tuple):
        """区域选择完成 -> 进入ROI标注"""
        self.bmode_rect = bmode_rect
        self.ceus_rect = ceus_rect

        # 设置ROI管理器
        h, w = self.current_frame.shape[:2]
        self.roi_manager.set_dual_regions(bmode_rect, ceus_rect, (h, w))

        # 裁剪B-mode区域
        bmode_image = self.roi_manager.get_bmode_region(self.current_frame)

        video_name = os.path.basename(self.current_task.video_path) if self.current_task else ""
        self.roi_annotator.set_bmode_image(bmode_image, bmode_rect, video_name)
        self.tabs.setCurrentIndex(2)

        self.step_label.setText("步骤: 3/3 在B-mode上标注ROI")
        self._log(f"区域确认: B-mode={bmode_rect}, CEUS={ceus_rect}")

    @pyqtSlot(np.ndarray, list, str)
    def _on_roi_annotated(self, bmode_mask: np.ndarray, points: list, shape_type: str):
        """ROI标注完成 -> 映射并保存"""
        if self.current_task is None:
            return

        # 设置B-mode ROI
        self.roi_manager.set_bmode_roi(points, shape_type, bmode_mask)

        # 获取映射后的CEUS mask
        h, w = self.current_frame.shape[:2]
        self.roi_manager.frame_shape = (h, w)
        self.roi_manager._create_ceus_mask()

        ceus_mask = self.roi_manager.ceus_mask

        # 完成标注
        self.task_manager.complete_annotation(
            self.current_task.video_path,
            self.bmode_rect,
            self.ceus_rect,
            (h, w),
            bmode_mask,
            points,
            shape_type,
            ceus_mask,
            self.current_frame_idx
        )

        self._log(f"✓ {self.current_task.relative_path} 标注完成")

        # 加载下一个
        self._load_next_task()

    @pyqtSlot()
    def _on_annotation_skipped(self):
        if self.current_task:
            self._log(f"跳过: {self.current_task.relative_path}")
            self.task_manager.skip_annotation(self.current_task.video_path)
            self._load_next_task()

    def _on_video_double_click(self, item: VideoListItem):
        task = self.task_manager.get_task(item.video_path)
        if task and task.status == VideoStatus.PENDING_ANNOTATION:
            self.current_task = task
            self._load_video(task)

    @pyqtSlot(str, str)
    def _on_task_status_changed(self, path: str, status: str):
        self._update_list_item(path)
        self._update_stats()

    @pyqtSlot(str)
    def _on_annotation_ready(self, path: str):
        self._update_list_item(path)

    @pyqtSlot()
    def _on_all_annotated(self):
        self._log("✓ 所有视频标注完成!")
        QMessageBox.information(self, "提示", "所有视频标注完成！后台正在处理...")

    @pyqtSlot()
    def _on_all_completed(self):
        self._log("🎉 全部处理完成!")
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        QMessageBox.information(self, "完成", "所有视频处理完成!")

    @pyqtSlot(str, int, str)
    def _on_process_progress(self, path: str, progress: int, msg: str):
        task = self.task_manager.get_task(path)
        if task:
            task.progress = progress
            self._update_list_item(path)

    @pyqtSlot(str, bool, int, str)
    def _on_video_completed(self, path: str, success: bool, clips: int, error: str):
        task = self.task_manager.get_task(path)
        if task:
            self._log(f"{'✓' if success else '✗'} {task.relative_path}: {clips}clips")
            self._update_list_item(path)
            self._update_stats()

    @pyqtSlot(str, object)
    def _on_tic_computed(self, path: str, result: TICFitResult):
        if result:
            self.tic_info.setText(
                f"TIC: 到达={result.arrival_time:.1f}s, 峰值={result.peak_time:.1f}s, R²={result.fit_quality:.3f}"
            )
        else:
            self.tic_info.setText("TIC: 拟合失败")

    def _on_toggle_pause(self):
        if self.pause_btn.text() == "⏸ 暂停":
            self.worker.pause()
            self.pause_btn.setText("▶ 继续")
        else:
            self.worker.resume()
            self.pause_btn.setText("⏸ 暂停")

    def _update_list_item(self, path: str):
        if path in self.list_items:
            task = self.task_manager.get_task(path)
            if task:
                self.list_items[path].update_display(task)

    def _update_stats(self):
        stats = self.task_manager.get_stats()
        pending = stats.get(VideoStatus.PENDING_ANNOTATION.value, 0)
        pending += stats.get(VideoStatus.ANNOTATING.value, 0)
        processing = stats.get(VideoStatus.PENDING_PROCESS.value, 0)
        processing += stats.get(VideoStatus.PROCESSING.value, 0)
        completed = stats.get(VideoStatus.COMPLETED.value, 0)
        failed = stats.get(VideoStatus.FAILED.value, 0)

        self.stats_label.setText(f"待标注: {pending} | 处理中: {processing} | 完成: {completed} | 失败: {failed}")

    def _refresh_progress(self):
        for path, item in self.list_items.items():
            task = self.task_manager.get_task(path)
            if task and task.status == VideoStatus.PROCESSING:
                item.update_display(task)
        self._update_stats()

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(self, "确认", "后台正在处理，确定退出?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.worker.stop()
            self.worker_thread.wait(3000)

        if self.video_loader:
            self.video_loader.release()

        self.roi_annotator.cleanup()
        self.memory.cleanup()
        self.refresh_timer.stop()
        event.accept()