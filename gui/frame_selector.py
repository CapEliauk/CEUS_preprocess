"""
帧选择器
"""
import numpy as np
import cv2
from typing import Optional
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QPushButton, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from core.video_loader import VideoLoader
from utils.logger import get_logger

logger = get_logger('frame_selector')


class FrameSelector(QWidget):
    """帧选择器"""

    # 信号
    frame_selected = pyqtSignal(int, np.ndarray)
    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_loader: Optional[VideoLoader] = None
        self.current_idx = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 显示区域
        self.display = QLabel()
        self.display.setMinimumSize(640, 480)
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        layout.addWidget(self.display)

        # 信息栏
        info = QHBoxLayout()
        self.frame_info = QLabel("帧: 0/0")
        self.time_info = QLabel("时间: 0.00s")
        self.brightness_info = QLabel("亮度: 0")
        for lbl in [self.frame_info, self.time_info, self.brightness_info]:
            lbl.setStyleSheet("font-size: 12px; color: #666;")
            info.addWidget(lbl)
        info.addStretch()
        layout.addLayout(info)

        # 滑块控制
        slider_group = QGroupBox("帧控制")
        slider_layout = QVBoxLayout(slider_group)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        slider_layout.addWidget(self.slider)

        # 控制按钮
        ctrl = QHBoxLayout()

        self.prev_btn = QPushButton("◀ 上一帧")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        ctrl.addWidget(self.prev_btn)

        self.spinbox = QSpinBox()
        self.spinbox.valueChanged.connect(self._on_spinbox)
        ctrl.addWidget(self.spinbox)

        self.next_btn = QPushButton("下一帧 ▶")
        self.next_btn.clicked.connect(lambda: self._step(1))
        ctrl.addWidget(self.next_btn)

        ctrl.addStretch()

        self.auto_btn = QPushButton("🔍 自动选最亮帧")
        self.auto_btn.clicked.connect(self._auto_select)
        self.auto_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px 16px;")
        ctrl.addWidget(self.auto_btn)

        self.confirm_btn = QPushButton("✓ 确认选择")
        self.confirm_btn.clicked.connect(self._confirm)
        self.confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 16px;")
        ctrl.addWidget(self.confirm_btn)

        slider_layout.addLayout(ctrl)
        layout.addWidget(slider_group)

    def set_video(self, loader: VideoLoader):
        """设置视频"""
        self.video_loader = loader
        max_idx = loader.frame_count - 1

        self.slider.setMaximum(max_idx)
        self.spinbox.setMaximum(max_idx)
        self.slider.setValue(0)
        self._show_frame(0)

        logger.debug(f"设置视频: {loader.frame_count}帧")

    def _show_frame(self, idx: int):
        """显示帧"""
        if not self.video_loader:
            return

        frame = self.video_loader.get_frame(idx)
        if frame is None:
            return

        self.current_idx = idx

        # 计算亮度（向量化）
        if frame.ndim == 3:
            gray = np.dot(frame[..., :3], [0.114, 0.587, 0.299])
        else:
            gray = frame
        brightness = np.mean(gray)

        # 更新信息
        self.frame_info.setText(f"帧: {idx}/{self.video_loader.frame_count-1}")
        self.time_info.setText(f"时间: {idx/self.video_loader.fps:.2f}s")
        self.brightness_info.setText(f"亮度: {brightness:.1f}")

        # 显示
        self._display_frame(frame)

        # 发送帧变化信号
        self.frame_changed.emit(idx)

    def _display_frame(self, frame: np.ndarray):
        """显示帧到界面"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape

        lbl_size = self.display.size()
        scale = min(lbl_size.width() / w, lbl_size.height() / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(rgb, (new_w, new_h))
        qimg = QImage(resized.data, new_w, new_h, c * new_w, QImage.Format_RGB888)
        self.display.setPixmap(QPixmap.fromImage(qimg))

    def _on_slider(self, val):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(val)
        self.spinbox.blockSignals(False)
        self._show_frame(val)

    def _on_spinbox(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)
        self._show_frame(val)

    def _step(self, delta):
        new_idx = max(0, min(self.current_idx + delta,
                            self.video_loader.frame_count - 1 if self.video_loader else 0))
        self.slider.setValue(new_idx)

    def _auto_select(self):
        """自动选择最亮帧"""
        if not self.video_loader:
            return

        from core.tic_analyzer import TICAnalyzer

        # 采样前1/3或前60秒
        end = min(self.video_loader.frame_count // 3, int(60 * self.video_loader.fps))

        frames = []
        for frame_data in self.video_loader.iter_frames(end=end, step=5):
            frames.append(frame_data.frame)
            if len(frames) >= 100:
                break

        if frames:
            frames_array = np.stack(frames, axis=0)
            analyzer = TICAnalyzer(self.video_loader.fps)
            brightest = analyzer.find_brightest_frame_batch(frames_array)
            self.slider.setValue(brightest * 5)

            logger.info(f"自动选择最亮帧: {brightest * 5}")

    def _confirm(self):
        """确认选择"""
        if self.video_loader:
            frame = self.video_loader.get_frame(self.current_idx)
            if frame is not None:
                self.frame_selected.emit(self.current_idx, frame)
                logger.info(f"确认选择帧: {self.current_idx}")

    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        if self.video_loader:
            return self.video_loader.get_frame(self.current_idx)
        return None