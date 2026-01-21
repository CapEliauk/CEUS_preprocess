"""
双区域选择器 - 支持绘制、拖拽调整矩形框
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List
from enum import Enum
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QGroupBox, QRadioButton, QButtonGroup,
                             QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor

from utils.logger import get_logger

logger = get_logger('region_selector')


class DragMode(Enum):
    """拖拽模式"""
    NONE = 0
    DRAW = 1        # 绘制新矩形
    MOVE = 2        # 移动整个矩形
    RESIZE_TL = 3   # 调整左上角
    RESIZE_TR = 4   # 调整右上角
    RESIZE_BL = 5   # 调整左下角
    RESIZE_BR = 6   # 调整右下角
    RESIZE_T = 7    # 调整上边
    RESIZE_B = 8    # 调整下边
    RESIZE_L = 9    # 调整左边
    RESIZE_R = 10   # 调整右边


class RegionDrawLabel(QLabel):
    """支持绘制和拖拽调整矩形的Label"""

    regions_changed = pyqtSignal(list)

    # 边缘检测距离
    EDGE_THRESHOLD = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.original_image: Optional[np.ndarray] = None
        self.display_scale: float = 1.0
        self.display_offset: QPoint = QPoint(0, 0)

        # 矩形列表
        self.rectangles: List[QRect] = []
        self.max_rectangles = 2

        # 拖拽状态
        self.drag_mode: DragMode = DragMode.NONE
        self.active_rect_index: int = -1
        self.drag_start_point: Optional[QPoint] = None
        self.drag_start_rect: Optional[QRect] = None

        # 当前正在绘制的矩形
        self.current_rect: Optional[QRect] = None

        # 颜色和标签（使用英文避免乱码）
        self.colors = [
            (0, 0, 255),    # 红色 (BGR)
            (255, 0, 0),    # 蓝色 (BGR)
        ]
        self.labels = ["Region 1", "Region 2"]

    def set_image(self, image: np.ndarray):
        """设置图像"""
        self.original_image = image.copy()
        self.rectangles.clear()
        self.current_rect = None
        self.drag_mode = DragMode.NONE
        self._update_display()

    def clear_rectangles(self):
        """清除所有矩形"""
        self.rectangles.clear()
        self.current_rect = None
        self.drag_mode = DragMode.NONE
        self._update_display()
        self.regions_changed.emit([])

    def _update_display(self):
        """更新显示"""
        if self.original_image is None:
            return

        display = self.original_image.copy()
        h, w = display.shape[:2]

        # 计算缩放
        lbl_size = self.size()
        self.display_scale = min(lbl_size.width() / w, lbl_size.height() / h, 1.0)
        new_w, new_h = int(w * self.display_scale), int(h * self.display_scale)

        # 计算偏移（居中）
        self.display_offset = QPoint(
            (lbl_size.width() - new_w) // 2,
            (lbl_size.height() - new_h) // 2
        )

        # 绘制已有矩形
        for i, rect in enumerate(self.rectangles):
            color = self.colors[i % len(self.colors)]
            self._draw_rectangle(display, rect, color, self.labels[i], i == self.active_rect_index)

        # 绘制当前正在画的矩形
        if self.current_rect is not None and self.drag_mode == DragMode.DRAW:
            idx = len(self.rectangles)
            color = self.colors[idx % len(self.colors)]
            self._draw_rectangle(display, self.current_rect, color, self.labels[idx], False)

        # 转换显示
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h))

        # 创建完整图像
        full_img = np.zeros((lbl_size.height(), lbl_size.width(), 3), dtype=np.uint8)
        full_img.fill(30)  # 深灰色背景

        y_start = self.display_offset.y()
        x_start = self.display_offset.x()
        full_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized

        qimg = QImage(full_img.data, lbl_size.width(), lbl_size.height(),
                     3 * lbl_size.width(), QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

    def _draw_rectangle(self, image: np.ndarray, rect: QRect,
                        color: Tuple[int, int, int], label: str, is_active: bool):
        """绘制矩形和标签"""
        thickness = 3 if is_active else 2

        # 绘制矩形
        cv2.rectangle(
            image,
            (rect.x(), rect.y()),
            (rect.x() + rect.width(), rect.y() + rect.height()),
            color, thickness
        )

        # 如果是活动矩形，绘制调整手柄
        if is_active:
            self._draw_handles(image, rect, color)

        # 绘制标签背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x = rect.x() + 5
        label_y = rect.y() + 5

        cv2.rectangle(
            image,
            (label_x - 2, label_y - 2),
            (label_x + label_size[0] + 4, label_y + label_size[1] + 6),
            color, -1
        )

        # 绘制标签文字（白色）
        cv2.putText(
            image, label,
            (label_x, label_y + label_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    def _draw_handles(self, image: np.ndarray, rect: QRect, color: Tuple[int, int, int]):
        """绘制调整手柄"""
        handle_size = 6

        # 8个手柄位置：4个角 + 4条边中点
        handles = [
            (rect.x(), rect.y()),                                          # 左上
            (rect.x() + rect.width(), rect.y()),                           # 右上
            (rect.x(), rect.y() + rect.height()),                          # 左下
            (rect.x() + rect.width(), rect.y() + rect.height()),           # 右下
            (rect.x() + rect.width() // 2, rect.y()),                      # 上中
            (rect.x() + rect.width() // 2, rect.y() + rect.height()),      # 下中
            (rect.x(), rect.y() + rect.height() // 2),                     # 左中
            (rect.x() + rect.width(), rect.y() + rect.height() // 2),      # 右中
        ]

        for hx, hy in handles:
            cv2.rectangle(
                image,
                (hx - handle_size, hy - handle_size),
                (hx + handle_size, hy + handle_size),
                (255, 255, 255), -1
            )
            cv2.rectangle(
                image,
                (hx - handle_size, hy - handle_size),
                (hx + handle_size, hy + handle_size),
                color, 2
            )

    def _display_to_image_coords(self, pos: QPoint) -> QPoint:
        """将显示坐标转换为图像坐标"""
        if self.display_scale == 0:
            return QPoint(0, 0)
        img_x = int((pos.x() - self.display_offset.x()) / self.display_scale)
        img_y = int((pos.y() - self.display_offset.y()) / self.display_scale)
        return QPoint(img_x, img_y)

    def _clamp_to_image(self, point: QPoint) -> QPoint:
        """限制点在图像范围内"""
        if self.original_image is None:
            return point
        h, w = self.original_image.shape[:2]
        return QPoint(
            max(0, min(point.x(), w - 1)),
            max(0, min(point.y(), h - 1))
        )

    def _get_drag_mode_at_point(self, img_point: QPoint, rect_index: int) -> DragMode:
        """判断点击位置的拖拽模式"""
        if rect_index < 0 or rect_index >= len(self.rectangles):
            return DragMode.NONE

        rect = self.rectangles[rect_index]
        px, py = img_point.x(), img_point.y()

        # 矩形边界
        left, top = rect.x(), rect.y()
        right, bottom = rect.x() + rect.width(), rect.y() + rect.height()
        mid_x, mid_y = (left + right) // 2, (top + bottom) // 2

        threshold = self.EDGE_THRESHOLD / self.display_scale

        # 检查4个角
        if abs(px - left) < threshold and abs(py - top) < threshold:
            return DragMode.RESIZE_TL
        if abs(px - right) < threshold and abs(py - top) < threshold:
            return DragMode.RESIZE_TR
        if abs(px - left) < threshold and abs(py - bottom) < threshold:
            return DragMode.RESIZE_BL
        if abs(px - right) < threshold and abs(py - bottom) < threshold:
            return DragMode.RESIZE_BR

        # 检查4条边
        if abs(py - top) < threshold and left < px < right:
            return DragMode.RESIZE_T
        if abs(py - bottom) < threshold and left < px < right:
            return DragMode.RESIZE_B
        if abs(px - left) < threshold and top < py < bottom:
            return DragMode.RESIZE_L
        if abs(px - right) < threshold and top < py < bottom:
            return DragMode.RESIZE_R

        # 在矩形内部 - 移动
        if left < px < right and top < py < bottom:
            return DragMode.MOVE

        return DragMode.NONE

    def _find_rect_at_point(self, img_point: QPoint) -> int:
        """查找点击位置的矩形索引"""
        px, py = img_point.x(), img_point.y()
        threshold = self.EDGE_THRESHOLD / self.display_scale

        for i, rect in enumerate(self.rectangles):
            left, top = rect.x(), rect.y()
            right, bottom = rect.x() + rect.width(), rect.y() + rect.height()

            # 扩大检测范围（包含边缘）
            if (left - threshold <= px <= right + threshold and
                top - threshold <= py <= bottom + threshold):
                return i

        return -1

    def _update_cursor(self, drag_mode: DragMode):
        """更新鼠标光标"""
        cursor_map = {
            DragMode.NONE: Qt.ArrowCursor,
            DragMode.DRAW: Qt.CrossCursor,
            DragMode.MOVE: Qt.SizeAllCursor,
            DragMode.RESIZE_TL: Qt.SizeFDiagCursor,
            DragMode.RESIZE_BR: Qt.SizeFDiagCursor,
            DragMode.RESIZE_TR: Qt.SizeBDiagCursor,
            DragMode.RESIZE_BL: Qt.SizeBDiagCursor,
            DragMode.RESIZE_T: Qt.SizeVerCursor,
            DragMode.RESIZE_B: Qt.SizeVerCursor,
            DragMode.RESIZE_L: Qt.SizeHorCursor,
            DragMode.RESIZE_R: Qt.SizeHorCursor,
        }
        self.setCursor(cursor_map.get(drag_mode, Qt.ArrowCursor))

    def mousePressEvent(self, event):
        """鼠标按下"""
        if self.original_image is None:
            return

        img_point = self._display_to_image_coords(event.pos())
        img_point = self._clamp_to_image(img_point)

        if event.button() == Qt.LeftButton:
            # 检查是否点击了现有矩形
            rect_index = self._find_rect_at_point(img_point)

            if rect_index >= 0:
                # 点击了现有矩形
                self.active_rect_index = rect_index
                self.drag_mode = self._get_drag_mode_at_point(img_point, rect_index)
                self.drag_start_point = img_point
                self.drag_start_rect = QRect(self.rectangles[rect_index])
            elif len(self.rectangles) < self.max_rectangles:
                # 绘制新矩形
                self.drag_mode = DragMode.DRAW
                self.drag_start_point = img_point
                self.current_rect = QRect(img_point, img_point)
                self.active_rect_index = -1

            self._update_display()

        elif event.button() == Qt.RightButton:
            # 右键删除
            rect_index = self._find_rect_at_point(img_point)
            if rect_index >= 0:
                self.rectangles.pop(rect_index)
                self.active_rect_index = -1
                self._update_display()
                self._emit_regions()

    def mouseMoveEvent(self, event):
        """鼠标移动"""
        if self.original_image is None:
            return

        img_point = self._display_to_image_coords(event.pos())
        img_point = self._clamp_to_image(img_point)

        if self.drag_mode == DragMode.NONE:
            # 更新光标
            rect_index = self._find_rect_at_point(img_point)
            if rect_index >= 0:
                mode = self._get_drag_mode_at_point(img_point, rect_index)
                self._update_cursor(mode)
            elif len(self.rectangles) < self.max_rectangles:
                self._update_cursor(DragMode.DRAW)
            else:
                self._update_cursor(DragMode.NONE)
            return

        if self.drag_start_point is None:
            return

        if self.drag_mode == DragMode.DRAW:
            # 绘制新矩形
            self.current_rect = QRect(self.drag_start_point, img_point).normalized()

        elif self.drag_mode == DragMode.MOVE and self.drag_start_rect is not None:
            # 移动矩形
            delta = img_point - self.drag_start_point
            new_rect = QRect(self.drag_start_rect)
            new_rect.translate(delta.x(), delta.y())

            # 限制在图像范围内
            h, w = self.original_image.shape[:2]
            if new_rect.x() < 0:
                new_rect.moveLeft(0)
            if new_rect.y() < 0:
                new_rect.moveTop(0)
            if new_rect.right() >= w:
                new_rect.moveRight(w - 1)
            if new_rect.bottom() >= h:
                new_rect.moveBottom(h - 1)

            self.rectangles[self.active_rect_index] = new_rect

        elif self.drag_start_rect is not None:
            # 调整大小
            self._resize_rect(img_point)

        self._update_display()

    def _resize_rect(self, img_point: QPoint):
        """调整矩形大小"""
        if self.drag_start_rect is None or self.active_rect_index < 0:
            return

        rect = QRect(self.drag_start_rect)
        px, py = img_point.x(), img_point.y()

        # 根据拖拽模式调整
        if self.drag_mode == DragMode.RESIZE_TL:
            rect.setTopLeft(img_point)
        elif self.drag_mode == DragMode.RESIZE_TR:
            rect.setTopRight(img_point)
        elif self.drag_mode == DragMode.RESIZE_BL:
            rect.setBottomLeft(img_point)
        elif self.drag_mode == DragMode.RESIZE_BR:
            rect.setBottomRight(img_point)
        elif self.drag_mode == DragMode.RESIZE_T:
            rect.setTop(py)
        elif self.drag_mode == DragMode.RESIZE_B:
            rect.setBottom(py)
        elif self.drag_mode == DragMode.RESIZE_L:
            rect.setLeft(px)
        elif self.drag_mode == DragMode.RESIZE_R:
            rect.setRight(px)

        # 确保矩形有最小尺寸
        rect = rect.normalized()
        if rect.width() >= 20 and rect.height() >= 20:
            self.rectangles[self.active_rect_index] = rect

    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        if event.button() == Qt.LeftButton:
            if self.drag_mode == DragMode.DRAW and self.current_rect is not None:
                # 完成绘制
                if self.current_rect.width() > 20 and self.current_rect.height() > 20:
                    self.rectangles.append(self.current_rect)
                    self.active_rect_index = len(self.rectangles) - 1
                    self._emit_regions()
                self.current_rect = None
            elif self.drag_mode != DragMode.NONE:
                # 完成拖拽/调整
                self._emit_regions()

            self.drag_mode = DragMode.NONE
            self.drag_start_point = None
            self.drag_start_rect = None
            self._update_display()

    def _emit_regions(self):
        """发送区域信号"""
        regions = [(r.x(), r.y(), r.width(), r.height()) for r in self.rectangles]
        self.regions_changed.emit(regions)

    def get_rectangles(self) -> List[Tuple[int, int, int, int]]:
        """获取矩形列表"""
        return [(r.x(), r.y(), r.width(), r.height()) for r in self.rectangles]


class DualRegionSelector(QWidget):
    """双区域选择器"""

    regions_confirmed = pyqtSignal(tuple, tuple)  # bmode_rect, ceus_rect

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[np.ndarray] = None
        self.regions: List[Tuple[int, int, int, int]] = []

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 提示
        tip_label = QLabel(
            "操作说明:\n"
            "• 左键拖动: 绘制新矩形 (最多2个)\n"
            "• 点击矩形: 选中后可拖动调整位置和大小\n"
            "• 拖动边角/边缘: 调整矩形大小\n"
            "• 右键点击: 删除矩形"
        )
        tip_label.setStyleSheet(
            "color: #333; padding: 10px; background-color: #f5f5f5; "
            "border-radius: 5px; font-size: 12px;"
        )
        tip_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(tip_label)

        # 图像显示
        self.image_label = RegionDrawLabel()
        self.image_label.setMinimumSize(700, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; border-radius: 8px;")
        self.image_label.regions_changed.connect(self._on_regions_changed)
        layout.addWidget(self.image_label)

        # 区域分配
        assign_group = QGroupBox("Region Assignment")
        assign_layout = QVBoxLayout(assign_group)

        # 区域1
        region1_layout = QHBoxLayout()
        self.region1_label = QLabel("Region 1 (Red):")
        self.region1_label.setMinimumWidth(120)
        region1_layout.addWidget(self.region1_label)

        self.region1_bmode = QRadioButton("B-mode")
        self.region1_ceus = QRadioButton("CEUS")
        self.region1_group = QButtonGroup()
        self.region1_group.addButton(self.region1_bmode, 0)
        self.region1_group.addButton(self.region1_ceus, 1)
        self.region1_bmode.setChecked(True)
        region1_layout.addWidget(self.region1_bmode)
        region1_layout.addWidget(self.region1_ceus)
        region1_layout.addStretch()
        assign_layout.addLayout(region1_layout)

        # 区域2
        region2_layout = QHBoxLayout()
        self.region2_label = QLabel("Region 2 (Blue):")
        self.region2_label.setMinimumWidth(120)
        region2_layout.addWidget(self.region2_label)

        self.region2_bmode = QRadioButton("B-mode")
        self.region2_ceus = QRadioButton("CEUS")
        self.region2_group = QButtonGroup()
        self.region2_group.addButton(self.region2_bmode, 0)
        self.region2_group.addButton(self.region2_ceus, 1)
        self.region2_ceus.setChecked(True)
        region2_layout.addWidget(self.region2_bmode)
        region2_layout.addWidget(self.region2_ceus)
        region2_layout.addStretch()
        assign_layout.addLayout(region2_layout)

        # 互斥逻辑
        self.region1_bmode.toggled.connect(self._on_region1_changed)
        self.region2_bmode.toggled.connect(self._on_region2_changed)

        layout.addWidget(assign_group)

        # 按钮
        btn_layout = QHBoxLayout()

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton { background-color: #f44336; color: white; padding: 10px 25px; 
                         border-radius: 5px; font-size: 13px; }
            QPushButton:hover { background-color: #d32f2f; }
        """)
        self.clear_btn.clicked.connect(self._clear)
        btn_layout.addWidget(self.clear_btn)

        btn_layout.addStretch()

        self.confirm_btn = QPushButton("Confirm Regions")
        self.confirm_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-size: 14px;
                         font-weight: bold; padding: 12px 35px; border-radius: 6px; }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.confirm_btn.clicked.connect(self._confirm)
        self.confirm_btn.setEnabled(False)
        btn_layout.addWidget(self.confirm_btn)

        layout.addLayout(btn_layout)

        # 状态信息
        self.status_label = QLabel("Please draw 2 rectangles for B-mode and CEUS regions")
        self.status_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def set_frame(self, frame: np.ndarray):
        """设置帧"""
        self.frame = frame.copy()
        self.image_label.set_image(frame)
        self.regions = []
        self.confirm_btn.setEnabled(False)
        self.status_label.setText("Please draw 2 rectangles for B-mode and CEUS regions")
        self._update_region_labels()

    def _on_regions_changed(self, regions: List[Tuple[int, int, int, int]]):
        """区域变化"""
        self.regions = regions
        self._update_region_labels()

        if len(regions) == 0:
            self.status_label.setText("Please draw 2 rectangles")
            self.confirm_btn.setEnabled(False)
        elif len(regions) == 1:
            self.status_label.setText("1 region drawn. Please draw 1 more.")
            self.confirm_btn.setEnabled(False)
        else:
            self.status_label.setText("2 regions ready. Please assign B-mode and CEUS, then confirm.")
            self.confirm_btn.setEnabled(True)

    def _update_region_labels(self):
        """更新区域标签显示"""
        if len(self.regions) >= 1:
            r = self.regions[0]
            self.region1_label.setText(f"Region 1 (Red): {r[2]}x{r[3]}")
        else:
            self.region1_label.setText("Region 1 (Red): --")

        if len(self.regions) >= 2:
            r = self.regions[1]
            self.region2_label.setText(f"Region 2 (Blue): {r[2]}x{r[3]}")
        else:
            self.region2_label.setText("Region 2 (Blue): --")

    def _on_region1_changed(self, checked):
        """区域1选择变化"""
        if checked:
            self.region2_ceus.setChecked(True)
        else:
            self.region2_bmode.setChecked(True)

    def _on_region2_changed(self, checked):
        """区域2选择变化"""
        if checked:
            self.region1_ceus.setChecked(True)
        else:
            self.region1_bmode.setChecked(True)

    def _clear(self):
        """清除"""
        self.image_label.clear_rectangles()
        self.regions = []
        self.confirm_btn.setEnabled(False)
        self._update_region_labels()

    def _confirm(self):
        """确认"""
        if len(self.regions) != 2:
            QMessageBox.warning(self, "Warning", "Please draw exactly 2 rectangles")
            return

        # 确定B-mode和CEUS
        if self.region1_bmode.isChecked():
            bmode_rect = self.regions[0]
            ceus_rect = self.regions[1]
        else:
            bmode_rect = self.regions[1]
            ceus_rect = self.regions[0]

        self.regions_confirmed.emit(bmode_rect, ceus_rect)

        logger.info(f"Regions confirmed: B-mode={bmode_rect}, CEUS={ceus_rect}")

    def get_bmode_region(self) -> Optional[np.ndarray]:
        """获取B-mode区域图像"""
        if self.frame is None or len(self.regions) != 2:
            return None

        if self.region1_bmode.isChecked():
            rect = self.regions[0]
        else:
            rect = self.regions[1]

        x, y, w, h = rect
        return self.frame[y:y+h, x:x+w].copy()