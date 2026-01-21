"""
ROI标注器 - 在B-mode区域上进行标注
"""
import json
import os
import subprocess
import shutil
import re
import time
import numpy as np
import cv2
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QGroupBox,
                             QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

from config import config
from core.roi_manager import DualRegionROIManager, ROIData
from utils.file_utils import cv2_imwrite
from utils.logger import get_logger

logger = get_logger('roi_annotator')


class ShapeConverter:
    """形状转换器"""

    @staticmethod
    def shape_to_mask(shape_type: str, points: List[List[float]],
                      img_shape: Tuple[int, int], **kwargs) -> np.ndarray:
        mask = np.zeros(img_shape, dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)

        shape_type = shape_type.lower()

        if shape_type == 'polygon':
            cv2.fillPoly(mask, [pts], 255)
        elif shape_type == 'rectangle' and len(pts) >= 2:
            cv2.rectangle(mask, tuple(pts[0]), tuple(pts[1]), 255, -1)
        elif shape_type == 'circle' and len(pts) >= 2:
            center = tuple(pts[0])
            radius = int(np.sqrt((pts[1][0]-pts[0][0])**2 + (pts[1][1]-pts[0][1])**2))
            cv2.circle(mask, center, radius, 255, -1)
        elif shape_type == 'ellipse' and len(pts) >= 2:
            center = tuple(pts[0])
            axes = (abs(pts[1][0]-pts[0][0]), abs(pts[1][1]-pts[0][1]))
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif shape_type == 'line' and len(pts) >= 2:
            thickness = kwargs.get('thickness', 10)
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), 255, thickness)
        elif shape_type == 'linestrip' and len(pts) >= 2:
            thickness = kwargs.get('thickness', 10)
            cv2.polylines(mask, [pts], False, 255, thickness)
        elif shape_type == 'point':
            radius = kwargs.get('radius', 15)
            for pt in pts:
                cv2.circle(mask, tuple(pt), radius, 255, -1)
        elif len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)

        return mask


class BmodeROIAnnotator(QWidget):
    """B-mode区域ROI标注器"""

    # 信号：发送B-mode mask和相关数据
    roi_annotated = pyqtSignal(np.ndarray, list, str)  # bmode_mask, points, shape_type
    annotation_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.bmode_image: Optional[np.ndarray] = None
        self.bmode_rect: Optional[Tuple[int, int, int, int]] = None
        self.mask: Optional[np.ndarray] = None
        self.points: List[List[float]] = []
        self.shape_type: str = "polygon"

        self.json_path: Optional[str] = None
        self.temp_image_path: Optional[str] = None
        self.video_name: str = ""

        self.temp_dir = self._get_safe_temp_dir()

        self._setup_ui()

    def _get_safe_temp_dir(self) -> str:
        """获取安全临时目录"""
        import tempfile
        candidates = [
            tempfile.gettempdir(),
            os.environ.get('TEMP'),
            os.environ.get('TMP'),
            'C:\\Temp' if os.name == 'nt' else '/tmp',
        ]

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    path.encode('ascii')
                    temp_dir = os.path.join(path, 'ceus_temp')
                    os.makedirs(temp_dir, exist_ok=True)
                    return temp_dir
                except (UnicodeEncodeError, UnicodeDecodeError):
                    continue

        return tempfile.gettempdir()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = os.path.basename(str(name))
        name = os.path.splitext(name)[0]
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name[:30] if name else "frame"

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 标题
        title = QLabel("在B-mode区域上标注ROI")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 图像显示
        self.image_label = QLabel()
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border-radius: 8px;")
        layout.addWidget(self.image_label)

        # 提示
        self.info_label = QLabel("请点击'启动LabelMe'在B-mode区域上标注ROI")
        self.info_label.setStyleSheet("font-size: 14px; color: #666; padding: 10px;")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        # 按钮
        btn_group = QGroupBox("标注操作")
        btn_layout = QHBoxLayout(btn_group)

        self.labelme_btn = QPushButton("🏷️ 启动LabelMe")
        self.labelme_btn.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-size: 14px; 
                         font-weight: bold; padding: 12px 24px; border-radius: 6px; }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.labelme_btn.clicked.connect(self._launch_labelme)
        btn_layout.addWidget(self.labelme_btn)

        self.load_btn = QPushButton("📂 加载标注")
        self.load_btn.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; font-size: 14px;
                         padding: 12px 24px; border-radius: 6px; }
            QPushButton:hover { background-color: #F57C00; }
        """)
        self.load_btn.clicked.connect(self._load_annotation)
        btn_layout.addWidget(self.load_btn)

        self.refresh_btn = QPushButton("🔄 刷新")
        self.refresh_btn.clicked.connect(self._refresh)
        self.refresh_btn.setEnabled(False)
        btn_layout.addWidget(self.refresh_btn)

        layout.addWidget(btn_group)

        # 选项
        options_group = QGroupBox("选项")
        options_layout = QHBoxLayout(options_group)

        options_layout.addWidget(QLabel("线条宽度:"))
        self.thickness_combo = QComboBox()
        self.thickness_combo.addItems(['5', '10', '15', '20', '30'])
        self.thickness_combo.setCurrentText('10')
        options_layout.addWidget(self.thickness_combo)

        options_layout.addWidget(QLabel("点半径:"))
        self.radius_combo = QComboBox()
        self.radius_combo.addItems(['5', '10', '15', '20', '30'])
        self.radius_combo.setCurrentText('15')
        options_layout.addWidget(self.radius_combo)

        options_layout.addStretch()
        layout.addWidget(options_group)

        # 确认按钮
        action_layout = QHBoxLayout()

        self.skip_btn = QPushButton("⏭️ 跳过")
        self.skip_btn.setStyleSheet("""
            QPushButton { background-color: #757575; color: white; padding: 12px 30px; border-radius: 6px; }
        """)
        self.skip_btn.clicked.connect(self._skip)
        action_layout.addWidget(self.skip_btn)

        action_layout.addStretch()

        self.confirm_btn = QPushButton("✓ 确认ROI")
        self.confirm_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-size: 16px;
                         font-weight: bold; padding: 15px 40px; border-radius: 8px; }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.confirm_btn.clicked.connect(self._confirm)
        self.confirm_btn.setEnabled(False)
        action_layout.addWidget(self.confirm_btn)

        layout.addLayout(action_layout)

        # ROI信息
        self.roi_info = QLabel("")
        self.roi_info.setStyleSheet("font-size: 12px; color: #4CAF50; padding: 5px;")
        self.roi_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.roi_info)

    def set_bmode_image(self, bmode_image: np.ndarray, bmode_rect: Tuple[int, int, int, int],
                        video_name: str = ""):
        """设置B-mode区域图像"""
        self.bmode_image = bmode_image.copy()
        self.bmode_rect = bmode_rect
        self.video_name = video_name
        self.mask = None
        self.points = []
        self.json_path = None

        self._display_image(bmode_image)
        self._update_ui()

        self.info_label.setText("请点击'启动LabelMe'在B-mode区域上标注ROI")
        self.roi_info.setText(f"B-mode区域: {bmode_rect}")

        logger.debug(f"设置B-mode图像: {bmode_image.shape}, rect={bmode_rect}")

    def _display_image(self, image: np.ndarray, show_mask: bool = True):
        """显示图像"""
        display = image.copy()

        if show_mask and self.mask is not None:
            overlay = display.copy()
            overlay[self.mask > 0] = [0, 255, 0]
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        lbl_size = self.image_label.size()
        scale = min(lbl_size.width() / w, lbl_size.height() / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))
        qimg = QImage(resized.data, new_w, new_h, c * new_w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def _update_ui(self):
        has_mask = self.mask is not None
        self.confirm_btn.setEnabled(has_mask)
        self.refresh_btn.setEnabled(self.json_path is not None)

    def _launch_labelme(self):
        """启动LabelMe"""
        if self.bmode_image is None:
            QMessageBox.warning(self, "警告", "请先设置B-mode图像")
            return

        os.makedirs(self.temp_dir, exist_ok=True)

        timestamp = int(time.time() * 1000) % 1000000
        safe_name = self._sanitize_filename(self.video_name)

        filename = f"{safe_name}_bmode_{timestamp}.png"
        json_filename = f"{safe_name}_bmode_{timestamp}.json"

        self.temp_image_path = os.path.join(self.temp_dir, filename)
        self.json_path = os.path.join(self.temp_dir, json_filename)

        try:
            success = cv2_imwrite(self.temp_image_path, self.bmode_image)
            if not success:
                raise IOError("图像保存失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图像失败:\n{e}")
            return

        self.info_label.setText("正在启动LabelMe...")

        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            cmd = ["labelme", self.temp_image_path, "--output", self.json_path]
            subprocess.run(cmd, capture_output=True, text=True)
            QTimer.singleShot(300, self._try_load)
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", "未找到LabelMe！\n\n请安装: pip install labelme")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败:\n{e}")

    def _try_load(self):
        """尝试加载标注"""
        if self.json_path and os.path.exists(self.json_path):
            self._load_json(self.json_path)
        else:
            base = os.path.splitext(self.temp_image_path)[0] if self.temp_image_path else None
            if base and os.path.exists(base + '.json'):
                self._load_json(base + '.json')
            else:
                self.info_label.setText("未检测到标注，请重试")
                self.info_label.setStyleSheet("font-size: 14px; color: #FF9800; padding: 10px;")

    def _load_annotation(self):
        """加载标注文件"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择标注", "", "JSON (*.json)")
        if filepath:
            self.json_path = filepath
            self._load_json(filepath)

    def _refresh(self):
        """刷新"""
        if self.json_path and os.path.exists(self.json_path):
            self._load_json(self.json_path)

    def _load_json(self, path: str):
        """加载JSON"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            shapes = data.get('shapes', [])
            if not shapes:
                QMessageBox.warning(self, "警告", "没有找到标注")
                return

            h, w = self.bmode_image.shape[:2]
            thickness = int(self.thickness_combo.currentText())
            radius = int(self.radius_combo.currentText())

            # 使用第一个形状
            first_shape = shapes[0]
            self.shape_type = first_shape.get('shape_type', 'polygon')
            self.points = first_shape.get('points', [])

            # 创建mask
            masks = []
            for shape in shapes:
                mask = ShapeConverter.shape_to_mask(
                    shape.get('shape_type', 'polygon'),
                    shape.get('points', []),
                    (h, w),
                    thickness=thickness, radius=radius
                )
                if np.any(mask > 0):
                    masks.append(mask)

            if not masks:
                QMessageBox.warning(self, "警告", "无法生成有效ROI")
                return

            # 合并masks
            self.mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                self.mask = cv2.bitwise_or(self.mask, m)

            self._display_image(self.bmode_image)
            self._update_ui()

            roi_area = np.sum(self.mask > 0)
            self.info_label.setText("✓ 标注加载成功!")
            self.info_label.setStyleSheet("font-size: 14px; color: #4CAF50; padding: 10px;")
            self.roi_info.setText(f"B-mode ROI: 面积={roi_area:,}px, 形状={self.shape_type}")

            logger.info(f"加载B-mode ROI: {len(masks)}个形状")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{e}")
            logger.error(f"加载JSON失败: {e}")

    def _confirm(self):
        """确认ROI"""
        if self.mask is None:
            QMessageBox.warning(self, "警告", "请先完成标注")
            return
        self.roi_annotated.emit(self.mask, self.points, self.shape_type)
        logger.info("确认B-mode ROI")

    def _skip(self):
        self.annotation_cancelled.emit()

    def get_mask(self) -> Optional[np.ndarray]:
        return self.mask

    def get_points(self) -> List[List[float]]:
        return self.points

    def cleanup(self):
        """清理"""
        try:
            if self.temp_image_path and os.path.exists(self.temp_image_path):
                os.remove(self.temp_image_path)
            if self.json_path and os.path.exists(self.json_path):
                os.remove(self.json_path)
        except:
            pass