"""
ROI管理模块 - 支持双区域（B-mode/CEUS）和坐标映射
"""
import json
import os
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from config import config
from utils.logger import get_logger

logger = get_logger('roi_manager')


@dataclass
class RegionRect:
    """区域矩形"""
    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]) -> 'RegionRect':
        return cls(t[0], t[1], t[2], t[3])

    def crop(self, image: np.ndarray) -> np.ndarray:
        """裁剪图像区域"""
        h, w = image.shape[:2]
        x1 = max(0, min(self.x, w - 1))
        y1 = max(0, min(self.y, h - 1))
        x2 = max(x1 + 1, min(self.x + self.width, w))
        y2 = max(y1 + 1, min(self.y + self.height, h))
        return image[y1:y2, x1:x2].copy()

    def contains_point(self, px: int, py: int) -> bool:
        """检查点是否在区域内"""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


@dataclass
class DualRegionConfig:
    """双区域配置"""
    bmode_rect: Optional[RegionRect] = None
    ceus_rect: Optional[RegionRect] = None

    def is_valid(self) -> bool:
        return self.bmode_rect is not None and self.ceus_rect is not None

    def to_dict(self) -> Dict:
        return {
            'bmode_rect': asdict(self.bmode_rect) if self.bmode_rect else None,
            'ceus_rect': asdict(self.ceus_rect) if self.ceus_rect else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DualRegionConfig':
        bmode = RegionRect(**data['bmode_rect']) if data.get('bmode_rect') else None
        ceus = RegionRect(**data['ceus_rect']) if data.get('ceus_rect') else None
        return cls(bmode_rect=bmode, ceus_rect=ceus)


@dataclass
class ROIData:
    """ROI数据结构"""
    points: List[List[float]]
    shape_type: str
    bbox: Tuple[int, int, int, int]
    mask_shape: Tuple[int, int]
    label: str = "ROI"
    created_time: str = ""
    video_path: str = ""
    frame_index: int = 0
    # 双区域配置
    dual_region: Optional[DualRegionConfig] = None
    # 标记这是B-mode上的ROI还是映射后的CEUS ROI
    region_type: str = "bmode"  # "bmode" or "ceus"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.dual_region:
            d['dual_region'] = self.dual_region.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'ROIData':
        dual_region = None
        if data.get('dual_region'):
            dual_region = DualRegionConfig.from_dict(data['dual_region'])
        data['dual_region'] = dual_region
        return cls(**data)


class ROICoordinateMapper:
    """ROI坐标映射器 - 在B-mode和CEUS之间映射"""

    def __init__(self, bmode_rect: RegionRect, ceus_rect: RegionRect):
        self.bmode_rect = bmode_rect
        self.ceus_rect = ceus_rect

    def bmode_to_ceus_point(self, bmode_x: float, bmode_y: float) -> Tuple[float, float]:
        """
        将B-mode上的点坐标映射到CEUS
        使用归一化计算
        """
        # 归一化到[0, 1]
        norm_x = (bmode_x - self.bmode_rect.x) / self.bmode_rect.width
        norm_y = (bmode_y - self.bmode_rect.y) / self.bmode_rect.height

        # 映射到CEUS坐标
        ceus_x = self.ceus_rect.x + norm_x * self.ceus_rect.width
        ceus_y = self.ceus_rect.y + norm_y * self.ceus_rect.height

        return (ceus_x, ceus_y)

    def bmode_to_ceus_points(self, points: List[List[float]]) -> List[List[float]]:
        """批量映射点"""
        return [list(self.bmode_to_ceus_point(p[0], p[1])) for p in points]

    def bmode_to_ceus_mask(self, bmode_mask: np.ndarray,
                           full_frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        将B-mode区域的mask映射到完整帧上的CEUS区域

        Args:
            bmode_mask: B-mode区域内的mask (相对于B-mode区域)
            full_frame_shape: 完整帧的尺寸 (H, W)

        Returns:
            完整帧上的mask，ROI位于CEUS区域
        """
        # 创建完整帧大小的空mask
        full_mask = np.zeros(full_frame_shape, dtype=np.uint8)

        # 缩放mask到CEUS区域大小
        scaled_mask = cv2.resize(
            bmode_mask,
            (self.ceus_rect.width, self.ceus_rect.height),
            interpolation=cv2.INTER_NEAREST
        )

        # 放置到CEUS区域位置
        y1, y2 = self.ceus_rect.y, self.ceus_rect.y + self.ceus_rect.height
        x1, x2 = self.ceus_rect.x, self.ceus_rect.x + self.ceus_rect.width

        # 确保不越界
        y2 = min(y2, full_frame_shape[0])
        x2 = min(x2, full_frame_shape[1])

        h = y2 - y1
        w = x2 - x1

        full_mask[y1:y2, x1:x2] = scaled_mask[:h, :w]

        return full_mask

    def get_bmode_local_mask(self, bmode_mask: np.ndarray) -> np.ndarray:
        """获取B-mode区域内的局部mask（用于SSIM计算）"""
        return bmode_mask

    def ceus_to_bmode_point(self, ceus_x: float, ceus_y: float) -> Tuple[float, float]:
        """反向映射：CEUS到B-mode"""
        norm_x = (ceus_x - self.ceus_rect.x) / self.ceus_rect.width
        norm_y = (ceus_y - self.ceus_rect.y) / self.ceus_rect.height

        bmode_x = self.bmode_rect.x + norm_x * self.bmode_rect.width
        bmode_y = self.bmode_rect.y + norm_y * self.bmode_rect.height

        return (bmode_x, bmode_y)


class ROIStorage:
    """ROI存储管理"""

    @staticmethod
    def save_roi(roi_data: ROIData, filepath: str):
        """保存ROI"""
        data = roi_data.to_dict()
        data['created_time'] = datetime.now().isoformat()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"ROI已保存: {filepath}")

    @staticmethod
    def load_roi(filepath: str) -> Optional[ROIData]:
        """加载ROI"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            roi = ROIData.from_dict(data)
            logger.info(f"ROI已加载: {filepath}")
            return roi
        except Exception as e:
            logger.error(f"加载ROI失败: {e}")
            return None

    @staticmethod
    def roi_to_mask(roi_data: ROIData) -> np.ndarray:
        """将ROI转为mask"""
        h, w = roi_data.mask_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        pts = np.array(roi_data.points, dtype=np.int32)

        if roi_data.shape_type == 'polygon':
            cv2.fillPoly(mask, [pts], 255)
        elif roi_data.shape_type == 'rectangle':
            if len(pts) >= 2:
                cv2.rectangle(mask, tuple(pts[0]), tuple(pts[1]), 255, -1)
        elif roi_data.shape_type == 'circle':
            if len(pts) >= 2:
                center = tuple(pts[0])
                radius = int(np.linalg.norm(pts[1] - pts[0]))
                cv2.circle(mask, center, radius, 255, -1)
        elif roi_data.shape_type == 'ellipse':
            if len(pts) >= 2:
                center = tuple(pts[0])
                axes = (abs(pts[1][0] - pts[0][0]), abs(pts[1][1] - pts[0][1]))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)

        return mask


class DualRegionROIManager:
    """双区域ROI管理器"""

    def __init__(self):
        self.dual_region: Optional[DualRegionConfig] = None
        self.bmode_roi_data: Optional[ROIData] = None
        self.bmode_mask: Optional[np.ndarray] = None      # B-mode区域内的mask
        self.ceus_mask: Optional[np.ndarray] = None       # 完整帧上的CEUS区域mask
        self.mapper: Optional[ROICoordinateMapper] = None
        self.frame_shape: Optional[Tuple[int, int]] = None

    def set_dual_regions(self, bmode_rect: Tuple[int, int, int, int],
                         ceus_rect: Tuple[int, int, int, int],
                         frame_shape: Tuple[int, int]):
        """设置双区域"""
        self.dual_region = DualRegionConfig(
            bmode_rect=RegionRect.from_tuple(bmode_rect),
            ceus_rect=RegionRect.from_tuple(ceus_rect)
        )
        self.frame_shape = frame_shape
        self.mapper = ROICoordinateMapper(
            self.dual_region.bmode_rect,
            self.dual_region.ceus_rect
        )
        logger.info(f"设置双区域: B-mode={bmode_rect}, CEUS={ceus_rect}")

    def set_bmode_roi(self, points: List[List[float]], shape_type: str,
                      bmode_local_mask: np.ndarray):
        """
        设置B-mode区域上的ROI

        Args:
            points: 相对于B-mode区域的点坐标
            shape_type: 形状类型
            bmode_local_mask: B-mode区域内的mask
        """
        if self.dual_region is None or self.mapper is None:
            raise ValueError("请先设置双区域")

        bmode_rect = self.dual_region.bmode_rect

        # 保存B-mode mask（用于SSIM计算）
        self.bmode_mask = bmode_local_mask.copy()

        # 计算bbox（相对于B-mode区域）
        pts = np.array(points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        self.bmode_roi_data = ROIData(
            points=points,
            shape_type=shape_type,
            bbox=(x, y, w, h),
            mask_shape=(bmode_rect.height, bmode_rect.width),
            dual_region=self.dual_region,
            region_type="bmode"
        )

        # 映射到CEUS区域
        self._create_ceus_mask()

        logger.info(f"设置B-mode ROI: {shape_type}, bbox={self.bmode_roi_data.bbox}")

    def _create_ceus_mask(self):
        """创建CEUS区域的mask"""
        if self.bmode_mask is None or self.mapper is None or self.frame_shape is None:
            return

        self.ceus_mask = self.mapper.bmode_to_ceus_mask(
            self.bmode_mask, self.frame_shape
        )

        logger.debug(f"创建CEUS mask: shape={self.ceus_mask.shape}")

    def get_bmode_region(self, frame: np.ndarray) -> np.ndarray:
        """获取B-mode区域"""
        if self.dual_region is None:
            return frame
        return self.dual_region.bmode_rect.crop(frame)

    def get_ceus_region(self, frame: np.ndarray) -> np.ndarray:
        """获取CEUS区域"""
        if self.dual_region is None:
            return frame
        return self.dual_region.ceus_rect.crop(frame)

    def get_bmode_mask_for_ssim(self) -> Optional[np.ndarray]:
        """获取用于SSIM计算的B-mode mask"""
        return self.bmode_mask

    def get_ceus_mask_for_tic(self) -> Optional[np.ndarray]:
        """获取用于TIC计算的CEUS mask（完整帧坐标）"""
        return self.ceus_mask

    def get_ceus_local_mask(self) -> Optional[np.ndarray]:
        """获取CEUS区域内的局部mask"""
        if self.ceus_mask is None or self.dual_region is None:
            return None

        ceus_rect = self.dual_region.ceus_rect
        return self.ceus_mask[
            ceus_rect.y:ceus_rect.y+ceus_rect.height,
            ceus_rect.x:ceus_rect.x+ceus_rect.width
        ]

    def save(self, filepath: str):
        """保存配置"""
        if self.bmode_roi_data:
            ROIStorage.save_roi(self.bmode_roi_data, filepath)

    def load(self, filepath: str) -> bool:
        """加载配置"""
        roi_data = ROIStorage.load_roi(filepath)
        if roi_data and roi_data.dual_region:
            self.dual_region = roi_data.dual_region
            self.bmode_roi_data = roi_data

            if self.dual_region.is_valid():
                self.mapper = ROICoordinateMapper(
                    self.dual_region.bmode_rect,
                    self.dual_region.ceus_rect
                )

            # 重建mask
            self.bmode_mask = ROIStorage.roi_to_mask(roi_data)
            if self.frame_shape:
                self._create_ceus_mask()

            return True
        return False

    def is_ready(self) -> bool:
        """检查是否准备就绪"""
        return (self.dual_region is not None and
                self.dual_region.is_valid() and
                self.bmode_mask is not None)