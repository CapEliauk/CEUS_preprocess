"""
配置文件 - 包含所有可配置参数和算法参数
"""
import os
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

@dataclass
class VideoConfig:
    """视频处理参数"""
    CLIP_SIZE: Tuple[int, int] = (224, 224)
    CLIP_LENGTH: int = 16
    SLIDING_WINDOW_STRIDE: int = 8
    DEFAULT_FPS: float = 25.0                                                     # 标准视频帧率（默认OpenCV读取）
    DEFAULT_FPS_DICOM: float = 25.0                                               # dicom帧率（默认pydicom读取）
    OUTPUT_FORMAT: str = '.mp4'
    OUTPUT_FPS: int = 30
    VIDEO_EXTENSIONS: Tuple[str, ...] = ('.avi', '.mp4', '.mkv', '.mov', '.wmv')
    DICOM_EXTENSIONS: Tuple[str, ...] = ('.dcm', '.dicom', '')


@dataclass
class TICConfig:
    """TIC分析参数"""
    # 到达时间检测
    ARRIVAL_THRESHOLD_RATIO: float = 0.1
    BASELINE_RATIO: float = 0.1  # 用于计算基线的帧比例

    # 平滑参数
    SMOOTH_WINDOW_SIZE: int = 11
    SMOOTH_POLY_ORDER: int = 3

    # Gamma拟合参数边界
    GAMMA_A_BOUNDS: Tuple[float, float] = (0.0, 1e6)      # 幅度
    GAMMA_ALPHA_BOUNDS: Tuple[float, float] = (0.1, 50.0) # 形状参数
    GAMMA_BETA_BOUNDS: Tuple[float, float] = (0.01, 10.0) # 尺度参数
    GAMMA_T0_BOUNDS: Tuple[float, float] = (0.0, 60.0)    # 起始时间
    GAMMA_C_BOUNDS: Tuple[float, float] = (0.0, 1e4)      # 基线偏移

    # 拟合初始猜测
    GAMMA_INITIAL_GUESS: Tuple[float, ...] = (1000.0, 2.0, 1.0, 5.0, 100.0)

    # 相位时间区间（秒）
    ARTERIAL_DURATION: float = 30.0
    PORTAL_DURATION: float = 60.0
    DELAY_DURATION: float = 90.0


@dataclass
class FilterConfig:
    """帧筛选参数"""
    SSIM_THRESHOLD: float = 0.7
    MOTION_THRESHOLD_SIGMA: float = 2.0  # 运动阈值 = mean + sigma * std
    MIN_VALID_FRAME_RATIO: float = 0.1   # 最少保留帧比例

    # 清晰度检测
    LAPLACIAN_THRESHOLD: float = 100.0

    # 快速筛选
    USE_FAST_FILTER: bool = True


@dataclass
class RegistrationConfig:
    """运动补偿/配准参数"""
    ENABLE_REGISTRATION: bool = True
    MAX_SHIFT_PIXELS: int = 50           # 最大位移像素
    TEMPLATE_UPDATE_INTERVAL: int = 10   # 模板更新间隔

    # 相位相关参数
    PHASE_CORRELATION_WINDOW: str = 'hann'

    # 特征点匹配参数
    FEATURE_METHOD: str = 'orb'  # 'orb', 'sift', 'akaze'
    MIN_MATCH_COUNT: int = 10
    RANSAC_THRESHOLD: float = 5.0


@dataclass
class MemoryConfig:
    """内存管理参数"""
    MAX_MEMORY_PERCENT: float = 70.0
    BATCH_SIZE: int = 50
    FRAME_BUFFER_SIZE: int = 100
    USE_MEMMAP: bool = True
    MEMMAP_THRESHOLD_MB: int = 500  # 超过此大小使用memmap
    MAX_WORKERS: int = 2
    WAIT_MEMORY_TIMEOUT: float = 30.0
    MONITOR_INTERNAL: float = 1.0
    TEMP_DIR: str = os.path.join(PROJECT_ROOT, 'temp')

@dataclass
class LogConfig:
    """日志配置"""
    LOG_DIR: str = os.path.join(PROJECT_ROOT, 'logs')
    LOG_LEVEL: str = 'DEBUG'
    CONSOLE_LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    MAX_LOG_FILES: int = 10
    MAX_LOG_SIZE_MB: int = 10


@dataclass
class Config:
    """主配置类"""
    video: VideoConfig = field(default_factory=VideoConfig)
    tic: TICConfig = field(default_factory=TICConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # 临时目录
    TEMP_DIR: str = field(default_factory=lambda: os.path.join(
        os.environ.get('TEMP', os.environ.get('TMP', '/tmp')),
        'ceus_temp'
    ))

    def __post_init__(self):
        # 确保目录存在
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.log.LOG_DIR, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'video': self.video.__dict__,
            'tic': self.tic.__dict__,
            'filter': self.filter.__dict__,
            'registration': self.registration.__dict__,
            'memory': self.memory.__dict__,
        }


# 全局配置实例
config = Config()