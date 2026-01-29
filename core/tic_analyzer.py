"""
TIC曲线分析模块 - 向量化操作 + 鲁棒拟合模型
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import warnings
from dataclasses import dataclass

from config import config
from utils.logger import get_logger

logger = get_logger('tic_analyzer')


@dataclass
class TICFitResult:
    """TIC拟合结果"""
    params: np.ndarray
    params_cov: Optional[np.ndarray]
    arrival_time: float
    peak_time: float
    peak_intensity: float
    wash_in_rate: float
    wash_out_rate: float
    auc: float
    fit_quality: float  # R²


class TICModels:
    """TIC拟合模型集合"""

    @staticmethod
    def gamma_variate(t: np.ndarray, A: float, alpha: float,
                      beta: float, t0: float, C: float) -> np.ndarray:
        """
        Gamma变异函数
        I(t) = A * (t - t0)^alpha * exp(-(t - t0) / beta) + C, for t > t0
        """
        result = np.zeros_like(t, dtype=np.float64)
        mask = t > t0
        if np.any(mask):
            dt = t[mask] - t0
            result[mask] = A * np.power(dt, alpha) * np.exp(-dt / beta) + C
        result[~mask] = C
        return result

    @staticmethod
    def lognormal(t: np.ndarray, A: float, mu: float,
                  sigma: float, t0: float, C: float) -> np.ndarray:
        """对数正态模型"""
        result = np.zeros_like(t, dtype=np.float64)
        mask = t > t0
        if np.any(mask):
            dt = t[mask] - t0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result[mask] = A * np.exp(-0.5 * ((np.log(dt) - mu) / sigma) ** 2) / (dt * sigma * np.sqrt(2 * np.pi)) + C
        result[~mask] = C
        return result


class TICAnalyzer:
    """TIC分析器 - 向量化版本"""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.tic_data: Optional[np.ndarray] = None
        self.time_axis: Optional[np.ndarray] = None
        self.smoothed_tic: Optional[np.ndarray] = None
        self.fit_result: Optional[TICFitResult] = None

        # 缓存
        self._phase_intervals: Optional[Dict] = None
        self._arrival_time: Optional[float] = None
        self._peak_time: Optional[float] = None

        logger.debug(f"TIC分析器初始化，fps={fps}")


    def extract_roi_intensities_batch(self, frames: list, mask: np.ndarray) -> np.ndarray:
        """
        批量提取ROI强度
        """
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return np.zeros(len(frames))

        intensities = []

        # [修改处] 改为循环逐帧处理，不要用 np.stack(frames)
        for frame in frames:
            # 只提取 ROI 区域的像素，大幅减少内存
            roi_pixels = frame[mask_bool]

            if roi_pixels.size == 0:
                intensities.append(0.0)
                continue

            # 如果是彩色 (N, 3)，手动计算灰度均值，避免全图转 float32
            if roi_pixels.ndim == 2 and roi_pixels.shape[1] == 3:
                # 先求所有像素的平均颜色 (3,)
                mean_bgr = np.mean(roi_pixels, axis=0)
                # 再转灰度: B*0.114 + G*0.587 + R*0.299
                intensity = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
            else:
                intensity = np.mean(roi_pixels)

            intensities.append(intensity)

        return np.array(intensities)

    def extract_roi_intensity_single(self, frame: np.ndarray,
                                      mask: np.ndarray) -> float:
        """单帧提取ROI强度"""
        if frame.ndim == 3:
            gray = np.dot(frame[..., :3].astype(np.float32), [0.114, 0.587, 0.299])
        else:
            gray = frame.astype(np.float32)

        mask_bool = mask > 0
        if np.any(mask_bool):
            return float(np.mean(gray[mask_bool]))
        return 0.0

    def compute_tic_from_frames(self, frames: List[np.ndarray],
                                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从帧列表计算TIC"""
        frames_array = np.stack(frames, axis=0)

        self.tic_data = self.extract_roi_intensities_batch(frames_array, mask)
        self.time_axis = np.arange(len(frames)) / self.fps
        self.smoothed_tic = None
        self._phase_intervals = None
        self._arrival_time = None
        self._peak_time = None

        logger.debug(f"计算TIC: {len(frames)}帧, 时长{self.time_axis[-1]:.1f}s")

        return self.time_axis, self.tic_data

    def compute_tic_from_generator(self, frame_generator,
                                    mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """从生成器计算TIC（节省内存）"""
        intensities = []

        for frame_data in frame_generator:
            intensity = self.extract_roi_intensity_single(frame_data.frame, mask)
            intensities.append(intensity)

        self.tic_data = np.array(intensities)
        self.time_axis = np.arange(len(intensities)) / self.fps
        self.smoothed_tic = None
        self._phase_intervals = None
        self._arrival_time = None
        self._peak_time = None

        return self.time_axis, self.tic_data

    def smooth_tic(self) -> np.ndarray:
        """平滑TIC曲线"""
        if self.tic_data is None:
            raise ValueError("TIC数据未计算")

        if self.smoothed_tic is not None:
            return self.smoothed_tic

        n = len(self.tic_data)
        window = config.tic.SMOOTH_WINDOW_SIZE

        if n < window:
            window = max(3, n // 2 * 2 + 1)

        poly_order = min(config.tic.SMOOTH_POLY_ORDER, window - 1)
        self.smoothed_tic = savgol_filter(self.tic_data, window, poly_order)

        return self.smoothed_tic

    def fit_gamma_model(self) -> Optional[TICFitResult]:
        """
        使用Gamma变异模型拟合TIC
        改进点：
        1. 数据归一化 (0-1)，极大提升拟合成功率和收敛速度
        2. 增加 maxfev (最大迭代次数)
        3. 动态边界调整
        """
        if self.tic_data is None:
            raise ValueError("TIC数据未计算")

        smoothed = self.smooth_tic()
        t = self.time_axis

        # --- 1. 数据预处理：归一化 ---
        # 将数据缩放到 0-1 范围，消除量级差异带来的优化困难
        max_val = np.max(smoothed)
        if max_val <= 1e-6:  # 信号过弱，视为无效
            logger.warning("TIC信号过弱(全0或极低)，跳过拟合")
            return None

        y_norm = smoothed / max_val

        # --- 2. 初始猜测 (基于归一化数据) ---
        peak_idx = np.argmax(y_norm)
        peak_val = y_norm[peak_idx]
        # 取前10%的均值作为基线
        baseline = max(0, np.min(y_norm[:max(1, len(y_norm) // 10)]))

        estimated_A = max(0.1, peak_val - baseline)
        estimated_t0 = max(0.1, t[peak_idx] * 0.4)  # 假设起效时间在峰值前

        # 初始参数 [A, alpha, beta, t0, C]
        # 注意：这里我们给 alpha, beta 设定比较通用的初始值
        p0 = [estimated_A, 2.0, 1.0, estimated_t0, baseline]

        # --- 3. 设定边界 (针对归一化数据) ---
        # A和C在0-1之间，时间参数维持原状
        # 格式: ([下界...], [上界...])
        norm_bounds = (
            [0.0, 0.1, 0.01, 0.0, 0.0],  # Lower
            [50.0, 10.0, 20.0, t[-1], 1.0]  # Upper (A上限放宽以适应超调)
        )

        # 强制钳制初始值 (Clamping)
        epsilon = 1e-4
        p0_clamped = []
        for i, val in enumerate(p0):
            low = norm_bounds[0][i] + epsilon
            up = norm_bounds[1][i] - epsilon
            val = max(low, min(val, up))
            p0_clamped.append(val)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # --- 4. 执行拟合 ---
                # maxfev=20000 允许更多次尝试
                popt_norm, pcov = curve_fit(
                    TICModels.gamma_variate, t, y_norm,
                    p0=p0_clamped, bounds=norm_bounds,
                    maxfev=20000,
                    nan_policy='omit'
                )

            # --- 5. 参数还原 (反归一化) ---
            # I_real = I_norm * max_val
            # Gamma公式中，只有幅值参数 A 和基线 C 需要乘以缩放比例
            # alpha, beta, t0 是时间相关的，不需要缩放
            A_norm, alpha, beta, t0, C_norm = popt_norm

            A = A_norm * max_val
            C = C_norm * max_val

            popt = np.array([A, alpha, beta, t0, C])

            # --- 6. 计算特征指标 ---
            # 使用还原后的参数计算拟合曲线
            fitted = TICModels.gamma_variate(t, *popt)

            # R² (拟合优度)
            ss_res = np.sum((smoothed - fitted) ** 2)
            ss_tot = np.sum((smoothed - np.mean(smoothed)) ** 2)
            r_squared = 1 - ss_res / (ss_tot + 1e-10)

            # 特征参数
            peak_time = t0 + alpha * beta
            # 峰值强度 (代入公式计算最准)
            peak_intensity = TICModels.gamma_variate(np.array([peak_time]), *popt)[0]

            wash_in_rate = alpha / beta if beta > 0 else 0
            wash_out_rate = 1 / beta if beta > 0 else 0

            # AUC (曲线下面积)
            auc = np.trapz(fitted - C, t)

            self.fit_result = TICFitResult(
                params=popt,
                params_cov=None,  # 归一化后的协方差矩阵对原数据意义不大，置空即可
                arrival_time=t0,
                peak_time=peak_time,
                peak_intensity=peak_intensity,
                wash_in_rate=wash_in_rate,
                wash_out_rate=wash_out_rate,
                auc=auc,
                fit_quality=r_squared
            )

            # 更新缓存
            self._arrival_time = t0
            self._peak_time = peak_time

            logger.info(f"TIC拟合成功: R²={r_squared:.3f}, TTP={peak_time:.1f}s")
            return self.fit_result

        except Exception as e:
            logger.warning(f"Gamma拟合最终失败 ({str(e)})，将使用简单的峰值检测代替")
            # 拟合失败时的兜底策略：使用原始数据的峰值
            # 这样程序不会崩溃，只是相控可能不那么精准
            peak_idx_raw = np.argmax(smoothed)
            self._peak_time = t[peak_idx_raw]
            self._arrival_time = max(0, self._peak_time - 2.0)  # 粗略估计
            return None

    def detect_arrival_time(self) -> float:
        """检测到达时间（向量化）"""
        if self.tic_data is None:
            raise ValueError("TIC数据未计算")

        # 如果有拟合结果，使用拟合的t0
        if self.fit_result is not None:
            return self.fit_result.arrival_time

        if self._arrival_time is not None:
            return self._arrival_time

        smoothed = self.smooth_tic()
        n = len(smoothed)

        # 基线（前10%）
        baseline_end = max(1, int(n * config.tic.BASELINE_RATIO))
        baseline = np.mean(smoothed[:baseline_end])

        # 峰值
        peak_idx = np.argmax(smoothed)
        peak_val = smoothed[peak_idx]

        # 阈值
        threshold = baseline + config.tic.ARRIVAL_THRESHOLD_RATIO * (peak_val - baseline)

        # 向量化查找
        search_range = smoothed[baseline_end:peak_idx]
        above_threshold = search_range > threshold

        if np.any(above_threshold):
            arrival_idx = baseline_end + np.argmax(above_threshold)
        else:
            arrival_idx = baseline_end

        self._arrival_time = self.time_axis[arrival_idx]
        return self._arrival_time

    def detect_peak_time(self) -> float:
        """检测峰值时间"""
        if self.tic_data is None:
            raise ValueError("TIC数据未计算")

        if self.fit_result is not None:
            return self.fit_result.peak_time

        if self._peak_time is not None:
            return self._peak_time

        smoothed = self.smooth_tic()

        peaks, _ = find_peaks(smoothed, prominence=np.std(smoothed), distance=int(self.fps))

        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(smoothed[peaks])]
        else:
            peak_idx = np.argmax(smoothed)

        self._peak_time = self.time_axis[peak_idx]
        return self._peak_time

    def get_phase_intervals(self) -> Dict[str, Tuple[float, float]]:
        """获取相位时间区间"""
        if self._phase_intervals is not None:
            return self._phase_intervals

        arrival = self.detect_arrival_time()
        total = self.time_axis[-1] if self.time_axis is not None else 180.0

        arterial_end = min(arrival + config.tic.ARTERIAL_DURATION, total)
        portal_end = min(arterial_end + config.tic.PORTAL_DURATION, total)

        self._phase_intervals = {
            'Arterial': (arrival, arterial_end),
            'Portal': (arterial_end, portal_end),
            'Delay': (portal_end, total)
        }

        return self._phase_intervals

    def get_frame_phase(self, frame_idx: int) -> str:
        """获取单个帧的相位标签"""
        if self.time_axis is None or len(self.time_axis) == 0:
            return "Unknown"

        time = frame_idx / self.fps
        intervals = self.get_phase_intervals()

        for phase, (start, end) in intervals.items():
            if start <= time < end:
                return phase

        return "Delay"

    def get_frame_phases_batch(self, frame_indices: np.ndarray) -> np.ndarray:
        """批量获取相位标签（向量化）"""
        times = frame_indices.astype(np.float64) / self.fps
        intervals = self.get_phase_intervals()

        # 默认Delay (2)
        phases = np.full(len(frame_indices), 2, dtype=np.int32)

        a_start, a_end = intervals['Arterial']
        p_start, p_end = intervals['Portal']

        phases[(times >= a_start) & (times < a_end)] = 0  # Arterial
        phases[(times >= p_start) & (times < p_end)] = 1  # Portal

        return phases

    def get_phase_name(self, phase_id: int) -> str:
        """将相位ID转换为名称"""
        phase_names = {0: 'Arterial', 1: 'Portal', 2: 'Delay'}
        return phase_names.get(phase_id, 'Unknown')

    def find_brightest_frame_batch(self, frames: np.ndarray,
                                    mask: Optional[np.ndarray] = None) -> int:
        """找最亮帧（向量化）"""
        if len(frames) == 0:
            return 0

        if mask is not None:
            intensities = self.extract_roi_intensities_batch(frames, mask)
        else:
            if frames.ndim == 4:
                gray = np.tensordot(frames[..., :3].astype(np.float32),
                                   [0.114, 0.587, 0.299], axes=([3], [0]))
            else:
                gray = frames.astype(np.float32)
            intensities = np.mean(gray, axis=(1, 2))

        return int(np.argmax(intensities))

    def get_fit_curve(self) -> Optional[np.ndarray]:
        """获取拟合曲线"""
        if self.fit_result is None or self.time_axis is None:
            return None
        return TICModels.gamma_variate(self.time_axis, *self.fit_result.params)

    def reset(self):
        """重置分析器状态"""
        self.tic_data = None
        self.time_axis = None
        self.smoothed_tic = None
        self.fit_result = None
        self._phase_intervals = None
        self._arrival_time = None
        self._peak_time = None