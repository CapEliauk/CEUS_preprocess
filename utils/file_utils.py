"""
文件工具模块
"""
import os
import re
import shutil
from pathlib import Path
from typing import Generator, Tuple
import cv2
import numpy as np
from config import config
from utils.logger import get_logger

logger = get_logger('file_utils')


def find_video_files(root_dir: str) -> Generator[Tuple[str, str], None, None]:
    """递归查找视频文件"""
    root_path = Path(root_dir)
    count = 0

    for path in root_path.rglob('*'):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in config.video.VIDEO_EXTENSIONS:
                count += 1
                yield str(path), str(path.relative_to(root_path))
            elif suffix in config.video.DICOM_EXTENSIONS or suffix == '':
                if is_dicom_file(str(path)):
                    count += 1
                    yield str(path), str(path.relative_to(root_path))

    logger.info(f"在 {root_dir} 中找到 {count} 个视频文件")


def is_dicom_file(filepath: str) -> bool:
    """检查是否为DICOM文件"""
    try:
        # 前置检查：文件长度不足132字节 -> 不可能是DICOM文件
        if Path(filepath).stat().st_size < 132:
            return False
        else:
            with open(filepath, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
    except:
        return False


def create_output_structure(input_dir: str, output_dir: str, relative_path: str) -> str:
    """创建输出目录结构"""
    output_path = Path(output_dir) / Path(relative_path).parent
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def get_output_filename(original_name: str, phase: str, clip_idx: int) -> str:
    """生成输出文件名"""
    stem = Path(original_name).stem
    return f"{stem}_{phase}_clip{clip_idx:04d}{config.video.OUTPUT_FORMAT}"

def remove_path(path: str) -> bool:
    """
    安全删除文件或文件夹
    :param path: 要删除的路径
    :return:
    """
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True) # ignore_errors防止因文件占用报错崩溃
            logger.info(f"已清理临时文件: {path}")
            return True
        return True
    except Exception as e:
        logger.error(f"清理临时文件失败{path}：{e}")
        return False

def sanitize_filename(name: str) -> str:
    """清理文件名"""
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name[:30] if name else "file"


def cv2_imread(filepath: str):
    """支持中文路径的图像读取"""
    try:
        return cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        return None


def cv2_imwrite(filepath: str, img) -> bool:
    """支持中文路径的图像保存"""
    try:
        ext = os.path.splitext(filepath)[1]
        result, encoded = cv2.imencode(ext, img)
        if result:
            encoded.tofile(filepath)
            return True
    except Exception as e:
        logger.error(f"保存图像失败: {e}")
    return False