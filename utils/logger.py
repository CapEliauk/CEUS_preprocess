"""
日志模块 - 统一日志管理
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

from config import config


class LoggerManager:
    """日志管理器"""

    _loggers: dict = {}
    _initialized: bool = False

    @classmethod
    def setup(cls):
        """初始化日志系统"""
        if cls._initialized:
            return

        log_dir = config.log.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'ceus_{timestamp}.log')

        # 配置根日志器
        root_logger = logging.getLogger('ceus')
        root_logger.setLevel(getattr(logging, config.log.LOG_LEVEL))

        # 文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.log.MAX_LOG_SIZE_MB * 1024 * 1024,
            backupCount=config.log.MAX_LOG_FILES,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt=config.log.LOG_FORMAT,
                datefmt=config.log.LOG_DATE_FORMAT
            )
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.log.CONSOLE_LOG_LEVEL))
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        cls._initialized = True
        root_logger.info(f"日志系统初始化完成，日志文件: {log_file}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """获取日志器"""
        if not cls._initialized:
            cls.setup()

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(f'ceus.{name}')

        return cls._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """获取日志器的便捷函数"""
    return LoggerManager.get_logger(name)