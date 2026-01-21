"""
CEUS视频处理系统 - 主入口
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from config import config
from utils.logger import LoggerManager
from gui.main_window import MainWindow


def main():
    """主函数"""
    # 初始化日志
    LoggerManager.setup()
    logger = LoggerManager.get_logger('main')
    logger.info("=" * 50)
    logger.info("CEUS视频处理系统启动")
    logger.info("=" * 50)

    # 确保目录存在
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.log.LOG_DIR, exist_ok=True)

    # 高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 样式
    app.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px;
        }
        QPushButton {
            padding: 8px 16px;
            border-radius: 4px;
            border: 1px solid #ccc;
            background-color: #fff;
        }
        QPushButton:hover { background-color: #e8e8e8; }
        QPushButton:pressed { background-color: #ddd; }
        QProgressBar {
            border: 1px solid #ccc;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk { background-color: #4CAF50; }
        QListWidget { border: 1px solid #ddd; border-radius: 4px; }
        QTextEdit { border: 1px solid #ddd; border-radius: 4px; }
        QTabWidget::pane { border: 1px solid #ddd; border-radius: 4px; }
    """)

    # 创建主窗口
    window = MainWindow()
    window.show()

    logger.info("主窗口已显示")

    ret = app.exec_()

    logger.info("程序退出")
    sys.exit(ret)


if __name__ == "__main__":
    main()