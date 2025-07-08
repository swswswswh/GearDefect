from PyQt5.QtWidgets import *
from detector import GearDetector
import os


class DefectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("齿轮缺陷检测系统")
        self.detector = GearDetector()
        self.test_dir = "test"

        # UI组件初始化
        self.image_label = QLabel()
        self.result_text = QTextEdit()
        self._setup_ui()

    def _setup_ui(self):
        # 布局设置代码...
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self._start_detection)

    def _start_detection(self):
        if hasattr(self, 'current_img'):
            defects = self.detector.detect(self.current_img)
            self._show_results(defects)
