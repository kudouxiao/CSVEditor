import sys
import os

# 将当前目录加入路径，确保能找到 src 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
import PyQt5.sip # Dummy import to force PyInstaller to bundle this C-extension
import qdarktheme

from src.ui.main_window import MainWindow
from src.core.auth import verify_license, AuthDialog

if __name__ == "__main__":
    # 高分屏适配
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # 应用主题
    qdarktheme.setup_theme("dark")

    # 检查授权
    if not verify_license():
        auth_dialog = AuthDialog()
        if auth_dialog.exec_() != AuthDialog.Accepted:
            sys.exit(0)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())