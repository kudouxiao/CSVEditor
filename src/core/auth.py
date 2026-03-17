import os
import uuid
import hashlib
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

from src.config import LICENSE_PATH
SECRET_SALT = "G1_PRO_EDITOR_2026_SECRET"

def get_machine_code():
    mac = uuid.getnode()
    return hashlib.md5(str(mac).encode('utf-8')).hexdigest()[:8].upper()

def generate_license_for_machine(machine_code):
    return hashlib.sha256((machine_code + SECRET_SALT).encode('utf-8')).hexdigest()[:16].upper()

def verify_license():
    if not os.path.exists(LICENSE_PATH):
        return False
    try:
        with open(LICENSE_PATH, "r") as f:
            key = f.read().strip()
        expected = generate_license_for_machine(get_machine_code())
        return key == expected
    except Exception:
        return False

def save_license(key):
    with open(LICENSE_PATH, "w") as f:
        f.write(key)

class AuthDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("G1 Pro Editor - Authorization")
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout(self)
        
        self.machine_code = get_machine_code()
        
        lbl_info = QLabel("本软件需要授权才能使用。\n请将以下机器码发送给供应商以获取激活码。")
        layout.addWidget(lbl_info)
        
        lbl_code = QLabel(f"<b>机器码: {self.machine_code}</b>")
        lbl_code.setTextInteractionFlags(lbl_code.textInteractionFlags() | 1) # Selectable
        layout.addWidget(lbl_code)
        
        self.txt_key = QLineEdit()
        self.txt_key.setPlaceholderText("在此处输入激活码 (License Key)...")
        layout.addWidget(self.txt_key)
        
        btn_activate = QPushButton("✅ 激活 (Activate)")
        btn_activate.clicked.connect(self.on_activate)
        layout.addWidget(btn_activate)

    def on_activate(self):
        entered_key = self.txt_key.text().strip()
        expected = generate_license_for_machine(self.machine_code)
        
        if entered_key == expected:
            save_license(entered_key)
            QMessageBox.information(self, "Success", "软件激活成功！非常感谢您的支持。 (Activated successfully!)")
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "激活码无效，请重试。(Invalid License Key)")

