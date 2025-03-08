import sys

from PySide6.QtWidgets import QApplication
from gui import CADASTRAR_CONFIG, CADASTRAR_LOGIN
from gui import MainWindows

def main():

    app = QApplication(sys.argv)
    window = MainWindows(cadastro_config=CADASTRAR_CONFIG, login_config=CADASTRAR_LOGIN)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
