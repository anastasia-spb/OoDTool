from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (
    QMessageBox,
)

from oodtool.pyqt_gui.data_loader.loader import DataLoader

from oodtool.pyqt_gui.ood_images_tab.settings_widget import SettingsWidget
from oodtool.pyqt_gui.main_window import App


def test_settings_widget_reload_button(qtbot, mocker):
    data_loader = DataLoader()
    widget = SettingsWidget(None, data_loader)
    qtbot.addWidget(widget)
    mocker.patch.object(QMessageBox, 'warning', return_value=QMessageBox.Ok)
    qtbot.mouseClick(widget.reload_button, Qt.MouseButton.LeftButton)


def test_sanity_app(qtbot):
    window = App()
    qtbot.addWidget(window)
    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "OoD Tool"


