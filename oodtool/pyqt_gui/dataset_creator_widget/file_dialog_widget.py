from pathlib import Path
from PyQt5.QtWidgets import (QHBoxLayout, QDialog,
                             QVBoxLayout, QTreeView,
                             QListView, QAbstractItemView,
                             QLabel, QFileSystemModel,
                             QLineEdit, QDialogButtonBox,
                             QPushButton, QStyle,
                             QDesktopWidget, QApplication
                             )
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, Qt


class FileDiaolgWidget(QDialog):
    update_parameters_dock_widget = pyqtSignal(list)

    def __init__(self, root_path: str, parent=None):

        super(FileDiaolgWidget, self).__init__(parent=parent)

        hlay = QHBoxLayout(self)
        vlay = QVBoxLayout()
        vlay2 = QVBoxLayout()
        hlay_run_line = QHBoxLayout()

        hlay_button = QHBoxLayout()

        self.setWindowTitle('Sources')
        self.setModal(True)

        self.treeview = QTreeView(
            selectionMode=QTreeView.ExtendedSelection
        )

        self.listview_open = QListView()
        self.listview_open.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )

        self.run_path = root_path

        self.listview_select_train = QListView(self)
        self.listview_select_test = QListView(self)

        # ------------------------------------------------
        self.model_train = QStandardItemModel()
        self.listview_select_train.setModel(self.model_train)

        self.model_test = QStandardItemModel()
        self.listview_select_test.setModel(self.model_test)
        # ------------------------------------------------

        self.add_btn_train = QPushButton()
        self.add_btn_train.setText("Add to train")

        self.add_btn_test = QPushButton()
        self.add_btn_test.setText("Add to test")

        self.run_btn = QDialogButtonBox()
        self.run_btn.setStandardButtons(QDialogButtonBox.Ok)

        self.back_button = QPushButton()
        pixmap = getattr(QStyle, 'SP_ArrowBack')
        icon = self.style().standardIcon(pixmap)
        self.back_button.setIcon(icon)
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(
            self.on_back_clicked
        )
        self.select_text_label = QLabel()
        self.select_text_label.setText('Selected: ')

        self.run_line = QLineEdit()
        self.run_line.textChanged.connect(self.text_change)
        self.run_line.setFocusPolicy(Qt.ClickFocus)

        self.path_label = QLabel()

        self.text_label = QLabel()
        self.text_label.setText('Look in: ')
        hlay_run_line.addWidget(self.text_label)
        hlay_run_line.addWidget(self.run_line)
        hlay_run_line.addWidget(self.back_button)

        vlay2.addLayout(hlay_run_line)

        vlay2.addWidget(self.treeview)

        hlay.addLayout(vlay2)

        vlay.addLayout(hlay_button)
        vlay.addWidget(self.listview_open)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.add_btn_train)
        button_layout.addWidget(self.add_btn_test)
        vlay.addLayout(button_layout)

        vlay.addWidget(self.select_text_label)

        listview_layout = QHBoxLayout()
        listview_layout.addWidget(self.listview_select_train)
        listview_layout.addWidget(self.listview_select_test)
        vlay.addLayout(listview_layout)

        vlay.addWidget(self.run_btn)

        hlay.addLayout(vlay)

        self.dirModel = QFileSystemModel(parent=self)

        for x in QtCore.QStorageInfo().mountedVolumes():
            self.dirModel.setRootPath(x.rootPath())

        self.dirModel.setFilter(
            QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)

        self.treeview.setModel(self.dirModel)

        self.fileModel = QFileSystemModel(parent=self)
        self.listview_open.setModel(self.fileModel)
        self.listview_open.setRootIndex(self.fileModel.index(QtCore.QDir.rootPath()))

        self.treeview.setColumnWidth(0, 400)

        self.resize(1000, 700)
        self.center()
        self.treeview.clicked.connect(self.on_clicked)
        self.treeview.doubleClicked.connect(self.on_double_clicked)
        self.treeview.activated.connect(self.on_clicked)

        self.listview_open.doubleClicked.connect(self.on_add_clicked)

        self.treeview.columnResized(0, 1, 300)

        self.add_btn_train.clicked.connect(lambda: self.on_add_clicked(test=False))
        self.add_btn_test.clicked.connect(lambda: self.on_add_clicked(test=True))

        self.run_btn.clicked.connect(self.on_ok_clicked)

        self.treeview.selectionModel().selectionChanged.connect(self.on_pressed)

        self.train_path_list = []
        self.test_path_list = []
        self.index = []

        self.run_line.setText(self.run_path)
        self.text_change(self.run_path)

    def text_change(self, text):
        if Path(text).is_dir():
            self.run_path = text
            new_path = self.dirModel.setRootPath(self.run_path)
            self.treeview.setRootIndex(new_path)
            self.back_button.setEnabled(True)

    def on_back_clicked(self):
        if self.run_path is not None:
            parent_path = Path(self.run_path).parent
            self.run_line.setText(str(parent_path))
        else:
            print('self.run_path is None')

    def on_double_clicked(self, index):
        self.run_line.setText(self.path)

    def __on_ok_clicked(self, model):
        path_list = []
        for row in range(model.rowCount()):
            index = model.item(row)
            path_list.append(index.text())
        return path_list

    def on_ok_clicked(self):
        self.train_path_list = self.__on_ok_clicked(self.model_train)
        self.test_path_list = self.__on_ok_clicked(self.model_test)

        self.update_parameters_dock_widget.emit(
            [self.train_path_list, self.test_path_list]
        )

        self.listview_select_train.model().removeRows(0, self.model_train.rowCount())
        self.listview_select_test.model().removeRows(0, self.model_test.rowCount())
        self.close()

    def on_pressed(self, ev):
        if len(ev) != 0:
            self.path = self.dirModel.fileInfo(
                ev.indexes()[0]).absoluteFilePath()
            self.show_listview()

    def on_clicked(self, index):
        self.index.append(index)
        self.path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.show_listview()

    def show_listview(self):
        self.path_label.setText(self.path)
        self.back_button.setEnabled(True)
        self.listview_open.clearSelection()
        self.listview_open.setRootIndex(
            self.fileModel.setRootPath(self.path)
        )

    def on_add_clicked(self, test: bool):
        self._filling_listview_select(test)

    def _filling_listview_select(self, test: bool):
        itms = self.listview_open.selectedIndexes()

        if len(itms) != 0:
            parent_path = self.dirModel.fileInfo(
                self.index[-1]).absoluteFilePath()
            for it in itms:
                selected_folder = (Path(parent_path)/it.data()).resolve()
                item = QStandardItem(str(selected_folder))
                if test:
                    self.model_test.appendRow(item)
                else:
                    self.model_train.appendRow(item)

    def delete_from_model(self, model, listview_select):
        for items in reversed(sorted(listview_select.selectedIndexes())):
            model.takeRow(items.row())

    def keyPressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()

        if event.key() == QtCore.Qt.Key_Delete:
            self.delete_from_model(self.model_train, self.listview_select_train)
            self.delete_from_model(self.model_test, self.listview_select_test)

        if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_Return:
            self.on_add_clicked(1)

        if modifiers == (QtCore.Qt.ControlModifier |
                         QtCore.Qt.ShiftModifier) and event.key() == QtCore.Qt.Key_Return:
            self.on_ok_clicked()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
