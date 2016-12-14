# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_design.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1180, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.userInterface = QtWidgets.QHBoxLayout()
        self.userInterface.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.userInterface.setObjectName("userInterface")
        self.pushButtonRestart = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonRestart.setObjectName("pushButtonRestart")
        self.userInterface.addWidget(self.pushButtonRestart)
        self.pushButtonStep = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonStep.setObjectName("pushButtonStep")
        self.userInterface.addWidget(self.pushButtonStep)
        self.labelSpeed = QtWidgets.QLabel(self.centralwidget)
        self.labelSpeed.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelSpeed.setObjectName("labelSpeed")
        self.userInterface.addWidget(self.labelSpeed)
        self.spinBoxSpeed = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBoxSpeed.setMinimum(1)
        self.spinBoxSpeed.setMaximum(100)
        self.spinBoxSpeed.setProperty("value", 20)
        self.spinBoxSpeed.setObjectName("spinBoxSpeed")
        self.userInterface.addWidget(self.spinBoxSpeed)
        self.pushButtonRun = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonRun.setObjectName("pushButtonRun")
        self.userInterface.addWidget(self.pushButtonRun)
        self.verticalLayout_2.addLayout(self.userInterface)
        self.topWorkers = QtWidgets.QHBoxLayout()
        self.topWorkers.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.topWorkers.setObjectName("topWorkers")
        self.worker_1 = QtWidgets.QFrame(self.centralwidget)
        self.worker_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.worker_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.worker_1.setObjectName("worker_1")
        self.label_worker_1 = QtWidgets.QLabel(self.worker_1)
        self.label_worker_1.setGeometry(QtCore.QRect(0, 0, 60, 20))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_worker_1.sizePolicy().hasHeightForWidth())
        self.label_worker_1.setSizePolicy(sizePolicy)
        self.label_worker_1.setObjectName("label_worker_1")
        self.label_worker_1.raise_()
        self.topWorkers.addWidget(self.worker_1)
        self.worker_2 = QtWidgets.QFrame(self.centralwidget)
        self.worker_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.worker_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.worker_2.setObjectName("worker_2")
        self.label_worker_2 = QtWidgets.QLabel(self.worker_2)
        self.label_worker_2.setGeometry(QtCore.QRect(0, 0, 60, 20))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_worker_2.sizePolicy().hasHeightForWidth())
        self.label_worker_2.setSizePolicy(sizePolicy)
        self.label_worker_2.setObjectName("label_worker_2")
        self.worker_1.raise_()
        self.worker_1.raise_()
        self.worker_1.raise_()
        self.label_worker_2.raise_()
        self.worker_1.raise_()
        self.topWorkers.addWidget(self.worker_2)
        self.verticalLayout_2.addLayout(self.topWorkers)
        self.bottomWorkers = QtWidgets.QHBoxLayout()
        self.bottomWorkers.setObjectName("bottomWorkers")
        self.worker_master = QtWidgets.QFrame(self.centralwidget)
        self.worker_master.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.worker_master.setFrameShadow(QtWidgets.QFrame.Raised)
        self.worker_master.setObjectName("worker_master")
        self.label_worker_master = QtWidgets.QLabel(self.worker_master)
        self.label_worker_master.setGeometry(QtCore.QRect(0, 0, 101, 20))
        self.label_worker_master.setObjectName("label_worker_master")
        self.label_actives = QtWidgets.QLabel(self.worker_master)
        self.label_actives.setGeometry(QtCore.QRect(10, 40, 101, 16))
        self.label_actives.setObjectName("label_actives")
        self.label_superstep = QtWidgets.QLabel(self.worker_master)
        self.label_superstep.setGeometry(QtCore.QRect(200, 40, 71, 16))
        self.label_superstep.setObjectName("label_superstep")
        self.list_actives = QtWidgets.QListWidget(self.worker_master)
        self.list_actives.setGeometry(QtCore.QRect(5, 60, 161, 231))
        self.list_actives.setObjectName("list_actives")
        self.display_superstep = QtWidgets.QLabel(self.worker_master)
        self.display_superstep.setGeometry(QtCore.QRect(210, 60, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.display_superstep.setFont(font)
        self.display_superstep.setObjectName("display_superstep")
        self.bottomWorkers.addWidget(self.worker_master)
        self.worker_3 = QtWidgets.QFrame(self.centralwidget)
        self.worker_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.worker_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.worker_3.setObjectName("worker_3")
        self.label_worker_3 = QtWidgets.QLabel(self.worker_3)
        self.label_worker_3.setGeometry(QtCore.QRect(0, 0, 60, 20))
        self.label_worker_3.setObjectName("label_worker_3")
        self.label_worker_3.raise_()
        self.worker_master.raise_()
        self.worker_master.raise_()
        self.worker_master.raise_()
        self.bottomWorkers.addWidget(self.worker_3)
        self.verticalLayout_2.addLayout(self.bottomWorkers)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1180, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButtonRestart.setText(_translate("MainWindow", "Restart"))
        self.pushButtonStep.setText(_translate("MainWindow", "Step"))
        self.labelSpeed.setText(_translate("MainWindow", "Speed:"))
        self.pushButtonRun.setText(_translate("MainWindow", "Run"))
        self.label_worker_1.setText(_translate("MainWindow", "Worker 1"))
        self.label_worker_2.setText(_translate("MainWindow", "Worker 2"))
        self.label_worker_master.setText(_translate("MainWindow", "Master Worker"))
        self.label_actives.setText(_translate("MainWindow", "Active Vertices"))
        self.label_superstep.setText(_translate("MainWindow", "Superstep"))
        self.display_superstep.setText(_translate("MainWindow", "0"))
        self.label_worker_3.setText(_translate("MainWindow", "Worker 3"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
