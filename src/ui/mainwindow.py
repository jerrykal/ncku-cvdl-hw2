# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'assets/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1206, 854)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnLoadImg = QtWidgets.QPushButton(self.centralwidget)
        self.btnLoadImg.setGeometry(QtCore.QRect(20, 70, 131, 32))
        self.btnLoadImg.setObjectName("btnLoadImg")
        self.btnLoadVid = QtWidgets.QPushButton(self.centralwidget)
        self.btnLoadVid.setGeometry(QtCore.QRect(20, 120, 131, 32))
        self.btnLoadVid.setObjectName("btnLoadVid")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(190, 30, 231, 91))
        self.groupBox.setObjectName("groupBox")
        self.btnQ1 = QtWidgets.QPushButton(self.groupBox)
        self.btnQ1.setGeometry(QtCore.QRect(20, 40, 191, 32))
        self.btnQ1.setObjectName("btnQ1")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(190, 150, 231, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.btnQ21 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ21.setGeometry(QtCore.QRect(20, 40, 191, 32))
        self.btnQ21.setObjectName("btnQ21")
        self.btnQ22 = QtWidgets.QPushButton(self.groupBox_2)
        self.btnQ22.setGeometry(QtCore.QRect(21, 80, 191, 32))
        self.btnQ22.setObjectName("btnQ22")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(190, 300, 231, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.btnQ3 = QtWidgets.QPushButton(self.groupBox_3)
        self.btnQ3.setGeometry(QtCore.QRect(20, 40, 191, 32))
        self.btnQ3.setObjectName("btnQ3")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(460, 30, 621, 251))
        self.groupBox_4.setObjectName("groupBox_4")
        self.btnQ41 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ41.setGeometry(QtCore.QRect(20, 40, 191, 32))
        self.btnQ41.setObjectName("btnQ41")
        self.btnQ42 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ42.setGeometry(QtCore.QRect(20, 80, 191, 32))
        self.btnQ42.setObjectName("btnQ42")
        self.btnQ43 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ43.setGeometry(QtCore.QRect(20, 120, 191, 32))
        self.btnQ43.setObjectName("btnQ43")
        self.btnQ44 = QtWidgets.QPushButton(self.groupBox_4)
        self.btnQ44.setGeometry(QtCore.QRect(20, 160, 191, 32))
        self.btnQ44.setObjectName("btnQ44")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(460, 300, 621, 371))
        self.groupBox_5.setObjectName("groupBox_5")
        self.btnQ5LoadImg = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ5LoadImg.setGeometry(QtCore.QRect(20, 40, 191, 32))
        self.btnQ5LoadImg.setObjectName("btnQ5LoadImg")
        self.btnQ51 = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ51.setGeometry(QtCore.QRect(20, 80, 191, 32))
        self.btnQ51.setObjectName("btnQ51")
        self.btnQ52 = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ52.setGeometry(QtCore.QRect(20, 120, 191, 32))
        self.btnQ52.setObjectName("btnQ52")
        self.btnQ53 = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ53.setGeometry(QtCore.QRect(20, 160, 191, 32))
        self.btnQ53.setObjectName("btnQ53")
        self.btnQ54 = QtWidgets.QPushButton(self.groupBox_5)
        self.btnQ54.setGeometry(QtCore.QRect(20, 200, 191, 32))
        self.btnQ54.setObjectName("btnQ54")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1206, 42))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnLoadImg.setText(_translate("MainWindow", "Load Image"))
        self.btnLoadVid.setText(_translate("MainWindow", "Load Video"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Background Subtraction"))
        self.btnQ1.setText(_translate("MainWindow", "1. Background Subtraction"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Optical Flow"))
        self.btnQ21.setText(_translate("MainWindow", "2.1 Preprocessing"))
        self.btnQ22.setText(_translate("MainWindow", "2.2 Video Tracking"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. PCA"))
        self.btnQ3.setText(_translate("MainWindow", "3. Dimension Reduction"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. MNIST Classifier Using VGG19"))
        self.btnQ41.setText(_translate("MainWindow", "4.1 Show Model Structure"))
        self.btnQ42.setText(_translate("MainWindow", "4.2 Show Acc && Loss"))
        self.btnQ43.setText(_translate("MainWindow", "4.3 Predict"))
        self.btnQ44.setText(_translate("MainWindow", "4.4 Reset"))
        self.groupBox_5.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.btnQ5LoadImg.setText(_translate("MainWindow", "Load Image"))
        self.btnQ51.setText(_translate("MainWindow", "5.1 Show Images"))
        self.btnQ52.setText(_translate("MainWindow", "5.2 Show Model Structure"))
        self.btnQ53.setText(_translate("MainWindow", "5.3 Show Comparison"))
        self.btnQ54.setText(_translate("MainWindow", "5.4 Inference"))
