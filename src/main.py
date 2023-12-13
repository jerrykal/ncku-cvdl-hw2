import sys

import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow

from bg_subtract import background_subtraction
from optical_flow import find_tracking_point, video_tracking
from pca import dimension_reduction
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.frames = []

        self.btnLoadImg.clicked.connect(self.load_image)
        self.btnLoadVid.clicked.connect(self.load_video)

        # Q1
        self.btnQ1.clicked.connect(lambda: background_subtraction(self.frames))

        # Q2
        self.btnQ21.clicked.connect(lambda: find_tracking_point(self.frames[0]))
        self.btnQ22.clicked.connect(lambda: video_tracking(self.frames))

        # Q3
        self.btnQ3.clicked.connect(lambda: dimension_reduction(self.image))

    def load_image(self):
        imgpath = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.png)"
        )[0]
        self.image = cv2.imread(imgpath)

    def load_video(self):
        vidpath = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )[0]
        self.frames = []

        # Read video
        cap = cv2.VideoCapture(vidpath)
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Add frame to list
            self.frames.append(frame)

        cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
