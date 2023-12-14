import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from PIL import ImageQt
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from torchsummary import summary
from torchvision.transforms import v2

from bg_subtract import background_subtraction
from optical_flow import find_tracking_point, video_tracking
from pca import dimension_reduction
from ui.drawing_widget import DrawingWidget
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.frames = []
        self.drawing_widget = DrawingWidget(self.groupBox_4, QRect(260, 40, 361, 191))

        self.btnLoadImg.clicked.connect(self.load_image)
        self.btnLoadVid.clicked.connect(self.load_video)

        # Load VGG19 model
        self.vgg19_model = torchvision.models.vgg19_bn(num_classes=10)
        self.vgg19_model.load_state_dict(
            torch.load(
                os.path.abspath(
                    os.path.join(
                        __file__, os.pardir, os.pardir, "models", "vgg19_bn.pth"
                    )
                ),
                map_location="cpu",
            )
        )
        self.vgg19_model.eval()

        # Q1
        self.btnQ1.clicked.connect(lambda: background_subtraction(self.frames))

        # Q2
        self.btnQ21.clicked.connect(lambda: find_tracking_point(self.frames[0]))
        self.btnQ22.clicked.connect(lambda: video_tracking(self.frames))

        # Q3
        self.btnQ3.clicked.connect(lambda: dimension_reduction(self.image))

        # Q4
        self.btnQ41.clicked.connect(lambda: summary(self.vgg19_model, (3, 32, 32)))
        self.btnQ42.clicked.connect(self.show_loss_and_acc)
        self.btnQ43.clicked.connect(self.vgg19_inference)
        self.btnQ44.clicked.connect(self.reset_q4)

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

    def show_loss_and_acc(self) -> None:
        """Show loss and accuracy for the VGG19 model."""
        image = cv2.imread(
            os.path.abspath(
                os.path.join(__file__, os.pardir, os.pardir, "logs", "loss_and_acc.png")
            )
        )
        cv2.imshow("Loss and Accuracy", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def vgg19_inference(self):
        qimg = self.drawing_widget.pix.toImage()

        # Convert QImage to PIL Image
        pil_img = ImageQt.fromqimage(qimg)

        # Preprocess image
        transform = v2.Compose(
            [
                v2.Resize((32, 32)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        img = transform(pil_img)

        # Inference
        input = img.unsqueeze(0)
        output = self.vgg19_model(input)
        pred = torch.argmax(output, dim=1).item()
        self.lblVGGPred.setText(str(pred))

        # Show predict probability
        plt.bar(range(0, 10), torch.softmax(output, dim=1).squeeze().tolist())
        plt.title("Probability of each class")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.yticks([i / 10 for i in range(0, 11)])
        plt.xlabel("Class")
        plt.xticks(range(0, 10))
        plt.show()

    def reset_q4(self):
        self.lblVGGPred.setText("")
        self.drawing_widget.reset()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
