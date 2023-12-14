import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image, ImageQt
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from torchsummary import summary
from torchvision.transforms import v2

from bg_subtract import background_subtraction
from custom_resnet50 import CustomResNet50
from optical_flow import find_tracking_point, video_tracking
from pca import dimension_reduction
from train_resnet50 import transform_val as transform_resnet50
from ui.drawing_widget import DrawingWidget
from ui.mainwindow import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.image = None
        self.q5_image = None
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

        # Load ResNet50 model
        self.resnet50_model = CustomResNet50()
        self.resnet50_model.load_state_dict(
            torch.load(
                os.path.abspath(
                    os.path.join(
                        __file__, os.pardir, os.pardir, "models", "resnet50_re.pth"
                    )
                ),
                map_location="cpu",
            )
        )
        self.resnet50_model.eval()

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

        # Q5
        self.btnQ5LoadImg.clicked.connect(self.load_q5_image)
        self.btnQ51.clicked.connect(self.show_q5_image)
        self.btnQ52.clicked.connect(lambda: summary(self.resnet50_model, (3, 224, 224)))
        self.btnQ53.clicked.connect(self.show_acc_comparison)
        self.btnQ54.clicked.connect(self.resnet50_inference)

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

    def load_q5_image(self):
        imgpath = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.png)"
        )[0]
        if imgpath == "":
            return

        self.q5_image = Image.open(imgpath)

        qimg = QPixmap(imgpath)
        self.lblQ5Img.setPixmap(qimg.scaled(self.lblQ5Img.size()))

    def show_q5_image(self):
        data_dir = os.path.abspath(
            os.path.join(
                __file__, os.pardir, os.pardir, "data", "Q5", "inference_dataset"
            )
        )

        # Pick a random cat image
        cat_dir = os.path.join(data_dir, "Cat")
        cat_img = Image.open(
            os.path.join(cat_dir, np.random.choice(os.listdir(cat_dir)))
        )

        # Pick a random dog image
        dog_dir = os.path.join(data_dir, "Dog")
        dog_img = Image.open(
            os.path.join(dog_dir, np.random.choice(os.listdir(dog_dir)))
        )

        # Resize image
        cat_img = v2.ToPILImage()(transform_resnet50(cat_img))
        dog_img = v2.ToPILImage()(transform_resnet50(dog_img))

        # Show images using matplotlib
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cat_img)
        ax[1].imshow(dog_img)
        ax[0].set_title("Cat")
        ax[1].set_title("Dog")
        plt.show()

    def show_acc_comparison(self):
        image = cv2.imread(
            os.path.abspath(
                os.path.join(
                    __file__,
                    os.pardir,
                    os.pardir,
                    "logs",
                    "resnet50_acc_comparison.png",
                )
            )
        )
        cv2.imshow("Accuracy Comparison", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resnet50_inference(self):
        if self.q5_image is None:
            return

        # Preprocess image
        input = transform_resnet50(self.q5_image).unsqueeze(0)

        # Inference
        output = self.resnet50_model(input)
        pred = torch.round(output).item()

        # Show result
        self.lblResNetPred.setText("Cat" if pred == 1 else "Dog")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
