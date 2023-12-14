from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel


class DrawingWidget(QLabel):
    def __init__(self, parent, rect):
        super().__init__(parent)
        self.init_ui(rect)

    def init_ui(self, rect):
        self.setGeometry(rect)

        self.pix = QPixmap(self.size())  # type: ignore
        self.pix.fill(Qt.black)  # type: ignore

        self.last_point = None
        self.end_point = None

    def paintEvent(self, event):
        pixmap_painter = QPainter(self.pix)
        pixmap_painter.setPen(QPen(Qt.white, 5))  # type: ignore

        if self.last_point is not None and self.end_point is not None:
            pixmap_painter.drawLine(self.last_point, self.end_point)
            self.last_point = self.end_point

        widget_painter = QPainter(self)
        widget_painter.drawPixmap(0, 0, self.pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # type: ignore
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:  # type: ignore
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:  # type: ignore
            self.end_point = event.pos()
            self.update()

    def reset(self):
        self.pix.fill(Qt.black)  # type: ignore
        self.last_point = None
        self.end_point = None
        self.update()
