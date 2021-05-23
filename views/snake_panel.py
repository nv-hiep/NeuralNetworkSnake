from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

from helper.snake import *

class SnakePanel(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), snake=None):
        super().__init__(parent)
        self.board_size = board_size

        if snake:
            self.snake = snake
        self.setFocus()

        self.show()

    def update(self):
        if self.snake.is_alive:
            self.snake.update()
            self.repaint()
        else:
            # print('Awwww! Dead!')
            pass

    
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_apple(painter)
        self.draw_snake(painter)
        
        painter.end()


    def draw_border(self, painter: QtGui.QPainter) -> None:
        '''
        Border of the panel
        '''
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()

        painter.setPen( QtGui.QPen(Qt.gray, 10) )    # color and line-width
        painter.drawLine(0, 0, width, 0)             # Top line
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)            # Left line

    

    def draw_snake(self, painter: QtGui.QPainter) -> None:
        '''
        draw the snake
        '''
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 127)))
        
        for point in self.snake.snake_array:
            painter.setPen(QtGui.QPen(Qt.green))
            painter.drawEllipse(point.x*DOT_SIZE, point.y*DOT_SIZE, DOT_SIZE, DOT_SIZE)

        

    def draw_apple(self, painter: QtGui.QPainter) -> None:
        apple_location = self.snake.apple_location
        if apple_location:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))
            painter.drawEllipse(apple_location.x*DOT_SIZE, apple_location.y*DOT_SIZE, DOT_SIZE, DOT_SIZE)