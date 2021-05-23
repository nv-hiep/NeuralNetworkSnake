from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

from typing import List

from network.neural_network import *
from helper.snake import Snake
from helper.config import config


class NetworkPanel(QtWidgets.QWidget):
    def __init__(self, parent, snake: Snake):
        super().__init__(parent)
        self.snake     = snake
        self.neuron_xy = {}

        font      = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
        font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)



        # creating a label widgets
        self.label_network = QtWidgets.QLabel('Network Structure', self)
        self.label_network.setFont(font_bold)
        self.label_network.move(200, 25)
        self.label_network.setFixedSize(400, 20)

        self.label_layers = QtWidgets.QLabel('Layer units: ' + '[{}, {}, 4]'.format(config['vision_type'] * 3 + 4 + 4, ', '.join([str(num_neurons) for num_neurons in config['hidden_layer_units'][config['vision_type']] ])), self)
        self.label_layers.setFont(font)
        self.label_layers.move(200, 50)
        self.label_layers.setFixedSize(400, 20)


        self.label_hidden = QtWidgets.QLabel('Hidden layer activation: ' + ' '.join([word.capitalize() for word in config['hidden_activation'].split('_')]), self)
        self.label_hidden.setFont(font)
        self.label_hidden.move(200, 75)
        self.label_hidden.setFixedSize(400, 20)


        self.label_ouput = QtWidgets.QLabel('Output layer activation: ' + ' '.join([word.capitalize() for word in config['output_activation'].split('_')]), self)
        self.label_ouput.setFont(font)
        self.label_ouput.move(200, 100)
        self.label_ouput.setFixedSize(400, 20)


        self.label_vision = QtWidgets.QLabel('Snake vision: ' + str(config['vision_type']) + ' directions', self)
        self.label_vision.setFont(font)
        self.label_vision.move(200, 125)
        self.label_vision.setFixedSize(400, 20)


        self.label_vision_type = QtWidgets.QLabel('Apple/Self Vision: ' + config['apple_and_self_vision'].lower(), self)
        self.label_vision_type.setFont(font)
        self.label_vision_type.move(200, 150)
        self.label_vision_type.setFixedSize(400, 20)





  
        # creating a label widgets
        self.label_generation = QtWidgets.QLabel('Snake Game', self)
        self.label_generation.setFont(font_bold)
        self.label_generation.move(200, 650)
        self.label_generation.setFixedSize(400, 30)

        self.label_generation = QtWidgets.QLabel('Generation: 1', self)
        self.label_generation.setFont(font)
        self.label_generation.move(200, 675)
        self.label_generation.setFixedSize(400, 30)

        self.label_curr_indiv = QtWidgets.QLabel('Individual: 1/100', self)
        self.label_curr_indiv.setFont(font)
        self.label_curr_indiv.move(200, 700)
        self.label_curr_indiv.setFixedSize(400, 30)


        self.label_best_score = QtWidgets.QLabel('Best score: 0', self)
        self.label_best_score.setFont(font)
        self.label_best_score.move(200, 725)
        self.label_best_score.setFixedSize(400, 30)

        self.label_best_fitness = QtWidgets.QLabel('Best fitness: 0', self)
        self.label_best_fitness.setFont(font)
        self.label_best_fitness.move(200, 750)
        self.label_best_fitness.setFixedSize(400, 30)


        self.label_snake_length = QtWidgets.QLabel('Snake length: 0', self)
        self.label_snake_length.setFont(font)
        self.label_snake_length.move(200, 775)
        self.label_snake_length.setFixedSize(400, 30)
        # self.label_snake_length.setStyleSheet('border: 1px solid black;')


        self.show()



    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)
        self._display(painter)
        painter.end()

    

    def update(self) -> None:
        self.repaint()

    

    def _display(self, painter: QtGui.QPainter) -> None:
        '''
        Plot the layers of network
        '''

        # Height and Width of the window
        H = self.frameGeometry().height()
        # W  = self.frameGeometry().width()

        # The input of the network
        inputs = self.snake.vision_as_array

         # [List of ] sizes of the layers
        layer_units = self.snake.network.layer_units

        # Output from a neural network
        pred_node = np.argmax( self.snake.network._forward_prop(inputs) )

        

        # vertical space among units, and radius size of a unit
        input_units = inputs.shape[0]
        vertical_space, radius = (4, 6) if input_units > 32 else (5, 8)
        
        # Margins and offsets
        left_margin = 15
        h_offset    = left_margin
        
        # Draw layers and their units
        for layer, n_units in enumerate(layer_units):

            # Vertical offset - for plotting
            v_offset = (H - ((2*radius + vertical_space) * n_units) )/2
            
            layer_output = None
            weights      = None
            if layer > 0:
                # Output of each layer
                layer_output = self.snake.network.layers[layer].A

                # Weights matrix of each layer
                weights = self.snake.network.layers[layer].W
                prev_n_units = weights.shape[1]
                curr_n_units = weights.shape[0]


            for curr_unit in range(n_units):
                _x = h_offset
                _y = curr_unit * (radius*2 + vertical_space) + v_offset
                
                t = (layer, curr_unit)
                if t not in self.neuron_xy:
                    self.neuron_xy[t] = (_x, _y + radius)
                
                
                
                # Input layer
                if layer == 0:
                    # If the node is fed, it's green, else it's gray
                    painter.setBrush(QtGui.QBrush(Qt.green if inputs[curr_unit, 0] > 0 else Qt.gray))
                
                # Hidden layers
                if (layer > 0) and (layer < len(layer_units) - 1):
                    painter.setBrush(QtGui.QBrush(Qt.cyan if layer_output[curr_unit, 0] > 0. else Qt.gray))
                
                # Output layer
                if layer == len(layer_units) - 1:
                    text = ('Up', 'Down', 'Left', 'Right')[curr_unit]
                    painter.setPen(QtGui.QPen(Qt.red if curr_unit == pred_node else Qt.black))
                    painter.drawText(h_offset + 30, curr_unit * (radius*2 + vertical_space) + v_offset + 1.5*radius, text)
                    painter.setBrush(QtGui.QBrush(Qt.green if curr_unit == pred_node else Qt.gray))

                # draw the nodes as circles
                painter.drawEllipse(_x, _y, radius*2, radius*2)

                # draw lines connecting the nodes
                if layer > 0:
                    for prev_unit in range(prev_n_units):
                        line_color = Qt.blue if weights[curr_unit, prev_unit] > 0 else Qt.gray
                        painter.setPen(QtGui.QPen(line_color))

                        # Locations of the nodes
                        start = self.neuron_xy[(layer-1, prev_unit)]
                        end   = self.neuron_xy[(layer, curr_unit)]
                        
                        # Offset start[0] by diameter of circle so that the line starts on the right of the circle
                        painter.drawLine(start[0] + radius*2, start[1], end[0], end[1])
            # End - unit nodes
            
            # Add distance between layers
            h_offset += 100
        # End - layer
    # End - def
# End - class