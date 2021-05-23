import sys
import numpy as np

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

from typing import List
from decimal import Decimal

from helper.snake import *
from network.neural_network import Network, sigmoid, linear, relu

from views.network_panel import NetworkPanel
from views.snake_panel   import SnakePanel

from GA.population import Population
from GA.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from GA.mutation import gaussian_mutation, random_uniform_mutation
from GA.crossover import simulated_binary_crossover as SBX
from GA.crossover import uniform_binary_crossover, single_point_binary_crossover

from helper.config import config

import random




class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,
                 config,
                 show=True,
                 fps=800        # frames per second
                 ):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.setPalette(palette)


        #
        # The board box of the game
        #
        self.board_size = config['board_size']
        self.border = (10, 10, 10, 10)  # Left, Top, Right, Bottom : border width
        self.snake_box_width  = DOT_SIZE * self.board_size
        self.snake_box_height = DOT_SIZE * self.board_size

        # Allows padding of the other elements even if we need to restrict the size of the play area
        self._snake_box_width  = max(self.snake_box_width, 720)
        self._snake_box_height = max(self.snake_box_height, 700)

        # Size of the frame, and its location wrt the PC screen
        self.top    = 50
        self.left   = 50
        self.width  = self._snake_box_width + 500 + self.border[0] + self.border[2]
        self.height = self._snake_box_height + self.border[1] + self.border[3] + 230




        #
        # config for Genetic Algorithm        
        #
        self.config = config
        self._SBX_eta = self.config['SBX_eta']
        self._mutation_bins  = np.cumsum([self.config['probability_gaussian'],
                                          self.config['probability_random_uniform'] ])
        self._crossover_bins = np.cumsum([self.config['probability_SBX'],
                                          self.config['probability_SPBX'] ])
        self._SPBX_type = self.config['SPBX_type'].lower()
        self._mutation_rate = self.config['mutation_rate']

        # Determine size of next gen based off selection type
        # selection_type = plus: size = num_parrents + num_offspring
        # selection_type = comma: size = num_offspring
        self._next_gen_size = None
        if self.config['selection_type'].lower() == 'plus':
            self._next_gen_size = self.config['num_parents'] + self.config['num_offspring']
        elif self.config['selection_type'].lower() == 'comma':
            self._next_gen_size = self.config['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(self.config['selection_type']))

        
        



        #
        # List of snake individuals for the Genetic Algorithm
        #
        individuals: List[Individual] = []

        for _ in range(self.config['num_parents']):
            individual = Snake(self.board_size,
                              hidden_layer_units=self.config['hidden_layer_units'][config['vision_type']],
                              hidden_activation=self.config['hidden_activation'],
                              output_activation=self.config['output_activation'],
                              lifespan=self.config['lifespan'],
                              apple_and_self_vision=self.config['apple_and_self_vision'])
            individuals.append(individual)

        self.best_fitness = 0
        self.best_score   = 0

        self._current_individual = 0
        self.population = Population(individuals)

        self.snake = self.population.individuals[self._current_individual]
        self.current_generation = 0

        


        #
        # Draw window of the game
        #
        self.init_window()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start( int(1000./fps) )

        if show:
            self.show() # Show the app window

    





    def init_window(self):
        '''
        Initiate the game window
        '''
        self.main_window = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_window)
        self.setWindowTitle('Snake - Neural Network')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create the Neural Network window
        self.network_window = NetworkPanel(self.main_window, self.snake)
        self.network_window.setGeometry(QtCore.QRect(0, 30, 1000, self._snake_box_height + self.border[1] + self.border[3] + 210))
        self.network_window.setObjectName('network_window')

        # Create Snake window
        self.snake_window = SnakePanel(self.main_window, self.board_size, self.snake)
        self.snake_window.setGeometry(QtCore.QRect(600 + self.border[0], 150 + self.border[1], self.snake_box_width, self.snake_box_height))
        self.snake_window.setObjectName('snake_window')

    

    def update(self) -> None:
        self.snake_window.update()
        self.network_window.update()
        
        # If current individual is alive
        if self.snake.is_alive:
            self.snake.move()
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score
                self.network_window.label_best_score.setText('Best score: ' + str(self.snake.score))

            self.network_window.label_snake_length.setText( 'Snake length: ' + str( len(self.snake.body_locations) ))
        # Current individual is dead         
        else:
            # Calculate fitness of current individual
            self.snake.calculate_fitness()
            fitness = self.snake.fitness
            # print(self._current_individual, fitness)
# 
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                txt = '{:.2E}'.format(Decimal(fitness) ) if fitness > 100. else  str( round(fitness, 1) )
                self.network_window.label_best_fitness.setText('Best fitness: ' + txt)

            # Next generation
            self._current_individual += 1
            if (self.current_generation > 0 and self._current_individual == self._next_gen_size) or\
                (self.current_generation == 0 and self._current_individual == config['num_parents']):
                print(self.config)
                print('======================= Gneration {} ======================='.format(self.current_generation))
                print('----Max fitness:', self.population.fittest_individual.fitness)
                print('----Best Score:', self.population.fittest_individual.score)
                print('----Average fitness:', self.population.average_fitness)
                self.next_generation()
            else:
                current_pop = self.config['num_parents'] if self.current_generation == 0 else self._next_gen_size
                self.network_window.label_curr_indiv.setText('Individual: ' + str(self._current_individual + 1) + '/' + str(current_pop))

            self.snake = self.population.individuals[self._current_individual]
            self.snake_window.snake = self.snake
            self.network_window.snake = self.snake

    



    def next_generation(self):
        self._increment_generation()
        self._current_individual = 0

        # Calculate fitness of individuals
        for individual in self.population.individuals:
            individual.calculate_fitness()
        
        self.population.individuals = elitism_selection(self.population, self.config['num_parents'])
        
        
        # Shuffle and generate next_population set
        random.shuffle(self.population.individuals)
        next_pop: List[Snake] = []

        # parents + offspring selection type ('plus')
        if self.config['selection_type'].lower() == 'plus':
            for individual in self.population.individuals:
                # Decrement lifespan
                individual.lifespan -= 1

                board_size         = individual.board_size
                
                indv_network       = individual.network
                hidden_layer_units = individual.hidden_layer_units
                hidden_activation  = individual.hidden_activation
                output_activation  = individual.output_activation
                
                lifespan                  = individual.lifespan
                apple_and_self_vision     = individual.apple_and_self_vision # distance or binary

                start_pos          = individual.start_position
                apple_seed         = individual.apple_seed
                starting_direction = individual.starting_direction

                # If the individual is still alive, they survive
                if lifespan > 0:
                    s = Snake(board_size, chromosome=indv_network,
                              hidden_layer_units=hidden_layer_units,
                              hidden_activation=hidden_activation,
                              output_activation=output_activation,
                              lifespan=lifespan,
                              apple_and_self_vision=apple_and_self_vision)
                    next_pop.append(s)


        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, num_individuals=2)

            L = len(p1.network.layer_units)

            # Each Weight 'W' and bias 'b' are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.layers[l].W
                p1_b_l = p1.network.layers[l].b

                p2_W_l = p2.network.layers[l].W
                p2_b_l = p2.network.layers[l].b

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                p1.network.layers[l].W = c1_W_l
                p1.network.layers[l].b = c1_b_l

                p2.network.layers[l].W = c2_W_l
                p2.network.layers[l].b = c2_b_l

                # Clip to [-1, 1]
                np.clip(p1.network.layers[l].W, -1, 1, out=p1.network.layers[l].W)
                np.clip(p1.network.layers[l].b, -1, 1, out=p1.network.layers[l].b)

                np.clip(p2.network.layers[l].W, -1, 1, out=p2.network.layers[l].W)
                np.clip(p2.network.layers[l].b, -1, 1, out=p2.network.layers[l].b)

            # Create children from chromosomes generated above
            c1 = Snake(p1.board_size, chromosome=p1.network, hidden_layer_units=p1.hidden_layer_units,
                       hidden_activation=p1.hidden_activation, output_activation=p1.output_activation,
                       lifespan=self.config['lifespan'])
            c2 = Snake(p2.board_size, chromosome=p2.network, hidden_layer_units=p2.hidden_layer_units,
                       hidden_activation=p2.hidden_activation, output_activation=p2.output_activation,
                       lifespan=self.config['lifespan'])

            # Add children to the next generation
            next_pop.extend([c1, c2])
        
        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    

    def _increment_generation(self):
        self.current_generation += 1
        self.network_window.label_generation.setText('Generation: ' + str(self.current_generation + 1))


    

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Weight 'W' and bias 'b' are treated as chromosome.
        '''
        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)
        
        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    


    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if self.config['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / np.sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            child1_weights = gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            child2_weights = gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            child1_bias = gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            child2_bias = gaussian_mutation(child2_bias, mutation_rate, scale=scale)
        
        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            child1_weights = random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            child2_weights = random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            child1_bias = random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            child2_bias = random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')


        return child1_weights, child2_weights, child1_bias, child2_bias



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())