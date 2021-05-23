import sys
import os
import json
import random

import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, List, Any
from fractions import Fraction
from collections import deque

from helper.tools import Slope, Point
from helper.config import config
from helper.const import *
from network.neural_network import Network, linear, sigmoid, tanh, relu, leaky_relu, get_activation_fcn

from GA.individual import Individual



class Vision(object):
    __slots__ = ('dist_to_wall', 'dist_to_apple', 'dist_to_self')
    def __init__(self,
                 dist_to_wall: Union[float, int],
                 dist_to_apple: Union[float, int],
                 dist_to_self: Union[float, int]
                 ):
        self.dist_to_wall  = float(dist_to_wall)
        self.dist_to_apple = float(dist_to_apple)
        self.dist_to_self  = float(dist_to_self)

class DrawableVision(object):
    __slots__ = ('wall_location', 'apple_location', 'self_location')
    def __init__(self,
                wall_location: Point,
                apple_location: Optional[Point] = None,
                self_location: Optional[Point] = None,
                ):
        self.wall_location = wall_location
        self.apple_location = apple_location
        self.self_location = self_location


class Snake(Individual):
    def __init__(self,
                 board_size: int,
                 chromosome: Optional[list] = None,
                 start_position: Optional[Point] = None, 
                 apple_seed: Optional[int] = None,
                 initial_velocity: Optional[str] = None,
                 starting_direction: Optional[str] = None,
                 hidden_layer_units: Optional[List[int]] = [16, 8],
                 hidden_activation: Optional[str] = 'relu',
                 output_activation: Optional[str] = 'sigmoid',
                 lifespan: Optional[Union[int, float]] = np.inf,
                 apple_and_self_vision: Optional[str] = 'binary'
                 ):

        self.lifespan = lifespan
        self.apple_and_self_vision = apple_and_self_vision.lower()  # binary or distance
        self.score    = 0                                           # score... from awards and penalties
        self._fitness = 0                                           # Overall fitness
        self._frames  = 0                                           # Number of frames that the snake has been alive
        self._frames_since_last_apple = 0
        self.possible_directions = ('u', 'd', 'l', 'r')

        self.board_size         = board_size
        self.hidden_layer_units = hidden_layer_units

        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # if start_positionition is not defined, then initiate THE HEAD of a snake of length = 3
        # (so, the box will be within 2 (e.g: [0, 1, 2]) ->  box-3 (e.g: [box-3, box-2, box-1]))
        if not start_position:
            x = random.randint(2, self.board_size - 3)
            y = random.randint(2, self.board_size - 3)

            start_position = Point(x, y)
        self.start_position = start_position


        self._vision_type = VISION_DICT[ config['vision_type'] ]        # set of slopes, config['vision_type'] = 4/8/16 directions
        self._vision: List[Vision] = [None] * len(self._vision_type)
        # This is just used so I can draw and is not actually used in the NN
        self._drawable_vision: List[DrawableVision] = [None] * len(self._vision_type)

        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = len(self._vision_type) * 3 + 4 + 4 #@TODO: Add one-hot back in 
        self.vision_as_array: np.ndarray = np.zeros((num_inputs, 1))
        self.network_model = [num_inputs]                          # Inputs
        self.network_model.extend(self.hidden_layer_units)  # Hidden layers
        self.network_model.append(NUM_OUTPUTS)                     # 4 outputs, ['u', 'd', 'l', 'r']

        

        # If chromosome is set, take it
        # otherwise, initiate a network with layers of random weights/biases
        if chromosome:
            self.network = chromosome
        else:
            self.network = Network(self.network_model,
                                   self.hidden_activation,
                                   self.output_activation)
            

        # For creating the next apple
        if apple_seed is None:
            apple_seed = np.random.randint(-1_0000_000, 1_0000_000)
        self.apple_seed = apple_seed  # Only needed for saving/loading replay


        self.apple_location = None
        if starting_direction:
            starting_direction = starting_direction[0].lower()
        else:
            starting_direction = self.possible_directions[random.randint(0, 3)]

        self.starting_direction = starting_direction  # Only needed for saving/loading replay
        self.init_snake(self.starting_direction)
        
        self.initial_velocity = initial_velocity
        self.init_velocity(self.starting_direction, self.initial_velocity)
        self.generate_apple()

    

    @property
    def fitness(self):
        return self._fitness
    
    

    def calculate_fitness(self):
        # Give positive minimum fitness for roulette wheel selection
        # _frames: Number of frames that the snake has been alive
        self._fitness = self.score - (self._frames**2)
        self._fitness = max(self._fitness, 0.1)



    def update(self) -> bool:
        if self.is_alive:
            self._frames += 1       # Number of frames that the snake has been alive
            self.look()
            self.network._forward_prop(self.vision_as_array)  # input array : self.vision_as_array
            self.direction = self.possible_directions[np.argmax(self.network.out)]
            return True
        
        return False

    


    def look(self):
        # Look all around
        # At a position, look around in 4/8/16 directions
        for i, slope in enumerate(self._vision_type): # Set of slopes ( (run, rise) )
            # Look around in 4/8/16 directions
            vision, drawable_vision = self.look_in_direction(slope)
            self._vision[i] = vision
            self._drawable_vision[i] = drawable_vision
        
        # Update the input array
        self.vision_as_input_array()


    


    def look_in_direction(self, slope: Slope) -> Tuple[Vision, DrawableVision]:
        '''
        At a position, look around in a specific direction
        Slope: (rise, run)
        '''
        dist_to_wall  = None
        dist_to_apple = np.inf
        dist_to_self  = np.inf

        wall_location  = None
        apple_location = None
        self_location  = None

        position = self.snake_array[0].copy() # snake's head: deque(snake), snake = [head, body, tail]
        # position = Point(0,0)
        distance = 1.
        total_distance = 0.

        # Can't start by looking at yourself
        position.x += slope.run
        position.y += slope.rise
        
        total_distance += distance
        
        body_found = False  # Only need to find the first occurance since it's the closest
        food_found = False  # Although there is only one food, stop looking once you find it

        # Keep going until the position is out of bounds
        while self.is_within_board(position):
            if not body_found and self.is_body_location(position):
                dist_to_self = total_distance
                self_location = position.copy()
                body_found = True
            if not food_found and self.is_apple_location(position):
                dist_to_apple = total_distance
                apple_location = position.copy()
                food_found = True

            wall_location = position
            position.x += slope.run
            position.y += slope.rise
            total_distance += distance
        
        assert(total_distance != 0.)


        # @TODO: May need to adjust numerator in case of VISION_16 since step size isn't always going to be on a tile
        dist_to_wall = 1. / total_distance

        if self.apple_and_self_vision == 'binary':
            dist_to_apple = 1. if dist_to_apple != np.inf else 0.
            dist_to_self = 1. if dist_to_self != np.inf else 0.

        elif self.apple_and_self_vision == 'distance':
            dist_to_apple = 1. / dist_to_apple
            dist_to_self = 1. / dist_to_self

        vision = Vision(dist_to_wall, dist_to_apple, dist_to_self)
        drawable_vision = DrawableVision(wall_location, apple_location, self_location)
        return (vision, drawable_vision)

    


    def vision_as_input_array(self) -> None:
        # Split _vision into np array where rows [0-2] are _vision[0].dist_to_wall, 
        # _vision[0].dist_to_apple,
        # _vision[0].dist_to_self,
        # rows [3-5] are _vision[1].dist_to_wall,
        # _vision[1].dist_to_apple,
        # _vision[1].dist_to_self, etc. etc. etc.
        for va_index, v_index in zip(range(0, len(self._vision) * 3, 3), range(len(self._vision))):
            vision = self._vision[v_index]
            self.vision_as_array[va_index, 0]     = vision.dist_to_wall
            self.vision_as_array[va_index + 1, 0] = vision.dist_to_apple
            self.vision_as_array[va_index + 2, 0] = vision.dist_to_self

        i = len(self._vision) * 3  # Start at the end

        direction = self.direction[0].lower()
        # One-hot encode direction
        direction_one_hot = np.zeros((len(self.possible_directions), 1))
        direction_one_hot[self.possible_directions.index(direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = direction_one_hot

        i += len(self.possible_directions)

        # One-hot tail direction
        tail_direction_one_hot = np.zeros((len(self.possible_directions), 1))
        tail_direction_one_hot[self.possible_directions.index(self.tail_direction), 0] = 1
        self.vision_as_array[i: i + len(self.possible_directions)] = tail_direction_one_hot

    


    def is_within_board(self, position: Point) -> bool:
        '''
        Check if the snake is still within the board box
        '''
        return position.x >= 0 and position.y >= 0 and\
               position.x < self.board_size and position.y < self.board_size

    


    def generate_apple(self) -> None:
        width = height = self.board_size # Square board

        # Find all possible points where the snake is not currently
        possibilities = [divmod(i, height) for i in range(width * height) if divmod(i, height) not in self.body_locations]
        # same as: possibilities = [(x,y) for x in range(width) for y in range(height)]

        if possibilities:
            x,y = random.choice(possibilities)
            self.apple_location = Point(x, y)
        else:
            print('You win!')
            pass

    def init_snake(self, starting_direction: str) -> None:
        '''
        Initialize the snake.
        starting_direction: ('u', 'd', 'l', 'r')
            direction that the snake should start facing. Whatever the direction is, the head
            of the snake will begin pointing that way.
        '''

        # initialize position of the head: ramdom in [2, self.board_size - 3]
        head = self.start_position
        
        # Body is below
        if starting_direction == 'u':
            snake = [head, Point(head.x, head.y + 1), Point(head.x, head.y + 2)]
        # Body is above
        elif starting_direction == 'd':
            snake = [head, Point(head.x, head.y - 1), Point(head.x, head.y - 2)]
        # Body is to the right
        elif starting_direction == 'l':
            snake = [head, Point(head.x + 1, head.y), Point(head.x + 2, head.y)]
        # Body is to the left
        elif starting_direction == 'r':
            snake = [head, Point(head.x - 1, head.y), Point(head.x - 2, head.y)]

        self.snake_array    = deque(snake)
        self.body_locations = set(snake)
        self.is_alive = True

    


    

    def move(self) -> bool:
        if not self.is_alive:
            return False

        direction = self.direction[0].lower()
        # Is the direction valid?
        if direction not in self.possible_directions:
            return False
        
        # Find next position
        # tail = self.snake_array.pop()  # Pop tail since we can technically move to the tail
        head = self.snake_array[0]

        if direction == 'u':
            next_pos = Point(head.x, head.y - 1)
        elif direction == 'd':
            next_pos = Point(head.x, head.y + 1)
        elif direction == 'r':
            next_pos = Point(head.x + 1, head.y)
        elif direction == 'l':
            next_pos = Point(head.x - 1, head.y)

        # Is the next position we want to move valid?
        if self.is_valid(next_pos):
            # Tail
            if next_pos == self.snake_array[-1]:
                # Pop tail and add next_pos (same as tail) to front
                # No need to remove tail from body_locations since it will go back in anyway
                self.snake_array.pop()
                self.snake_array.appendleft(next_pos) 

                # No need to do with self.body_locations
            
            # Eat the apple
            elif next_pos == self.apple_location:
                self.score += 1000                      # If snake eats an apple, award 5000 points
                self._frames_since_last_apple = 0
                
                # Move head
                self.snake_array.appendleft(next_pos)
                self.body_locations.update({next_pos})
                
                # Don't remove tail since the snake grew
                self.generate_apple()
            
            # Normal movement
            else:
                # Move head
                self.snake_array.appendleft(next_pos)
                self.body_locations.update({next_pos})

                # Remove tail
                tail = self.snake_array.pop()

                # Remove the items that are present in both sets, AND insert the items that is not present in both sets:
                self.body_locations.symmetric_difference_update({tail})  # symmetric_difference_update uses a set as arg

            # Figure out which direction the tail is moving
            p2 = self.snake_array[-2]
            p1 = self.snake_array[-1]
            diff = p2 - p1
            if diff.x < 0:
                self.tail_direction = 'l'
            elif diff.x > 0:
                self.tail_direction = 'r'
            elif diff.y > 0:
                self.tail_direction = 'd'
            elif diff.y < 0:
                self.tail_direction = 'u'

            self._frames_since_last_apple += 1
            # you may want to change this
            if self._frames_since_last_apple > self.board_size * self.board_size:
                self.is_alive = False
                self.score -= 100.        # If snake is dead, penalize by 150 points
                return False

            return True
        else:
            self.is_alive = False
            self.score -= 100.            # If snake is dead, penalize by 150 points
            return False

    def is_apple_location(self, position: Point) -> bool:
        return position == self.apple_location

    def is_body_location(self, position: Point) -> bool:
        return position in self.body_locations

    def is_valid(self, position: Point) -> bool:
        """
        Determine whether a given position is valid.
        Return True if the position is on the board and does not intersect the snake.
        Return False otherwise
        """
        if (position.x < 0) or (position.x > self.board_size - 1):
            return False
        if (position.y < 0) or (position.y > self.board_size - 1):
            return False

        # position == tail
        if position == self.snake_array[-1]:
            return True
        # If the position is a body location, not valid.
        # @NOTE: body_locations will contain tail, so need to check tail first
        elif position in self.body_locations:
            return False
        # Otherwise you good
        else:
            return True

    def init_velocity(self, starting_direction, initial_velocity: Optional[str] = None) -> None:
        if initial_velocity:
            self.direction = initial_velocity[0].lower()
        # Whichever way the starting_direction is
        else:
            self.direction = starting_direction

        # Tail starts moving the same direction
        self.tail_direction = self.direction