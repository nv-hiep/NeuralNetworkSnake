import numpy as np
import random
from typing import List
from .population import Population
from .individual import Individual


def elitism_selection(population: Population,
                      num_individuals: int) -> List[Individual]:
    '''
    Select the individual with highest fitness
    '''
    chromosomes = sorted(population.individuals, key = lambda individual: individual.fitness, reverse=True)
    return chromosomes[:num_individuals]



def roulette_wheel_selection(population: Population,
                             num_individuals: int) -> List[Individual]:
    '''
    Consider a circular wheel. The wheel is divided into n pies, where n is the number of individuals in the population.
    Each individual gets a portion of the circle which is proportional to its fitness value.

    A fixed point is chosen on the wheel circumference as shown and the wheel is rotated.
    The region of the wheel which comes in front of the fixed point is chosen as the parent.
    For the second parent, the same process is repeated.
    '''
    selection = []
    wheel = sum(individual.fitness for individual in population.individuals)
    for _ in range(num_individuals):
        pick = random.uniform(0, wheel)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current > pick:
                selection.append(individual)
                break

    return selection

def tournament_selection(population: Population,
                        num_individuals: int, tournament_size: int) -> List[Individual]:
    '''
    1. select K individuals from the population at random
    2. select the best out of these to become a parent
    '''
    selection = []
    for _ in range(num_individuals):
        tournament = np.random.choice(population.individuals, tournament_size)
        best_from_tournament = max(tournament, key = lambda individual: individual.fitness)
        selection.append(best_from_tournament)

    return selection