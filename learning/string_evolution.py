"""
A simple genetic algorithm to evolve a target string.
"""

import numpy as np

# Parameters
TARGET_STRING = "HELLO, WORLD!"
POPULATION_SIZE = 100
MUTATION_RATE = 0.01

# Convert target string to a list of ASCII values
TARGET_GENOME = [ord(c) for c in TARGET_STRING]
GENOME_LENGTH = len(TARGET_GENOME)

# Fitness function
def fitness(genome):
    return np.sum(np.abs(np.array(genome) - np.array(TARGET_GENOME)))

# Create initial population
population = np.random.randint(32, 127, (POPULATION_SIZE, GENOME_LENGTH))

generation = 0
while True:
    generation += 1

    # Calculate fitness (lower is better)
    fitness_scores = np.array([fitness(ind) for ind in population])

    # Find the best individual
    best_index = np.argmin(fitness_scores)
    best_fitness = fitness_scores[best_index]
    best_individual = population[best_index]
    best_string = "".join([chr(c) for c in best_individual])

    print(f"Generation {generation}: Best fitness = {best_fitness}, String = {best_string}")

    if best_fitness == 0:
        print("Target string evolved!")
        break

    # Select parents (tournament selection)
    parents = np.zeros_like(population)
    for i in range(POPULATION_SIZE):
        tournament_indices = np.random.choice(range(POPULATION_SIZE), 2)
        winner_index = tournament_indices[np.argmin(fitness_scores[tournament_indices])]
        parents[i] = population[winner_index]

    # Create next generation (crossover)
    offspring = np.zeros_like(population)
    for i in range(0, POPULATION_SIZE, 2):
        parent1, parent2 = parents[i], parents[i+1]
        crossover_point = np.random.randint(1, GENOME_LENGTH)
        offspring[i, :crossover_point] = parent1[:crossover_point]
        offspring[i, crossover_point:] = parent2[crossover_point:]
        offspring[i+1, :crossover_point] = parent2[:crossover_point]
        offspring[i+1, crossover_point:] = parent1[crossover_point:]

    # Mutation
    mutation_mask = np.random.rand(*offspring.shape) < MUTATION_RATE
    offspring[mutation_mask] = np.random.randint(32, 127, np.sum(mutation_mask))

    population = offspring
