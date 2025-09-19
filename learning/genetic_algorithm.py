"""
A simple genetic algorithm to solve the "onemax" problem (maximizing the number of ones in a bitstring).
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
POPULATION_SIZE = 100
GENOME_LENGTH = 50
MUTATION_RATE = 0.01
N_GENERATIONS = 100

# Fitness function (onemax)
def fitness(genome):
    return np.sum(genome)

# Create initial population
population = np.random.choice([0, 1], (POPULATION_SIZE, GENOME_LENGTH))

best_fitness_history = []

for generation in range(N_GENERATIONS):
    # Calculate fitness for each individual
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_fitness_history.append(np.max(fitness_scores))

    # Select parents (roulette wheel selection)
    fitness_sum = np.sum(fitness_scores)
    selection_probs = fitness_scores / fitness_sum
    parent_indices = np.random.choice(range(POPULATION_SIZE), size=POPULATION_SIZE, p=selection_probs)
    parents = population[parent_indices]

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
    offspring[mutation_mask] = 1 - offspring[mutation_mask]

    population = offspring

print("Genetic algorithm finished.")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history)
plt.title("Genetic Algorithm Performance")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.savefig("genetic_algorithm.png")
print("Genetic algorithm plot saved to genetic_algorithm.png")
