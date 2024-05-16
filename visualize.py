import matplotlib.pyplot as plt
import neat
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = [np.mean(fitness)
                   for fitness in statistics.get_fitness_mean()]
    stdev_fitness = [np.std(fitness)
                     for fitness in statistics.get_fitness_stdev()]

    plt.figure()
    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.plot(generation, stdev_fitness, 'g-', label="std dev")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.yscale('log')

    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def plot_species(statistics, view=False, filename='speciation.svg'):
    generation = range(len(statistics.most_fit_genomes))
    num_species = [len(species) for species in statistics.get_species_sizes()]

    plt.figure()
    plt.plot(generation, num_species, 'b-')

    plt.title("Speciation")
    plt.xlabel("Generations")
    plt.ylabel("Number of Species")
    plt.grid()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()
