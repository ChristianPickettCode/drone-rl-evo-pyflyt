import random
import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401
from PyFlyt.gym_envs import FlattenWaypointEnv
from deap import base, creator, tools, algorithms
from tensorboardX import SummaryWriter
import pickle

# Create the environment without rendering for training
train_env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode=None)
train_env = FlattenWaypointEnv(train_env, context_length=1)

# Create the environment with rendering for evaluation
eval_env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode="human")
eval_env = FlattenWaypointEnv(eval_env, context_length=1)

# Define the evaluation function


def evaluate(individual):
    obs = train_env.reset()
    total_reward = 0
    done = False
    while not done:
        action = np.array(individual)
        obs, reward, done, truncation, _ = train_env.step(action)
        total_reward += reward
        done = done or truncation
    return total_reward,


# Define the evolutionary algorithm parameters
# Size of the individual (action space size)
IND_SIZE = train_env.action_space.shape[0]
POP_SIZE = 100  # Population size
N_GEN = 100  # Number of generations

# Create the toolbox with the right parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform,
                 train_env.action_space.low[0], train_env.action_space.high[0])
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize the population
pop = toolbox.population(n=POP_SIZE)

# Set up TensorBoard logging
log_dir = "./tensorboard_logs/"
writer = SummaryWriter(log_dir)

# Define a custom evolutionary algorithm with logging


def ea_with_logging(population, toolbox, cxpb, mutpb, ngen, stats=None,
                    halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Log statistics to TensorBoard
        for key, value in record.items():
            writer.add_scalar(key, value, gen)

    return population, logbook


# Statistics to log
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Hall of fame to keep track of the best individual
hof = tools.HallOfFame(1)

# Run the evolutionary algorithm with logging
pop, logbook = ea_with_logging(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN,
                               stats=stats, halloffame=hof, verbose=True)

# Find the best individual
best_ind = hof[0]
print("Best individual is: %s\nwith fitness: %s" %
      (best_ind, best_ind.fitness.values))

# Save the best individual
with open("best_individual.pkl", "wb") as f:
    pickle.dump(best_ind, f)

# Test the best individual with rendering
obs = eval_env.reset()
done = False
while not done:
    action = np.array(best_ind)
    obs, reward, done, truncation, info = eval_env.step(action)
    done = done or truncation
    print(f"Observation: {obs}, Reward: {reward}")

# Close the environments
train_env.close()
eval_env.close()

# Close the TensorBoard writer
writer.close()
