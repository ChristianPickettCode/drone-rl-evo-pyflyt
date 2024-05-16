import random
import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401
from PyFlyt.gym_envs import FlattenWaypointEnv
from deap import base, creator, tools, algorithms
from tensorboardX import SummaryWriter
import pickle


def create_env(render_mode=None):
    env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode=render_mode)
    return FlattenWaypointEnv(env, context_length=1)


def linear_policy(obs, weights, biases):
    return np.dot(obs, weights) + biases


def evaluate(individual, env, obs_size, IND_SIZE):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    weights = np.array(
        individual[:obs_size * IND_SIZE]).reshape(obs_size, IND_SIZE)
    biases = np.array(individual[obs_size * IND_SIZE:])
    while not done:
        action = linear_policy(obs, weights, biases)
        obs, reward, done, truncation, _ = env.step(action)
        total_reward += reward
        done = done or truncation
    return total_reward,


def setup_deap_toolbox(obs_size, IND_SIZE, evaluate_func):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=(obs_size * IND_SIZE + IND_SIZE))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select_elite", tools.selBest)  # Elitism

    return toolbox


def evaluate_population(population, toolbox):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return invalid_ind


def mate_and_mutate_population(offspring, toolbox, cxpb, mutpb):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values


def ea_with_logging(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = evaluate_population(population, toolbox)
    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        mate_and_mutate_population(offspring, toolbox, cxpb, mutpb)
        invalid_ind = evaluate_population(offspring, toolbox)

        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        for key, value in record.items():
            writer.add_scalar(key, value, gen)

    return population, logbook


def save_best_individual(best_ind, filename="best_individual.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(best_ind, f)


def test_best_individual(best_ind, env, obs_size, IND_SIZE):
    weights = np.array(best_ind[:obs_size * IND_SIZE]
                       ).reshape(obs_size, IND_SIZE)
    biases = np.array(best_ind[obs_size * IND_SIZE:])
    obs, _ = env.reset()
    done = False
    while not done:
        action = linear_policy(obs, weights, biases)
        obs, reward, done, truncation, info = env.step(action)
        done = done or truncation
        print(f"Observation: {obs}, Reward: {reward}")


def main():
    global writer
    train_env = create_env(render_mode=None)
    eval_env = create_env(render_mode="human")

    obs_size = train_env.observation_space.shape[0]
    IND_SIZE = train_env.action_space.shape[0]
    POP_SIZE = 50
    N_GEN = 50

    toolbox = setup_deap_toolbox(obs_size, IND_SIZE, lambda ind: evaluate(
        ind, train_env, obs_size, IND_SIZE))

    pop = toolbox.population(n=POP_SIZE)

    log_dir = "./tensorboard_logs/"
    writer = SummaryWriter(log_dir)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    pop, logbook = ea_with_logging(
        pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN, stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print("Best individual is: %s\nwith fitness: %s" %
          (best_ind, best_ind.fitness.values))

    save_best_individual(best_ind)

    test_best_individual(best_ind, eval_env, obs_size, IND_SIZE)

    train_env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    main()
