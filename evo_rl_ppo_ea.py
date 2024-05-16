import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from deap import base, creator, tools, algorithms
from tensorboardX import SummaryWriter
import PyFlyt.gym_envs  # noqa: F401
from PyFlyt.gym_envs import FlattenWaypointEnv
import pickle

# Create the environment


def create_env(render_mode=None):
    env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode=render_mode)
    return FlattenWaypointEnv(env, context_length=1)

# Define the PPO training function


def train_ppo(env, hyperparams):
    print(f"Training PPO with hyperparameters: {hyperparams}")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        tensorboard_log="./ppo_pyflyt_tensorboard/"
    )
    model.learn(total_timesteps=10000)
    print("Training complete.")
    return model

# Define the fitness evaluation function


def evaluate_hyperparams(individual, env, eval_env):
    hyperparams = {
        "learning_rate": individual[0],
        "n_steps": int(individual[1]),
        "batch_size": int(individual[2]),
        "n_epochs": int(individual[3]),
        "gamma": individual[4],
        "gae_lambda": individual[5],
        "clip_range": individual[6],
        "ent_coef": individual[7],
    }
    model = train_ppo(env, hyperparams)

    print("Evaluating PPO model.")
    rewards = []
    obs = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncation, info = eval_env.step(action)
        rewards.append(reward)
    total_reward = np.sum(rewards)
    print(f"Evaluation complete. Total reward: {total_reward}")
    return total_reward,

# Set up DEAP


def setup_deap(env, eval_env):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform,
                     0.0001, 0.01)  # learning_rate
    toolbox.register("attr_int_nsteps", random.randint, 16, 2048)  # n_steps
    toolbox.register("attr_int_batch", random.randint, 16, 256)  # batch_size
    toolbox.register("attr_int_epochs", random.randint, 1, 30)  # n_epochs
    toolbox.register("attr_float_gamma", random.uniform, 0.8, 0.9999)  # gamma
    toolbox.register("attr_float_gael", random.uniform, 0.8, 1.0)  # gae_lambda
    toolbox.register("attr_float_clip", random.uniform, 0.1, 0.4)  # clip_range
    toolbox.register("attr_float_ent", random.uniform,
                     0.0001, 0.01)  # ent_coef

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_int_nsteps, toolbox.attr_int_batch, toolbox.attr_int_epochs,
                      toolbox.attr_float_gamma, toolbox.attr_float_gael, toolbox.attr_float_clip, toolbox.attr_float_ent), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_hyperparams,
                     env=env, eval_env=eval_env)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

# Evolutionary Algorithm with Logging


def ea_with_logging(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    print("Evaluating initial population.")
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        print(f"Generation {gen}")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        print("Evaluating offspring.")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

# Main function


def main():
    global writer
    print("Creating training and evaluation environments.")
    train_env = create_env(render_mode=None)
    eval_env = create_env(render_mode='human')

    print("Setting up DEAP toolbox.")
    toolbox = setup_deap(train_env, eval_env)

    pop = toolbox.population(n=20)  # Set population size

    log_dir = "./tensorboard_logs/"
    writer = SummaryWriter(log_dir)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    print("Starting evolutionary algorithm.")
    pop, logbook = ea_with_logging(
        pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    print("Best individual found: %s\nwith fitness: %s" %
          (best_ind, best_ind.fitness.values))

    # Save the best hyperparameters
    print("Saving the best hyperparameters.")
    with open("best_hyperparams.pkl", "wb") as f:
        pickle.dump(best_ind, f)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
