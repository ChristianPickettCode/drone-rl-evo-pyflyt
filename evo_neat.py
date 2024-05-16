import neat
import numpy as np
import gymnasium as gym
import PyFlyt.gym_envs  # noqa: F401
from PyFlyt.gym_envs import FlattenWaypointEnv
import pickle
import visualize  # Assuming you have visualization utilities


def create_env(render_mode=None):
    env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode=render_mode)
    return FlattenWaypointEnv(env, context_length=1)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = evaluate_genome(net)


def evaluate_genome(net):
    env = create_env(render_mode=None)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = net.activate(obs)
        action = np.array(action)  # Convert action to numpy array
        obs, reward, done, truncation, _ = env.step(action)
        total_reward += reward
        done = done or truncation
    env.close()
    return total_reward


def run_neat(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        5, filename_prefix='checkpoint/checkpoint-'))

    winner = p.run(eval_genomes, 150)  # Run for up to 150 generations

    with open("neat_winner.pkl", "wb") as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    return winner, config


def test_winner(winner, config):
    env = create_env(render_mode="human")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    obs, _ = env.reset()
    done = False
    while not done:
        action = net.activate(obs)
        action = np.array(action)  # Convert action to numpy array
        obs, reward, done, truncation, info = env.step(action)
        done = done or truncation
        print(f"Observation: {obs}, Reward: {reward}")
    env.close()


if __name__ == "__main__":
    config_path = "neat-config.txt"
    winner, config = run_neat(config_path)
    test_winner(winner, config)
