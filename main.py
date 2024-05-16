import gymnasium
import PyFlyt.gym_envs  # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v1", render_mode="human")
obs = env.reset()

termination = False
truncation = False

while not termination or truncation:
    observation, reward, termination, truncation, info = env.step(
        env.action_space.sample())
