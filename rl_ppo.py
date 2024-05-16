import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import PyFlyt.gym_envs  # noqa: F401
from PyFlyt.gym_envs import FlattenWaypointEnv


def create_env(render_mode=None):
    env = gym.make("PyFlyt/QuadX-Waypoints-v1", render_mode=render_mode)
    return FlattenWaypointEnv(env, context_length=1)


def wrap_env(env):
    return DummyVecEnv([lambda: env])


def create_model(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./ppo_pyflyt_tensorboard/"
    )


def setup_callbacks(eval_env):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path='./checkpoints/', name_prefix='ppo_model')
    eval_callback = EvalCallback(
        eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=5000)
    return [checkpoint_callback, eval_callback]


def train_model(model, callbacks):
    model.learn(total_timesteps=500000, tb_log_name="PPO", callback=callbacks)
    model.save("ppo_pyflyt_quadx_waypoint")


def load_model():
    return PPO.load("ppo_pyflyt_quadx_waypoint")


def test_model(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Observation: {obs}, Reward: {reward}")


def main():
    # Create and wrap the training environment
    # train_env = create_env(render_mode=None)
    # print("Original observation space shape:",
    #       train_env.observation_space.shape)
    # train_env = wrap_env(train_env)
    # print("Wrapped observation space shape:",
    #       train_env.observation_space.shape)

    # Create and wrap the evaluation environment
    eval_env = create_env(render_mode="human")
    eval_env = wrap_env(eval_env)

    # Create the PPO model
    # model = create_model(train_env)

    # Setup callbacks
    # callbacks = setup_callbacks(eval_env)

    # Train the model
    # train_model(model, callbacks)

    # Load the trained model
    model = load_model()

    # Test the trained model
    test_model(model, eval_env)

    # Close the environment
    eval_env.close()


if __name__ == "__main__":
    main()
