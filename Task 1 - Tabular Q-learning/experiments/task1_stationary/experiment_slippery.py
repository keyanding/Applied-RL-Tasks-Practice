import gymnasium as gym
import numpy as np

from src.q_learning import render_policy, train_q_learning


def evaluate(Q, episodes=5000):
    """
    Evaluate a learned greedy policy in slippery FrozenLake.

    We use many evaluation episodes because the environment is stochastic.
    With only 100 episodes, the measured success rate can be noisy.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    successes = 0

    for _ in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            # During evaluation, we use the greedy policy.
            # No epsilon exploration is used here.
            action = int(np.argmax(Q[state]))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            successes += 1

    env.close()
    return successes / episodes


def run_episode_sweep():
    """
    Train Q-learning with different training budgets.

    Goal:
        Check whether poor slippery performance is caused by insufficient training.

    This is our first controlled RL experiment:
        vary one factor: number of training episodes
        keep other hyperparameters fixed
    """
    episode_settings = [3000, 10000, 30000, 100000]

    for episodes in episode_settings:
        Q = train_q_learning(
            episodes=episodes,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
        )

        success_rate = evaluate(Q, episodes=5000)

        print("=" * 50)
        print(f"Training episodes: {episodes}")
        print(f"Eval success rate: {success_rate:.3f}")

        print("Learned policy:")
        policy = render_policy(Q)
        for row in policy:
            print(" ".join(row))


if __name__ == "__main__":
    run_episode_sweep()