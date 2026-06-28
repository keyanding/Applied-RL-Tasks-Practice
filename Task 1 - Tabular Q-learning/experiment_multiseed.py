import numpy as np
import gymnasium as gym

from q_learning import train_q_learning


def evaluate(Q, episodes=5000, seed=None):
    """
    Evaluate a greedy policy in slippery FrozenLake.

    Args:
        Q:
            Learned Q-table.

        episodes:
            Number of evaluation episodes.
            We use many episodes because the environment is stochastic.

        seed:
            Optional seed for the evaluation environment.
            This helps make the reported result reproducible.

    Returns:
        success_rate:
            Fraction of episodes that reached the goal.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    if seed is not None:
        env.action_space.seed(seed)

    successes = 0

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        while not done:
            action = int(np.argmax(Q[state]))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            successes += 1

    env.close()
    return successes / episodes


def run_multiseed_experiment():
    """
    Run Q-learning across multiple random seeds.

    Why this matters:
        In RL, one training run is not reliable evidence.
        Different random seeds can produce different trajectories,
        different Q-tables, and different final greedy policies.

    We vary:
        number of training episodes

    We repeat:
        multiple random seeds

    We report:
        mean, std, min, and max success rate
    """
    episode_settings = [3000, 10000, 30000, 100000]
    seeds = [0, 1, 2, 3, 4]

    for episodes in episode_settings:
        success_rates = []

        for seed in seeds:
            Q = train_q_learning(
                episodes=episodes,
                alpha=0.1,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                seed=seed,
            )

            # Use a different but deterministic eval seed.
            success_rate = evaluate(Q, episodes=5000, seed=1000 + seed)
            success_rates.append(success_rate)

        success_rates = np.array(success_rates)

        print("=" * 60)
        print(f"Training episodes: {episodes}")
        print(f"Success rates: {success_rates}")
        print(f"Mean: {success_rates.mean():.3f}")
        print(f"Std:  {success_rates.std():.3f}")
        print(f"Min:  {success_rates.min():.3f}")
        print(f"Max:  {success_rates.max():.3f}")


if __name__ == "__main__":
    run_multiseed_experiment()