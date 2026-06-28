import gymnasium as gym
import numpy as np

from src.q_learning import (
    render_policy,
    train_q_learning,
    train_q_learning_with_sqrt_decay,
    train_q_learning_with_visit_decay,
)


def evaluate(Q, episodes=5000, seed=None):
    """
    Evaluate a greedy policy in slippery FrozenLake.

    Because the environment is stochastic, we use many episodes to estimate
    the policy's success probability.
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


def print_policy(Q):
    """
    Print the learned greedy policy as a readable 4x4 grid.
    """
    policy = render_policy(Q)
    for row in policy:
        print(" ".join(row))


def compare_three_alpha_schedules():
    """
    Compare three learning-rate schedules for tabular Q-learning.

    Goal:
        Understand how learning-rate choice affects finite-sample RL behavior.

    Schedules:
        constant:
            alpha = 0.1

        visit_decay:
            alpha = 1 / N(s, a)

        sqrt_decay:
            alpha = alpha0 / sqrt(N(s, a))
    """
    episodes = 3000
    seeds = [0, 1, 2, 3, 4]

    constant_rates = []
    visit_decay_rates = []
    sqrt_decay_rates = []

    for seed in seeds:
        Q_constant = train_q_learning(
            episodes=episodes,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        Q_visit_decay, visit_counts = train_q_learning_with_visit_decay(
            episodes=episodes,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        Q_sqrt_decay, sqrt_visit_counts = train_q_learning_with_sqrt_decay(
            episodes=episodes,
            alpha0=0.5,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        eval_seed = 1000 + seed

        constant_rate = evaluate(Q_constant, episodes=5000, seed=eval_seed)
        visit_decay_rate = evaluate(Q_visit_decay, episodes=5000, seed=eval_seed)
        sqrt_decay_rate = evaluate(Q_sqrt_decay, episodes=5000, seed=eval_seed)

        constant_rates.append(constant_rate)
        visit_decay_rates.append(visit_decay_rate)
        sqrt_decay_rates.append(sqrt_decay_rate)

        print("=" * 60)
        print(f"Seed: {seed}")
        print(f"Constant alpha:     {constant_rate:.3f}")
        print(f"Visit decay 1/N:    {visit_decay_rate:.3f}")
        print(f"Sqrt decay alpha0=.5: {sqrt_decay_rate:.3f}")

        if seed == 4:
            print("\nSeed 4 policy comparison:")

            print("\nConstant alpha policy:")
            print_policy(Q_constant)
            print("Q[state 9]:", Q_constant[9])

            print("\nVisit decay policy:")
            print_policy(Q_visit_decay)
            print("Q[state 9]:", Q_visit_decay[9])
            print("Q[state 3]:", Q_visit_decay[3])
            print("Q[state 14]:", Q_visit_decay[14])

            print("\nSqrt decay policy:")
            print_policy(Q_sqrt_decay)
            print("Q[state 9]:", Q_sqrt_decay[9])
            print("Q[state 3]:", Q_sqrt_decay[3])
            print("Q[state 14]:", Q_sqrt_decay[14])

    constant_rates = np.array(constant_rates)
    visit_decay_rates = np.array(visit_decay_rates)
    sqrt_decay_rates = np.array(sqrt_decay_rates)

    print("=" * 60)
    print("Summary")
    print(f"Constant alpha rates:  {constant_rates}")
    print(f"Constant mean/std:     {constant_rates.mean():.3f} / {constant_rates.std():.3f}")

    print(f"Visit decay rates:     {visit_decay_rates}")
    print(f"Visit decay mean/std:  {visit_decay_rates.mean():.3f} / {visit_decay_rates.std():.3f}")

    print(f"Sqrt decay rates:      {sqrt_decay_rates}")
    print(f"Sqrt decay mean/std:   {sqrt_decay_rates.mean():.3f} / {sqrt_decay_rates.std():.3f}")


if __name__ == "__main__":
    compare_three_alpha_schedules()