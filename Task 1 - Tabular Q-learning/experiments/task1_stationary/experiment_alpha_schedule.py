import gymnasium as gym
import numpy as np

from src.q_learning import render_policy, train_q_learning, train_q_learning_with_visit_decay


def evaluate(Q, episodes=5000, seed=None):
    """
    Evaluate a greedy policy in slippery FrozenLake.

    We use many episodes because the environment transition is stochastic.
    The returned success rate estimates the probability of reaching the goal
    under the learned greedy policy.
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
    Print the learned greedy policy as a 4x4 grid.
    """
    policy = render_policy(Q)
    for row in policy:
        print(" ".join(row))


def compare_alpha_schedules():
    """
    Compare two Q-learning variants:

    1. Constant alpha:
        Q-values keep changing with the same update size forever.

    2. Visit-count decay alpha:
        Q-values change less as a state-action pair is visited more often.

    Goal:
        Check whether visit decay reduces bad-seed failures in slippery FrozenLake.
    """
    episodes = 3000
    seeds = [0, 1, 2, 3, 4]

    constant_rates = []
    decay_rates = []

    for seed in seeds:
        Q_constant = train_q_learning(
            episodes=episodes,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        Q_decay, visit_counts = train_q_learning_with_visit_decay(
            episodes=episodes,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        constant_rate = evaluate(Q_constant, episodes=5000, seed=1000 + seed)
        decay_rate = evaluate(Q_decay, episodes=5000, seed=1000 + seed)

        constant_rates.append(constant_rate)
        decay_rates.append(decay_rate)

        print("=" * 60)
        print(f"Seed: {seed}")

        print(f"Constant alpha success rate: {constant_rate:.3f}")
        print("Constant alpha policy:")
        print_policy(Q_constant)

        print(f"\nVisit-decay alpha success rate: {decay_rate:.3f}")
        print("Visit-decay alpha policy:")
        print_policy(Q_decay)

        print("\nVisit counts at key states for decay version:")
        for state in [8, 9, 10, 13, 14]:
            print(f"state {state}: {visit_counts[state]}")

        print("\nQ-values at state 9:")
        print("constant:", Q_constant[9])
        print("decay:   ", Q_decay[9])

    constant_rates = np.array(constant_rates)
    decay_rates = np.array(decay_rates)

    print("=" * 60)
    print("Summary")
    print(f"Constant alpha rates: {constant_rates}")
    print(f"Constant alpha mean/std: {constant_rates.mean():.3f} / {constant_rates.std():.3f}")

    print(f"Visit-decay rates: {decay_rates}")
    print(f"Visit-decay mean/std: {decay_rates.mean():.3f} / {decay_rates.std():.3f}")


if __name__ == "__main__":
    compare_alpha_schedules()