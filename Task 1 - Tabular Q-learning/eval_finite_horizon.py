import gymnasium as gym
import numpy as np

from finite_horizon_q_learning import (
    train_finite_horizon_q_learning,
    greedy_action,
    print_policy_at_time,
)


def evaluate_finite_horizon_policy(Q, episodes=5000, horizon=100, seed=None):
    """
    Evaluate a time-dependent greedy policy.

    At each step t:
        action = argmax_a Q[t, state, a]

    This differs from Task 1 evaluation:
        Task 1 used a stationary policy:
            action = argmax_a Q[state, a]

        Task 1B uses a finite-horizon non-stationary policy:
            action = argmax_a Q[t, state, a]

    Returns:
        success_rate:
            Fraction of episodes that reached the goal within horizon steps.
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
        final_reward = 0.0

        for t in range(horizon):
            if done:
                break

            action = greedy_action(Q, t, state)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_reward = reward

        if final_reward == 1:
            successes += 1

    env.close()
    return successes / episodes


def run_multiseed_eval():
    """
    Train and evaluate finite-horizon Q-learning across multiple seeds.

    This matches our Task 1 experimental habit:
        do not trust a single seed.
    """
    seeds = [0, 1, 2, 3, 4]
    rates = []

    for seed in seeds:
        Q, visit_counts = train_finite_horizon_q_learning(
            episodes=10000,
            horizon=100,
            alpha0=0.5,
            gamma=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            seed=seed,
        )

        rate = evaluate_finite_horizon_policy(
            Q,
            episodes=5000,
            horizon=100,
            seed=1000 + seed,
        )

        rates.append(rate)

        print("=" * 60)
        print(f"Seed: {seed}")
        print(f"Finite-horizon success rate: {rate:.3f}")

        print("\nPolicy at t = 0:")
        print_policy_at_time(Q, 0)

        print("\nPolicy at t = 50:")
        print_policy_at_time(Q, 50)

        print("\nPolicy at t = 90:")
        print_policy_at_time(Q, 90)

        if seed == 4:
            print("\nDiagnostics for seed 4:")

            key_times = [0, 20, 50, 80, 90]
            key_states = [0, 4, 8, 9, 10, 13, 14]

            for t in key_times:
                print("=" * 40)
                print(f"Time t = {t}")

                for state in key_states:
                    q_values = Q[t, state]
                    counts = visit_counts[t, state]
                    best_action = int(np.argmax(q_values))

                    print(f"state {state}:")
                    print(f"  Q      = {q_values}")
                    print(f"  visits = {counts}")
                    print(f"  best   = {best_action}")

    rates = np.array(rates)

    print("=" * 60)
    print("Summary")
    print(f"Rates: {rates}")
    print(f"Mean/std: {rates.mean():.3f} / {rates.std():.3f}")


if __name__ == "__main__":
    run_multiseed_eval()