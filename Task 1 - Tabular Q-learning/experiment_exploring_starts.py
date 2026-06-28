import numpy as np
import gymnasium as gym

from finite_horizon_q_learning_exploring_starts import (
    train_finite_horizon_q_learning_exploring_starts,
    print_policy_at_time,
)

from finite_horizon_dp import compute_finite_horizon_optimal_policy


def learned_policy_from_Q(Q, horizon):
    return np.argmax(Q[:horizon], axis=2)


def exact_eval_time_dependent_policy(policy, horizon=100, gamma=1.0):
    """
    Exact finite-horizon evaluation using env.P.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n
    V = np.zeros((horizon + 1, n_states))

    for t in reversed(range(horizon)):
        for state in range(n_states):
            action = int(policy[t, state])
            value = 0.0

            for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
                if terminated:
                    value += prob * reward
                else:
                    value += prob * (reward + gamma * V[t + 1, next_state])

            V[t, state] = value

    env.close()
    return V


def run_exploring_start_experiment():
    """
    Compare finite-horizon Q-learning with different exploring-start probabilities.

    exploring_start_prob = 0.0:
        Equivalent to normal start-state training.

    exploring_start_prob > 0:
        Some episodes start from random safe states and random time layers,
        improving coverage of the finite-horizon Q-table.
    """
    horizon = 100
    train_episodes = 100_000
    seeds = [0, 1, 2]

    exploring_probs = [0.0, 0.25, 0.5, 0.75]

    Q_opt, V_opt, policy_opt = compute_finite_horizon_optimal_policy(
        horizon=horizon,
        gamma=1.0,
    )

    print("=" * 80)
    print(f"DP optimal value: {V_opt[0, 0]:.6f}")

    for p in exploring_probs:
        rates = []

        print("=" * 80)
        print(f"Exploring start probability: {p}")

        for seed in seeds:
            Q, visit_counts = train_finite_horizon_q_learning_exploring_starts(
                episodes=train_episodes,
                horizon=horizon,
                alpha0=0.5,
                gamma=1.0,
                epsilon_start=1.0,
                epsilon_end=0.05,
                exploring_start_prob=p,
                seed=seed,
            )

            policy = learned_policy_from_Q(Q, horizon)
            V = exact_eval_time_dependent_policy(policy, horizon=horizon)
            value = V[0, 0]

            rates.append(value)

            print("-" * 60)
            print(f"Seed {seed}: exact value = {value:.6f}")

            if seed == 0:
                print("\nPolicy at t = 0:")
                print_policy_at_time(Q, 0)

                print("\nPolicy at t = 30:")
                print_policy_at_time(Q, 30)

                print("\nPolicy at t = 50:")
                print_policy_at_time(Q, 50)

                print("\nState 0 Q-values at t = 30:")
                print(Q[30, 0])

                print("\nState 4 Q-values at t = 45:")
                print(Q[45, 4])

                print("\nState 9 Q-values at t = 50:")
                print(Q[50, 9])

        rates = np.array(rates)

        print("-" * 60)
        print(f"Values: {rates}")
        print(f"Mean/std: {rates.mean():.6f} / {rates.std():.6f}")
        print(f"Gap to optimal: {V_opt[0, 0] - rates.mean():.6f}")


if __name__ == "__main__":
    run_exploring_start_experiment()