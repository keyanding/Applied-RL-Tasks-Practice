import gymnasium as gym
import numpy as np

from src.finite_horizon_dp import compute_finite_horizon_optimal_policy
from src.finite_horizon_q_learning_exploring_starts import (
    train_finite_horizon_q_learning_exploring_starts,
)


def learned_policy_from_Q(Q, horizon):
    """
    Convert learned Q[t, state, action] into policy[t, state].
    """
    return np.argmax(Q[:horizon], axis=2)


def exact_eval_time_dependent_policy(policy, horizon=100, gamma=1.0):
    """
    Exactly evaluate a finite-horizon time-dependent policy.
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


def replace_states_across_all_times(policy_learned, policy_opt, states):
    """
    Replace learned actions for selected physical states across all time steps.
    """
    new_policy = policy_learned.copy()

    for state in states:
        new_policy[:, state] = policy_opt[:, state]

    return new_policy


def replace_time_window(policy_learned, policy_opt, start_t, end_t):
    """
    Replace learned policy with optimal policy for a time window.
    """
    new_policy = policy_learned.copy()
    new_policy[start_t:end_t, :] = policy_opt[start_t:end_t, :]
    return new_policy


def main():
    horizon = 100
    seed = 0
    train_episodes = 100_000
    exploring_start_prob = 0.5

    Q_learned, visit_counts = train_finite_horizon_q_learning_exploring_starts(
        episodes=train_episodes,
        horizon=horizon,
        alpha0=0.5,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        exploring_start_prob=exploring_start_prob,
        seed=seed,
    )

    policy_learned = learned_policy_from_Q(Q_learned, horizon)

    Q_opt, V_opt, policy_opt = compute_finite_horizon_optimal_policy(
        horizon=horizon,
        gamma=1.0,
    )

    V_learned = exact_eval_time_dependent_policy(policy_learned, horizon=horizon)

    baseline = V_learned[0, 0]
    optimal = V_opt[0, 0]

    print("=" * 80)
    print("Baseline")
    print(f"Learned value: {baseline:.6f}")
    print(f"Optimal value: {optimal:.6f}")
    print(f"Gap: {optimal - baseline:.6f}")

    print("=" * 80)
    print("Single / grouped state interventions")

    state_sets = [
        [0],
        [2],
        [8],
        [0, 2],
        [0, 8],
        [2, 8],
        [0, 2, 8],
        [0, 2, 8, 4, 9, 13, 14],
        [0, 1, 2, 4, 8, 9, 10, 13, 14],
    ]

    for states in state_sets:
        intervened_policy = replace_states_across_all_times(
            policy_learned,
            policy_opt,
            states,
        )

        V_intervened = exact_eval_time_dependent_policy(
            intervened_policy,
            horizon=horizon,
        )

        value = V_intervened[0, 0]

        print(
            f"Replace states={states}: "
            f"value={value:.6f}, "
            f"improvement={value - baseline:.6f}, "
            f"remaining_gap={optimal - value:.6f}"
        )

    print("=" * 80)
    print("Time-window interventions")

    windows = [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
        (80, 100),
    ]

    for start_t, end_t in windows:
        intervened_policy = replace_time_window(
            policy_learned,
            policy_opt,
            start_t,
            end_t,
        )

        V_intervened = exact_eval_time_dependent_policy(
            intervened_policy,
            horizon=horizon,
        )

        value = V_intervened[0, 0]

        print(
            f"Replace t={start_t:02d}..{end_t - 1:02d}: "
            f"value={value:.6f}, "
            f"improvement={value - baseline:.6f}, "
            f"remaining_gap={optimal - value:.6f}"
        )


if __name__ == "__main__":
    main()