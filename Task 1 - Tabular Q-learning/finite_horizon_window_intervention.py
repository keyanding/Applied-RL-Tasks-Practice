import numpy as np
import gymnasium as gym

from finite_horizon_q_learning import train_finite_horizon_q_learning
from finite_horizon_dp import compute_finite_horizon_optimal_policy


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


def replace_time_window(policy_learned, policy_opt, start_t, end_t):
    """
    Replace learned policy with DP optimal policy for a time window.

    The window is:
        start_t <= t < end_t

    Example:
        replace_time_window(policy_learned, policy_opt, 40, 60)
        replaces t = 40, 41, ..., 59.
    """
    new_policy = policy_learned.copy()
    new_policy[start_t:end_t, :] = policy_opt[start_t:end_t, :]
    return new_policy


def replace_state_across_all_times(policy_learned, policy_opt, state):
    """
    Replace learned actions for one state across all time steps.

    This tests whether one physical state is a major bottleneck.
    """
    new_policy = policy_learned.copy()
    new_policy[:, state] = policy_opt[:, state]
    return new_policy


def replace_states_across_all_times(policy_learned, policy_opt, states):
    """
    Replace learned actions for several states across all time steps.
    """
    new_policy = policy_learned.copy()

    for state in states:
        new_policy[:, state] = policy_opt[:, state]

    return new_policy


def main():
    horizon = 100
    seed = 0
    train_episodes = 300_000

    Q_learned, visit_counts = train_finite_horizon_q_learning(
        episodes=train_episodes,
        horizon=horizon,
        alpha0=0.5,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
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

    print("=" * 80)
    print("Single-state interventions across all times")

    key_states = [0, 1, 2, 4, 6, 8, 9, 10, 13, 14]

    for state in key_states:
        intervened_policy = replace_state_across_all_times(
            policy_learned,
            policy_opt,
            state,
        )

        V_intervened = exact_eval_time_dependent_policy(
            intervened_policy,
            horizon=horizon,
        )

        value = V_intervened[0, 0]

        print(
            f"Replace state={state:2d} across all t: "
            f"value={value:.6f}, "
            f"improvement={value - baseline:.6f}, "
            f"remaining_gap={optimal - value:.6f}"
        )

    print("=" * 80)
    print("State-set interventions across all times")

    state_sets = [
        [4, 8, 9, 13],
        [4, 8, 9, 13, 14],
        [0, 4, 8, 9, 13, 14],
        [1, 2, 4, 8, 9, 10, 13, 14],
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


if __name__ == "__main__":
    main()