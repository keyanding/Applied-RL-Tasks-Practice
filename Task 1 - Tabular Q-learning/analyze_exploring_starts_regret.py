import numpy as np
import gymnasium as gym

from finite_horizon_q_learning_exploring_starts import (
    train_finite_horizon_q_learning_exploring_starts,
    ACTION_NAMES,
)

from finite_horizon_dp import compute_finite_horizon_optimal_policy


def learned_policy_from_Q(Q, horizon):
    """
    Convert Q[t, state, action] into a deterministic greedy policy.
    """
    return np.argmax(Q[:horizon], axis=2)


def exact_eval_time_dependent_policy(policy, horizon=100, gamma=1.0):
    """
    Exactly evaluate a finite-horizon time-dependent policy using env.P.
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


def compute_occupancy(policy, horizon=100):
    """
    Compute occupancy[t, state] under the learned policy.

    occupancy[t, state] is the probability that the agent is alive and located
    at state at the start of time t.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_states = env.observation_space.n

    occupancy = np.zeros((horizon + 1, n_states))
    occupancy[0, 0] = 1.0

    for t in range(horizon):
        for state in range(n_states):
            prob_state = occupancy[t, state]

            if prob_state == 0:
                continue

            action = int(policy[t, state])

            for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
                if not terminated:
                    occupancy[t + 1, next_state] += prob_state * prob

    env.close()
    return occupancy


def analyze_policy(
    train_episodes=100_000,
    exploring_start_prob=0.5,
    seed=0,
):
    """
    Analyze the learned policy from exploring-start finite-horizon Q-learning.

    We compare it against DP optimal policy using:
        1. exact value
        2. occupancy-weighted regret
        3. state 0 across-time action errors
    """
    horizon = 100

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

    V_learned = exact_eval_time_dependent_policy(
        policy_learned,
        horizon=horizon,
    )

    occupancy = compute_occupancy(policy_learned, horizon=horizon)

    print("=" * 80)
    print("Exact values")
    print(f"episodes: {train_episodes}")
    print(f"exploring_start_prob: {exploring_start_prob}")
    print(f"seed: {seed}")
    print(f"Optimal value: {V_opt[0, 0]:.6f}")
    print(f"Learned value: {V_learned[0, 0]:.6f}")
    print(f"Gap:           {V_opt[0, 0] - V_learned[0, 0]:.6f}")

    safe_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]

    mistakes = []

    for t in range(horizon):
        for state in safe_states:
            occ = occupancy[t, state]

            if occ < 1e-8:
                continue

            learned_action = int(policy_learned[t, state])
            optimal_action = int(policy_opt[t, state])

            if learned_action == optimal_action:
                continue

            one_step_regret = (
                Q_opt[t, state, optimal_action]
                - Q_opt[t, state, learned_action]
            )

            if one_step_regret <= 1e-8:
                continue

            weighted_regret = occ * one_step_regret
            total_visits = visit_counts[t, state].sum()

            mistakes.append(
                (
                    weighted_regret,
                    one_step_regret,
                    occ,
                    t,
                    state,
                    learned_action,
                    optimal_action,
                    total_visits,
                )
            )

    mistakes.sort(reverse=True)

    print("=" * 80)
    print("Top occupancy-weighted mistakes")

    for item in mistakes[:20]:
        (
            weighted_regret,
            one_step_regret,
            occ,
            t,
            state,
            learned_action,
            optimal_action,
            total_visits,
        ) = item

        print(
            f"t={t:3d}, state={state:2d}: "
            f"{ACTION_NAMES[learned_action]:5s} -> {ACTION_NAMES[optimal_action]:5s}, "
            f"occ={occ:.6f}, "
            f"regret={one_step_regret:.6f}, "
            f"weighted={weighted_regret:.6f}, "
            f"visits={total_visits:.0f}"
        )

    print("=" * 80)
    print("State 0 across time: top weighted mistakes")

    state0_rows = []

    state = 0

    for t in range(horizon):
        learned_action = int(policy_learned[t, state])
        optimal_action = int(policy_opt[t, state])
        occ = occupancy[t, state]

        one_step_regret = (
            Q_opt[t, state, optimal_action]
            - Q_opt[t, state, learned_action]
        )

        weighted_regret = occ * one_step_regret
        total_visits = visit_counts[t, state].sum()

        if weighted_regret > 1e-8:
            state0_rows.append(
                (
                    weighted_regret,
                    one_step_regret,
                    occ,
                    t,
                    learned_action,
                    optimal_action,
                    total_visits,
                    Q_learned[t, state],
                    Q_opt[t, state],
                )
            )

    state0_rows.sort(reverse=True)

    for item in state0_rows[:15]:
        (
            weighted_regret,
            one_step_regret,
            occ,
            t,
            learned_action,
            optimal_action,
            total_visits,
            learned_q,
            optimal_q,
        ) = item

        print(
            f"t={t:3d}: "
            f"{ACTION_NAMES[learned_action]:5s} -> {ACTION_NAMES[optimal_action]:5s}, "
            f"occ={occ:.6f}, "
            f"regret={one_step_regret:.6f}, "
            f"weighted={weighted_regret:.6f}, "
            f"visits={total_visits:.0f}"
        )
        print(f"      learned Q = {learned_q}")
        print(f"      optimal Q = {optimal_q}")


if __name__ == "__main__":
    analyze_policy(
        train_episodes=100_000,
        exploring_start_prob=0.5,
        seed=0,
    )