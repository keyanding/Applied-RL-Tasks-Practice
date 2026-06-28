import gymnasium as gym
import numpy as np

from src.finite_horizon_dp import compute_finite_horizon_optimal_policy
from src.finite_horizon_q_learning import ACTION_NAMES, train_finite_horizon_q_learning


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


def apply_time_state_overrides(policy, overrides):
    """
    Return a copy of policy with selected time-state actions overridden.

    Args:
        policy:
            Array of shape [horizon, n_states].

        overrides:
            Dict mapping (t, state) -> forced_action.
            Example:
                {(43, 4): 0}
            means:
                at time 43 and state 4, force action LEFT.
    """
    new_policy = policy.copy()

    for (t, state), action in overrides.items():
        new_policy[t, state] = action

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

    baseline_V = exact_eval_time_dependent_policy(policy_learned, horizon=horizon)

    print("=" * 80)
    print("Baseline")
    print(f"Learned policy value: {baseline_V[0, 0]:.6f}")
    print(f"Optimal policy value: {V_opt[0, 0]:.6f}")
    print(f"Gap: {V_opt[0, 0] - baseline_V[0, 0]:.6f}")

    # Top mistakes from occupancy-weighted regret.
    # Each tuple: (t, state, optimal_action)
    targeted_fixes = [
        (43, 4, 0),   # state 4: LEFT
        (46, 4, 0),   # state 4: LEFT
        (47, 4, 0),   # state 4: LEFT
        (49, 4, 0),   # state 4: LEFT
        (49, 9, 1),   # state 9: DOWN
        (51, 13, 2),  # state 13: RIGHT
        (51, 8, 3),   # state 8: UP
        (52, 13, 2),  # state 13: RIGHT
    ]

    overrides = {}

    for k in range(1, len(targeted_fixes) + 1):
        t, state, action = targeted_fixes[k - 1]
        overrides[(t, state)] = action

        intervened_policy = apply_time_state_overrides(
            policy_learned,
            overrides,
        )

        V_intervened = exact_eval_time_dependent_policy(
            intervened_policy,
            horizon=horizon,
        )

        print("-" * 80)
        print(f"After applying top {k} targeted fixes:")
        print(f"  fixed (t={t}, state={state}) -> {ACTION_NAMES[action]}")
        print(f"  value: {V_intervened[0, 0]:.6f}")
        print(f"  improvement: {V_intervened[0, 0] - baseline_V[0, 0]:.6f}")
        print(f"  remaining gap: {V_opt[0, 0] - V_intervened[0, 0]:.6f}")

    # Also test a broader range: fix state 4 to optimal action LEFT
    # for t=35..55, because many top mistakes involve state 4 in this region.
    broad_overrides = {}

    for t in range(35, 56):
        broad_overrides[(t, 4)] = int(policy_opt[t, 4])

    broad_policy = apply_time_state_overrides(policy_learned, broad_overrides)
    V_broad = exact_eval_time_dependent_policy(broad_policy, horizon=horizon)

    print("=" * 80)
    print("Broad intervention: force state 4 to DP-optimal action for t=35..55")
    print(f"value: {V_broad[0, 0]:.6f}")
    print(f"improvement: {V_broad[0, 0] - baseline_V[0, 0]:.6f}")
    print(f"remaining gap: {V_opt[0, 0] - V_broad[0, 0]:.6f}")


if __name__ == "__main__":
    main()