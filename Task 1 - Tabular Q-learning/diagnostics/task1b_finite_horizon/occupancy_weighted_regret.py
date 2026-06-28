import gymnasium as gym
import numpy as np

from src.finite_horizon_dp import compute_finite_horizon_optimal_policy
from src.finite_horizon_q_learning import (
    ACTION_NAMES,
    train_finite_horizon_q_learning,
)


def learned_policy_from_Q(Q, horizon):
    """
    Convert learned Q[t, state, action] into policy[t, state].
    """
    return np.argmax(Q[:horizon], axis=2)


def compute_occupancy(policy, horizon=100):
    """
    Compute state occupancy probabilities under a time-dependent policy.

    occupancy[t, s] means:
        probability that the agent is alive and located at state s
        at the start of time step t.

    We start from:
        occupancy[0, 0] = 1

    Important:
        If a transition terminates, we do NOT carry probability forward,
        because the episode has ended.

    Args:
        policy:
            Array of shape [horizon, n_states].
            policy[t, state] gives the action taken at time t.

        horizon:
            Maximum number of steps.

    Returns:
        occupancy:
            Array of shape [horizon + 1, n_states].
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n
    occupancy = np.zeros((horizon + 1, n_states))

    # At time 0, the agent starts at state 0 with probability 1.
    occupancy[0, 0] = 1.0

    for t in range(horizon):
        for state in range(n_states):
            prob_state = occupancy[t, state]

            if prob_state == 0:
                continue

            action = int(policy[t, state])

            for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
                # TODO:
                # If the transition is terminal, do not propagate probability.
                # Otherwise, add probability mass to occupancy[t + 1, next_state].
                #
                # The probability mass is:
                #     prob_state * prob
                if not terminated:
                    occupancy[t + 1, next_state] += prob_state * prob

    env.close()
    return occupancy


def exact_eval_time_dependent_policy(policy, horizon=100, gamma=1.0):
    """
    Exact finite-horizon evaluation of a time-dependent policy.

    Returns:
        V[t, state], where V[0,0] is success probability from start.
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


def analyze_weighted_regret():
    """
    Compare learned finite-horizon Q-learning against DP optimal policy.

    Unlike raw one-step regret, this analysis weights each mistake by how likely
    the learned policy is to actually visit that time-state pair.
    """
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

    V_learned = exact_eval_time_dependent_policy(
        policy_learned,
        horizon=horizon,
        gamma=1.0,
    )

    occupancy = compute_occupancy(
        policy_learned,
        horizon=horizon,
    )

    print("=" * 80)
    print("Exact values")
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

            mistakes.append(
                (
                    weighted_regret,
                    one_step_regret,
                    occ,
                    t,
                    state,
                    learned_action,
                    optimal_action,
                    V_opt[t, state],
                    visit_counts[t, state].sum(),
                )
            )

    mistakes.sort(reverse=True)

    print("=" * 80)
    print("Top occupancy-weighted mistakes")
    print(
        "Format: t, state, learned -> optimal, "
        "occupancy, one-step regret, weighted regret, visits"
    )

    for item in mistakes[:25]:
        (
            weighted_regret,
            one_step_regret,
            occ,
            t,
            state,
            learned_action,
            optimal_action,
            v_opt,
            total_visits,
        ) = item

        print(
            f"t={t:3d}, state={state:2d}: "
            f"{ACTION_NAMES[learned_action]:5s} -> {ACTION_NAMES[optimal_action]:5s}, "
            f"occ={occ:.6f}, "
            f"regret={one_step_regret:.6f}, "
            f"weighted={weighted_regret:.6f}, "
            f"V_opt={v_opt:.6f}, "
            f"visits={total_visits:.0f}"
        )

    total_weighted_regret = sum(x[0] for x in mistakes)

    print("=" * 80)
    print(f"Sum of listed occupancy-weighted one-step regrets: {total_weighted_regret:.6f}")
    print(
        "Note: this is a diagnostic score, not exactly equal to the full value gap, "
        "because changing one action also changes future occupancy."
    )


if __name__ == "__main__":
    analyze_weighted_regret()