import gymnasium as gym
import numpy as np

from src.finite_horizon_dp import compute_finite_horizon_optimal_policy
from src.finite_horizon_q_learning import (
    ACTION_NAMES,
    train_finite_horizon_q_learning,
)


def learned_policy_from_Q(Q, horizon):
    return np.argmax(Q[:horizon], axis=2)


def compute_occupancy(policy, horizon=100):
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

    occupancy = compute_occupancy(policy_learned, horizon=horizon)

    state = 0

    print("=" * 80)
    print("State 0 across time")
    print(
        "Format: t, learned action, optimal action, occupancy, "
        "one-step regret, weighted regret, visits"
    )

    rows = []

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

        rows.append(
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

    rows.sort(reverse=True)

    for item in rows[:30]:
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

        if occ < 1e-8 and one_step_regret < 1e-8:
            continue

        print(
            f"t={t:3d}: "
            f"learned={ACTION_NAMES[learned_action]:5s}, "
            f"optimal={ACTION_NAMES[optimal_action]:5s}, "
            f"occ={occ:.6f}, "
            f"regret={one_step_regret:.6f}, "
            f"weighted={weighted_regret:.6f}, "
            f"visits={total_visits:.0f}"
        )
        print(f"      learned Q = {learned_q}")
        print(f"      optimal Q = {optimal_q}")


if __name__ == "__main__":
    main()