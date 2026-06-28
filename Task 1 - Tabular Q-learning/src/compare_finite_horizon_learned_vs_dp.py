import gymnasium as gym
import numpy as np

from src.finite_horizon_dp import (
    compute_finite_horizon_optimal_policy,
)
from src.finite_horizon_dp import (
    print_policy_at_time as print_optimal_policy_at_time,
)
from src.finite_horizon_q_learning import (
    ACTION_NAMES,
    train_finite_horizon_q_learning,
)
from src.finite_horizon_q_learning import (
    print_policy_at_time as print_learned_policy_at_time,
)


def learned_policy_from_Q(Q, horizon):
    """
    Convert learned finite-horizon Q-table into a deterministic policy.

    Q shape:
        [horizon + 1, n_states, n_actions]

    learned_policy shape:
        [horizon, n_states]

    We only need t = 0 ... horizon - 1 because t = horizon means
    no action is taken.
    """
    return np.argmax(Q[:horizon], axis=2)


def exact_eval_time_dependent_policy(policy, horizon=100, gamma=1.0):
    """
    Exactly evaluate a finite-horizon time-dependent policy.

    policy[t, state] gives the action taken at time t.

    This uses env.P, so it has no Monte Carlo sampling noise.
    It evaluates the probability of reaching the goal within horizon steps.
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


def print_key_state_comparison(Q_learned, Q_opt, policy_learned, policy_opt, V_opt, times):
    """
    Compare learned and optimal actions at selected time steps and states.

    This helps us see whether the learned policy differs from DP optimal
    in safety-critical states such as:
        state 0  = start
        state 9  = critical middle state
        state 14 = near goal
    """
    key_states = [0, 4, 8, 9, 10, 13, 14]

    for t in times:
        print("=" * 80)
        print(f"Key state comparison at t = {t}")

        for state in key_states:
            learned_action = int(policy_learned[t, state])
            optimal_action = int(policy_opt[t, state])

            learned_q = Q_learned[t, state]
            optimal_q = Q_opt[t, state]

            # This is the value loss according to the DP optimal Q-table
            # caused by choosing the learned action instead of the optimal action.
            one_step_regret = (
                optimal_q[optimal_action] - optimal_q[learned_action]
            )

            print(f"state {state}:")
            print(f"  learned action: {ACTION_NAMES[learned_action]}")
            print(f"  optimal action: {ACTION_NAMES[optimal_action]}")
            print(f"  optimal V[t,state]: {V_opt[t, state]:.6f}")
            print(f"  learned Q: {learned_q}")
            print(f"  optimal Q: {optimal_q}")
            print(f"  one-step regret under optimal Q: {one_step_regret:.6f}")


def find_top_action_disagreements(Q_opt, V_opt, policy_learned, policy_opt, horizon):
    """
    Find states where the learned policy disagrees with the DP optimal policy.

    We rank disagreements by one-step regret under the DP optimal Q-table:

        regret = Q_opt[t,s,opt_action] - Q_opt[t,s,learned_action]

    Large regret means:
        choosing the learned action instead of the optimal action is costly
        at that time-state pair.

    We filter to safe states and states with meaningful optimal value.
    """
    safe_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]

    disagreements = []

    for t in range(horizon):
        for state in safe_states:
            learned_action = int(policy_learned[t, state])
            optimal_action = int(policy_opt[t, state])

            if learned_action == optimal_action:
                continue

            # If optimal value is almost zero, the state may already be too late
            # or too far from the goal. Disagreements there are less meaningful.
            if V_opt[t, state] < 1e-6:
                continue

            regret = (
                Q_opt[t, state, optimal_action]
                - Q_opt[t, state, learned_action]
            )

            if regret > 1e-6:
                disagreements.append(
                    (regret, t, state, learned_action, optimal_action, V_opt[t, state])
                )

    disagreements.sort(reverse=True)

    print("=" * 80)
    print("Top learned-vs-optimal action disagreements")
    print("Ranked by one-step regret under DP optimal Q")

    for item in disagreements[:25]:
        regret, t, state, learned_action, optimal_action, optimal_value = item

        print(
            f"t={t:3d}, state={state:2d}: "
            f"learned={ACTION_NAMES[learned_action]:5s}, "
            f"optimal={ACTION_NAMES[optimal_action]:5s}, "
            f"regret={regret:.6f}, "
            f"V_opt={optimal_value:.6f}"
        )

    print(f"\nTotal meaningful disagreements: {len(disagreements)}")


def main():
    horizon = 100
    seed = 0
    train_episodes = 300_000

    print("=" * 80)
    print("Training finite-horizon Q-learning")
    print(f"episodes = {train_episodes}, seed = {seed}")

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

    print("=" * 80)
    print("Computing DP optimal finite-horizon policy")

    Q_opt, V_opt, policy_opt = compute_finite_horizon_optimal_policy(
        horizon=horizon,
        gamma=1.0,
    )

    V_learned = exact_eval_time_dependent_policy(
        policy_learned,
        horizon=horizon,
        gamma=1.0,
    )

    print("=" * 80)
    print("Exact policy values")
    print(f"DP optimal success probability: {V_opt[0, 0]:.6f}")
    print(f"Learned policy success probability: {V_learned[0, 0]:.6f}")
    print(f"Gap: {V_opt[0, 0] - V_learned[0, 0]:.6f}")

    for t in [0, 20, 50, 80, 90]:
        print("=" * 80)
        print(f"Learned policy at t = {t}")
        print_learned_policy_at_time(Q_learned, t)

        print("\nDP optimal policy at t = {t}".format(t=t))
        print_optimal_policy_at_time(policy_opt, t)

    print_key_state_comparison(
        Q_learned=Q_learned,
        Q_opt=Q_opt,
        policy_learned=policy_learned,
        policy_opt=policy_opt,
        V_opt=V_opt,
        times=[0, 20, 50, 80, 90],
    )

    find_top_action_disagreements(
        Q_opt=Q_opt,
        V_opt=V_opt,
        policy_learned=policy_learned,
        policy_opt=policy_opt,
        horizon=horizon,
    )


if __name__ == "__main__":
    main()