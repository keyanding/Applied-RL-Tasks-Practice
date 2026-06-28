import numpy as np
import gymnasium as gym

from q_learning import train_q_learning_with_sqrt_decay, render_policy


ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def greedy_policy_from_Q(Q):
    """
    Convert a Q-table into a deterministic greedy policy.

    policy[state] = action with the highest Q-value at that state.
    """
    return np.argmax(Q, axis=1)


def exact_policy_evaluation(policy, gamma=1.0, theta=1e-12):
    """
    Exactly evaluate a fixed deterministic policy in slippery FrozenLake.

    We use dynamic programming with the known transition model env.P.

    Args:
        policy:
            Array where policy[state] is the action chosen at that state.

        gamma:
            Discount factor for evaluation.
            For success probability in episodic FrozenLake, gamma=1.0 is natural.

        theta:
            Stop when value changes are smaller than this threshold.

    Returns:
        V:
            Value function under the policy.
            V[0] is the exact success probability from the start state.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    n_states = env.observation_space.n
    V = np.zeros(n_states)

    while True:
        delta = 0.0

        for state in range(n_states):
            action = int(policy[state])

            old_v = V[state]
            new_v = 0.0

            # env.P[state][action] is a list of transitions:
            # (probability, next_state, reward, terminated)
            for prob, next_state, reward, terminated in P[state][action]:
                if terminated:
                    new_v += prob * reward
                else:
                    new_v += prob * (reward + gamma * V[next_state])

            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            break

    env.close()
    return V


def finite_horizon_policy_evaluation(policy, horizon=100):
    """
    Evaluate a fixed policy under a finite step limit.

    V[h, s] means:
        probability of reaching the goal within h remaining steps
        when starting from state s.

    This matches Monte Carlo evaluation with a TimeLimit more closely than
    infinite-horizon policy evaluation.

    Args:
        policy:
            Deterministic policy array. policy[state] = action.

        horizon:
            Maximum number of steps allowed.

    Returns:
        V:
            Array of shape [horizon + 1, n_states].
            V[horizon, 0] is the success probability from start within horizon steps.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n
    P = env.unwrapped.P

    # V[h, s] = success probability from state s with h steps remaining.
    V = np.zeros((horizon + 1, n_states))

    for h in range(1, horizon + 1):
        for state in range(n_states):
            action = int(policy[state])
            value = 0.0

            for prob, next_state, reward, terminated in P[state][action]:
                if terminated:
                    value += prob * reward
                else:
                    value += prob * V[h - 1, next_state]

            V[h, state] = value

    env.close()
    return V


def print_policy(policy):
    """
    Print policy as a 4x4 grid.
    """
    symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑",
    }

    lake = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]

    for row in range(4):
        items = []
        for col in range(4):
            state = row * 4 + col
            cell = lake[row][col]
            if cell in ["S", "H", "G"]:
                items.append(cell)
            else:
                items.append(symbols[int(policy[state])])
        print(" ".join(items))


def main():
    seed = 2

    Q, visit_counts = train_q_learning_with_sqrt_decay(
        episodes=3000,
        alpha0=0.5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=seed,
    )

    policy = greedy_policy_from_Q(Q)

    print("=" * 60)
    # print("Original sqrt-decay seed 2 policy")
    # print_policy(policy)

    # V = exact_policy_evaluation(policy, gamma=1.0)
    # print(f"Exact success probability from state 0: {V[0]:.6f}")

    # Intervention: force state 0 to LEFT.
    policy_intervened = policy.copy()
    policy_intervened[0] = 0

    print("\nAfter intervention: state 0 -> LEFT")
    print_policy(policy_intervened)

    # V_intervened = exact_policy_evaluation(policy_intervened, gamma=1.0)
    # print(f"Exact success probability from state 0: {V_intervened[0]:.6f}")

    V_finite = finite_horizon_policy_evaluation(policy, horizon=100)
    print(f"Finite-horizon success probability from state 0: {V_finite[100, 0]:.6f}")

    V_intervened_finite = finite_horizon_policy_evaluation(policy_intervened, horizon=100)
    print(
        f"Finite-horizon success probability after intervention: "
        f"{V_intervened_finite[100, 0]:.6f}"
    )


if __name__ == "__main__":
    main()