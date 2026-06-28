import gymnasium as gym
import numpy as np

ACTION_SYMBOLS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}


ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def make_env():
    """
    Create slippery FrozenLake.

    We use the transition model env.P to compute the exact finite-horizon
    optimal value function and policy by dynamic programming.
    """
    return gym.make("FrozenLake-v1", is_slippery=True)


def compute_finite_horizon_optimal_policy(horizon=100, gamma=1.0):
    """
    Compute the exact optimal finite-horizon Q-table by backward induction.

    Q[t, state, action] means:
        value of taking action at state when the current time is t.

    V[t, state] means:
        best achievable success probability from state at time t.

    Boundary:
        V[horizon, state] = 0

    Recurrence:
        Q[t, s, a] =
            sum over transitions:
                prob * (reward + gamma * V[t+1, next_state])
            if next_state is non-terminal

        If transition terminates:
            future value is zero, but immediate reward still counts.

    This gives us a ground-truth benchmark for finite-horizon control.
    """
    env = make_env()
    P = env.unwrapped.P

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros((horizon + 1, n_states))
    Q = np.zeros((horizon, n_states, n_actions))
    policy = np.zeros((horizon, n_states), dtype=int)

    # TODO:
    # Work backward from t = horizon - 1 down to 0.
    #
    # For each time t:
    #   For each state:
    #       For each action:
    #           Use env.P[state][action] to compute expected value.
    #
    # env.P[state][action] contains tuples:
    #   (probability, next_state, reward, terminated)
    #
    # If terminated:
    #   contribution = prob * reward
    #
    # Else:
    #   contribution = prob * (reward + gamma * V[t + 1, next_state])
    #
    # After computing all action values:
    #   V[t, state] = max_a Q[t, state, a]
    #   policy[t, state] = argmax_a Q[t, state, a]

    for t in reversed(range(horizon)):
        for state in range(n_states):
            for action in range(n_actions):
                value = 0.0

                for prob, next_state, reward, terminated in P[state][action]:
                    if terminated:
                        value += prob * reward
                    else:
                        value += prob * (reward + gamma * V[t + 1, next_state])

                Q[t, state, action] = value

            V[t, state] = np.max(Q[t, state])
            policy[t, state] = int(np.argmax(Q[t, state]))

    env.close()
    return Q, V, policy


def render_policy_at_time(policy, t):
    """
    Render a finite-horizon policy at a specific time.

    Unlike the previous renderer, we show the action at S explicitly:
        S(←), S(↓), S(→), or S(↑)

    This is important because the start state's action was one of our
    major failure points in Task 1.
    """
    lake = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]

    grid = []

    for row in range(4):
        row_items = []

        for col in range(4):
            state = row * 4 + col
            cell = lake[row][col]

            if cell == "S":
                action = policy[t, state]
                row_items.append(f"S({ACTION_SYMBOLS[action]})")
            elif cell in ["H", "G"]:
                row_items.append(cell)
            else:
                action = policy[t, state]
                row_items.append(ACTION_SYMBOLS[action])

        grid.append(row_items)

    return grid


def print_policy_at_time(policy, t):
    """
    Print policy at a chosen time step.
    """
    grid = render_policy_at_time(policy, t)
    for row in grid:
        print(" ".join(row))


def main():
    horizon = 100

    Q, V, policy = compute_finite_horizon_optimal_policy(
        horizon=horizon,
        gamma=1.0,
    )

    print("=" * 60)
    print("Finite-horizon optimal policy by dynamic programming")
    print(f"Optimal success probability from state 0 at t=0: {V[0, 0]:.6f}")

    for t in [0, 20, 50, 80, 90, 99]:
        print("=" * 60)
        print(f"Optimal policy at t = {t}")
        print_policy_at_time(policy, t)

        print("Key state actions:")
        for state in [0, 4, 8, 9, 10, 13, 14]:
            action = policy[t, state]
            print(
                f"state {state}: {ACTION_NAMES[action]}, "
                f"Q = {Q[t, state]}"
            )


if __name__ == "__main__":
    main()