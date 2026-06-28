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

    In Task 1B we intentionally use is_slippery=True because the point
    is to study finite-horizon decision-making under stochastic dynamics.

    The environment is still the 4x4 FrozenLake:
        S F F F
        F H F H
        F F F H
        H F F G
    """
    return gym.make("FrozenLake-v1", is_slippery=True)


def epsilon_greedy_time_action(Q, t, state, epsilon, n_actions):
    """
    Choose an action using epsilon-greedy exploration for a time-dependent Q-table.

    Difference from Task 1:
        Task 1:
            Q[state, action]

        Task 1B:
            Q[t, state, action]

    With probability epsilon:
        choose a random action.

    With probability 1 - epsilon:
        choose one of the best actions according to Q[t, state].

    We randomize tie-breaking among equally good actions.
    """
    # TODO:
    # 1. With probability epsilon, return a random action.
    # 2. Otherwise, find all actions tied for max Q[t, state, action].
    # 3. Randomly choose one best action.
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        max_q = np.max(Q[t, state])
        best_actions = np.where(Q[t, state] == max_q)[0]
        return np.random.choice(best_actions)


def train_finite_horizon_q_learning(
    episodes=10000,
    horizon=100,
    alpha0=0.5,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=None,
):
    """
    Train finite-horizon tabular Q-learning.

    Q-table shape:
        Q[t, state, action]

    where:
        t ranges from 0 to horizon.
        state is the FrozenLake position.
        action is one of LEFT, DOWN, RIGHT, UP.

    Why horizon + 1?
        At time t, the update bootstraps from Q[t + 1, next_state].
        Therefore we need Q[horizon] as the terminal time layer.
        It remains zero, meaning:
            with no time left, future value is zero.

    Learning rate:
        alpha = alpha0 / sqrt(N[t, state, action])

    This is the slower visit-count decay that worked better in Task 1.
    """
    if seed is not None:
        np.random.seed(seed)

    env = make_env()

    if seed is not None:
        env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q[t, state, action] estimates value at a specific time step.
    Q = np.zeros((horizon + 1, n_states, n_actions))

    # Count visits separately for each time-state-action triple.
    visit_counts = np.zeros((horizon + 1, n_states, n_actions))

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        # Linear epsilon decay.
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / episodes,
        )

        # Finite-horizon loop.
        # t represents the current decision time.
        for t in range(horizon):
            if done:
                break

            action = epsilon_greedy_time_action(Q, t, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # TODO 1:
            # Increase visit count for this time-state-action triple.
            # visit_counts[t, state, action] = ?
            visit_counts[t, state, action] += 1

            # TODO 2:
            # Compute sqrt-decay learning rate:
            # alpha = alpha0 / sqrt(visit_counts[t, state, action])
            alpha = alpha0 / np.sqrt(visit_counts[t, state, action])

            # TODO 3:
            # Compute finite-horizon target.
            #
            # If done:
            #     no future value.
            #
            # Else:
            #     bootstrap from Q[t + 1, next_state].
            #
            # Important:
            #     This is the key finite-horizon difference.
            #     We use t + 1, not the same time layer.
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[t + 1, next_state])

            # TODO 4:
            # Compute TD error and update Q[t, state, action].
            td_error = target - Q[t, state, action]
            Q[t, state, action] += alpha * td_error

            state = next_state

    env.close()
    return Q, visit_counts


def greedy_action(Q, t, state):
    """
    Return the greedy action at a specific time and state.
    """
    return int(np.argmax(Q[t, state]))


def render_policy_at_time(Q, t):
    """
    Render the greedy policy at a chosen time step.

    Because the policy is time-dependent, we can inspect:
        policy at t = 0
        policy at t = 20
        policy at t = 80

    This helps us see whether the agent becomes more aggressive
    as the deadline approaches.
    """
    lake = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]

    policy_grid = []

    for row in range(4):
        policy_row = []

        for col in range(4):
            state = row * 4 + col
            cell = lake[row][col]

            if cell == "S":
                # For Task 1B, show the action at S explicitly.
                action = greedy_action(Q, t, state)
                policy_row.append(f"S({ACTION_SYMBOLS[action]})")
            elif cell in ["H", "G"]:
                policy_row.append(cell)
            else:
                action = greedy_action(Q, t, state)
                policy_row.append(ACTION_SYMBOLS[action])

        policy_grid.append(policy_row)

    return policy_grid


def print_policy_at_time(Q, t):
    """
    Print a time-dependent policy grid at time t.
    """
    policy = render_policy_at_time(Q, t)
    for row in policy:
        print(" ".join(row))


if __name__ == "__main__":
    Q, visit_counts = train_finite_horizon_q_learning(
        episodes=10000,
        horizon=100,
        alpha0=0.5,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=0,
    )

    for t in [0, 20, 50, 80]:
        print("=" * 60)
        print(f"Greedy policy at t = {t}")
        print_policy_at_time(Q, t)