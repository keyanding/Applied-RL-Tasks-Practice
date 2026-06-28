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


SAFE_STATES = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]


def make_env():
    """
    Create slippery FrozenLake.

    We use the same environment as before, but this trainer will sometimes
    manually set the internal state to implement exploring starts.
    """
    return gym.make("FrozenLake-v1", is_slippery=True)


def set_env_state(env, state):
    """
    Manually set the FrozenLake internal state.

    This is a simulator-only trick.

    In real control training, this corresponds to resetting the simulator to
    selected initial conditions, such as:
        - different altitudes
        - different velocities
        - different wind conditions
        - different distances to target

    For Gymnasium FrozenLake, the unwrapped environment stores the current
    position in env.unwrapped.s.
    """
    env.unwrapped.s = state


def epsilon_greedy_time_action(Q, t, state, epsilon, n_actions):
    """
    Epsilon-greedy action selection for finite-horizon Q[t, state, action].
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)

    max_q = np.max(Q[t, state])
    best_actions = np.where(Q[t, state] == max_q)[0]
    return int(np.random.choice(best_actions))


def sample_start_time_and_state(horizon, exploring_start_prob):
    """
    Decide whether to use a normal start or an exploring start.

    Normal start:
        t_start = 0
        state_start = 0

    Exploring start:
        t_start is randomly sampled from 0 to horizon - 1.
        state_start is randomly sampled from safe states.

    Args:
        horizon:
            Maximum episode length.

        exploring_start_prob:
            Probability of using an exploring start.

    Returns:
        t_start, state_start
    """
    # TODO:
    # With probability exploring_start_prob:
    #     sample t_start randomly from [0, horizon)
    #     sample state_start randomly from SAFE_STATES
    #
    # Otherwise:
    #     return normal start: t_start = 0, state_start = 0
    if np.random.random() < exploring_start_prob:
        t_start = np.random.randint(horizon)
        state_start = int(np.random.choice(SAFE_STATES))
    else:
        t_start = 0
        state_start = 0
    return t_start, state_start


def train_finite_horizon_q_learning_exploring_starts(
    episodes=100000,
    horizon=100,
    alpha0=0.5,
    gamma=1.0,
    epsilon_start=1.0,
    epsilon_end=0.05,
    exploring_start_prob=0.5,
    seed=None,
):
    """
    Train finite-horizon Q-learning with exploring starts.

    Difference from naive finite-horizon Q-learning:
        Before:
            every episode starts from t=0, state=0

        Now:
            with some probability, an episode starts from random time and
            random safe state.

    Why this helps:
        It increases coverage of time-state pairs such as:
            t=40, state=4
            t=50, state=9
            t=60, state=13

        These were exactly the under-learned regions revealed by
        occupancy-weighted regret analysis.

    Returns:
        Q:
            Learned finite-horizon Q-table.

        visit_counts:
            Visit counts for [t, state, action].
    """
    if seed is not None:
        np.random.seed(seed)

    env = make_env()

    if seed is not None:
        env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((horizon + 1, n_states, n_actions))
    visit_counts = np.zeros((horizon + 1, n_states, n_actions))

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        t_start, state_start = sample_start_time_and_state(
            horizon=horizon,
            exploring_start_prob=exploring_start_prob,
        )

        # Manually place the agent at the selected start state.
        set_env_state(env, state_start)
        state = state_start

        done = False

        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / episodes,
        )

        # Start the finite-horizon rollout from t_start rather than always t=0.
        for t in range(t_start, horizon):
            if done:
                break

            action = epsilon_greedy_time_action(Q, t, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # TODO 1:
            # Update visit count for this time-state-action.
            # visit_counts[t, state, action] = ?
            visit_counts[t, state, action] += 1

            # TODO 2:
            # Compute sqrt-decay learning rate.
            # alpha = ?
            alpha = alpha0 / np.sqrt(visit_counts[t, state, action])
            # TODO 3:
            # Compute finite-horizon Q-learning target.
            # If done: target = reward
            # Else: target = reward + gamma * max_a Q[t+1, next_state, a]
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
    Greedy action for finite-horizon Q-table.
    """
    return int(np.argmax(Q[t, state]))


def render_policy_at_time(Q, t):
    """
    Render policy at a specific time t.
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
                action = greedy_action(Q, t, state)
                row_items.append(f"S({ACTION_SYMBOLS[action]})")
            elif cell in ["H", "G"]:
                row_items.append(cell)
            else:
                action = greedy_action(Q, t, state)
                row_items.append(ACTION_SYMBOLS[action])

        grid.append(row_items)

    return grid


def print_policy_at_time(Q, t):
    """
    Print policy at time t.
    """
    grid = render_policy_at_time(Q, t)
    for row in grid:
        print(" ".join(row))