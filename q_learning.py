import gymnasium as gym
import numpy as np


def make_env():
    """
    Create the FrozenLake environment.

    FrozenLake is a small grid-world environment.
    The agent starts at S and tries to reach G.
    Some cells are holes H, and falling into them ends the episode.

    is_slippery=False makes the environment deterministic:
    when the agent chooses a direction, it moves exactly in that direction.
    This is easier for our first Q-learning task.
    """
    return gym.make("FrozenLake-v1", is_slippery=True)


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """
    Choose an action using epsilon-greedy exploration.

    With probability epsilon:
        choose a random action, so the agent explores.

    With probability 1 - epsilon:
        choose one of the currently best actions according to Q[state].

    We randomly break ties among equally good actions.
    This avoids always choosing the first action when all Q-values are equal,
    which is important early in training.
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        best_actions = np.where(Q[state] == np.max(Q[state]))[0]
        return np.random.choice(best_actions)


def train_q_learning(
    episodes=3000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=None
):
    """
    Train a tabular Q-learning agent.

    Q-learning learns a table Q[state, action].
    Each entry estimates how good it is to take a given action
    in a given state.

    Args:
        episodes:
            Number of training episodes.

        alpha:
            Learning rate. Controls how strongly new experience updates Q.

        gamma:
            Discount factor. Controls how much the agent values future reward.

        epsilon_start:
            Initial exploration probability.

        epsilon_end:
            Minimum exploration probability after decay.

    Returns:
        Q:
            A learned Q-table of shape [n_states, n_actions].
    """
    if seed is not None:
        np.random.seed(seed)

    env = make_env()

    if seed is not None:
        # Seed the action space for reproducible random action sampling.
        env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q[state, action] stores the estimated future return
    # of taking action at state.
    Q = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        if seed is not None and episode == 0:
            # Seed the environment's internal RNG once.
            # Later resets continue from that seeded RNG stream.
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()
        done = False

        # Linearly decay epsilon from epsilon_start to epsilon_end.
        # Early training: more exploration.
        # Later training: more exploitation.
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / episodes,
        )

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_q = Q[state, action]

            # If the episode is done, there is no future value.
            # Otherwise, bootstrap from the best action at next_state.
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            td_error = target - current_q

            # Move the current Q estimate slightly toward the target.
            Q[state, action] = current_q + alpha * td_error

            state = next_state

    env.close()
    return Q


def train_q_learning_with_visit_decay(
    episodes=3000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=None,
):
    """
    Train tabular Q-learning with visit-count based learning rates.

    Difference from the previous version:
        Previous:
            alpha is constant, e.g. alpha = 0.1

        This version:
            alpha depends on how many times we have visited (state, action).

    For each state-action pair:
        N[state, action] counts visits.

    Learning rate:
        alpha = 1 / N[state, action]

    Why this matters:
        In stochastic environments, constant alpha keeps Q-values noisy forever.
        Decaying alpha makes estimates stabilize as evidence accumulates.

    Returns:
        Q:
            Learned Q-table.

        visit_counts:
            Table showing how often each state-action pair was visited.
    """
    if seed is not None:
        np.random.seed(seed)

    env = make_env()

    if seed is not None:
        env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q[state, action] estimates expected future return.
    Q = np.zeros((n_states, n_actions))

    # visit_counts[state, action] tracks how many times we updated this entry.
    visit_counts = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / episodes,
        )

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # TODO 1:
            # Increase visit count for this state-action pair.
            # visit_counts[state, action] = ?
            visit_counts[state, action] += 1

            # TODO 2:
            # Set alpha using visit count.
            # alpha = ?
            alpha = 1 / visit_counts[state, action]

            current_q = Q[state, action]

            # TODO 3:
            # Compute the Q-learning target.
            # If done: target = reward
            # Else: target = reward + gamma * max Q[next_state]
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            # TODO 4:
            # Compute TD error and update Q[state, action].
            td_error = target - current_q
            Q[state, action] = current_q + alpha * td_error

            state = next_state

    env.close()
    return Q, visit_counts


def train_q_learning_with_sqrt_decay(
    episodes=3000,
    alpha0=0.5,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    seed=None,
):
    """
    Train tabular Q-learning with a slower visit-count learning-rate decay.

    Learning rate:
        alpha = alpha0 / sqrt(N[state, action])

    Compared with alpha = 1 / N:
        - early learning is less extreme
        - later learning remains possible
        - useful reward signals can still propagate after many visits

    This is not the only possible schedule, but it is a useful practical
    compromise between constant alpha and fast 1/N decay.
    """
    if seed is not None:
        np.random.seed(seed)

    env = make_env()

    if seed is not None:
        env.action_space.seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    visit_counts = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / episodes,
        )

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # TODO 1:
            # Increase visit count for this state-action pair.
            visit_counts[state, action] += 1

            # TODO 2:
            # Compute alpha = alpha0 / sqrt(visit count).
            alpha = alpha0 / np.sqrt(visit_counts[state, action])

            # TODO 3:
            # Compute Q-learning target.
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            # TODO 4:
            # Compute TD error and update Q.
            td_error = target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error

            state = next_state

    env.close()
    return Q, visit_counts


def render_policy(Q):
    """
    Convert a learned Q-table into a readable 4x4 policy grid.

    For each state:
        choose the greedy action according to Q[state]
        convert that action into an arrow symbol

    Special cells:
        S = start
        H = hole
        G = goal

    This function helps us inspect what the agent actually learned.
    A high success rate is useful, but policy inspection helps reveal
    whether the learned behavior makes sense.
    """
    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑",
    }

    # FrozenLake 4x4 layout.
    # We use this only for visualization.
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
            cell_type = lake[row][col]

            # TODO:
            # If the cell is S, H, or G, keep that symbol.
            # Otherwise:
            #   1. choose the greedy action from Q[state]
            #   2. convert that action to an arrow
            #   3. append the arrow to policy_row
            if cell_type in ["S", "H", "G"]:
                policy_row.append(cell_type)
            else:
                # best_action = np.argmax(Q[state])
                best_actions = np.where(Q[state] == np.max(Q[state]))[0]
                best_action = np.random.choice(best_actions)
                symbol = action_symbols[best_action]
                policy_row.append(symbol)

        policy_grid.append(policy_row)

    return policy_grid

if __name__ == "__main__":
    Q = train_q_learning()

    print("Learned Q-table:")
    print(Q)

    print("\nLearned policy:")
    policy = render_policy(Q)
    for row in policy:
        print(" ".join(row))