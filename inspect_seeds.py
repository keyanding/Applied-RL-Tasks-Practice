import numpy as np
import gymnasium as gym

from q_learning import train_q_learning, render_policy


def evaluate(Q, episodes=5000, seed=None):
    """
    Evaluate a greedy policy in slippery FrozenLake.

    We reuse this helper here because we want to print both:
        1. success rate
        2. learned policy

    This lets us connect quantitative performance with actual behavior.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    if seed is not None:
        env.action_space.seed(seed)

    successes = 0

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        while not done:
            # Greedy evaluation: choose the best action according to Q.
            action = int(np.argmax(Q[state]))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            successes += 1

    env.close()
    return successes / episodes


def print_policy(policy):
    """
    Print a 4x4 policy grid in a readable format.
    """
    for row in policy:
        print(" ".join(row))


def inspect_seeds(episodes):
    """
    Train Q-learning with several seeds and inspect the learned policy.

    Goal:
        Identify whether low-performing seeds learn visibly different policies.

    This is a common RL debugging pattern:
        metric first,
        then policy / trajectory inspection,
        then hyperparameter change.
    """
    # seeds = [0, 1, 2, 3, 4]

    # for seed in seeds:
    Q = train_q_learning(
        episodes=episodes,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=4,
    )

    normal_rate = evaluate_with_action_override(
        Q,
        override_actions={},
        episodes=5000,
        seed=1004,
    )

    intervention_rate = evaluate_with_action_override(
        Q,
        override_actions={9: 1},  # force DOWN at state 9
        episodes=5000,
        seed=1004,
    )

    print("Normal rate:", normal_rate)
    print("Intervention rate:", intervention_rate)

    # success_rate = evaluate(Q, episodes=5000, seed=1000 + seed)

    # print("=" * 60)
    # print(f"Episodes: {episodes}, Seed: {seed}")
    # print(f"Success rate: {success_rate:.3f}")
    # print("Learned policy:")

    # policy = render_policy(Q)
    # print_policy(policy)
    # print("Q-values at key states:")
    # for state in [8, 9, 10, 13, 14]:
    #     print(f"state {state}: {Q[state]}")


def evaluate_with_action_override(Q, override_actions=None, episodes=5000, seed=None):
    """
    Evaluate a greedy policy, but optionally override actions at selected states.

    Args:
        Q:
            Learned Q-table.

        override_actions:
            A dictionary like {9: 1}.
            This means:
                when the agent is at state 9,
                force action 1, which is DOWN.

        episodes:
            Number of evaluation episodes.

        seed:
            Optional random seed for reproducible environment randomness.

    Why this is useful:
        If changing one state-action decision dramatically improves performance,
        then that state is likely a critical failure point.
    """
    env = gym.make("FrozenLake-v1", is_slippery=True)

    if seed is not None:
        env.action_space.seed(seed)

    if override_actions is None:
        override_actions = {}

    successes = 0

    for episode in range(episodes):
        if seed is not None and episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False

        while not done:
            # TODO:
            # If the current state is in override_actions,
            # use the forced action.
            #
            # Otherwise, use the normal greedy action from Q.
            if state in override_actions:
                action = override_actions[state]
            else:
                action = int(np.argmax(Q[state]))

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            successes += 1

    env.close()
    return successes / episodes


if __name__ == "__main__":
    inspect_seeds(episodes=3000)