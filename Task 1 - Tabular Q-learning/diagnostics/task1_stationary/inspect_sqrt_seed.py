import gymnasium as gym
import numpy as np

from src.q_learning import render_policy, train_q_learning_with_sqrt_decay

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def evaluate(Q, episodes=5000, seed=None):
    """
    Evaluate a greedy policy in slippery FrozenLake.

    This returns the empirical success probability of the learned policy.
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
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        if reward == 1:
            successes += 1

    env.close()
    return successes / episodes


def print_policy(Q):
    """
    Print the learned greedy policy as a 4x4 grid.
    """
    policy = render_policy(Q)
    for row in policy:
        print(" ".join(row))


def inspect_all_nonterminal_q_values(Q):
    """
    Print Q-values and greedy action for all non-terminal, non-hole states.

    This helps us identify states where:
        1. the greedy action looks risky
        2. two actions have very similar Q-values
        3. the policy differs from known good policies
    """
    safe_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]

    for state in safe_states:
        q_values = Q[state]
        best_action = int(np.argmax(q_values))

        print(f"state {state}:")
        print(f"  Q-values: {q_values}")
        print(f"  greedy action: {best_action} ({ACTION_NAMES[best_action]})")


def main():
    seed = 2
    episodes = 3000

    Q, visit_counts = train_q_learning_with_sqrt_decay(
        episodes=episodes,
        alpha0=0.5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=seed,
    )

    success_rate = evaluate(Q, episodes=5000, seed=1000 + seed)

    print("=" * 60)
    print("Sqrt decay inspection")
    print(f"Episodes: {episodes}")
    print(f"Seed: {seed}")
    print(f"Success rate: {success_rate:.3f}")

    print("\nLearned policy:")
    print_policy(Q)

    print("\nQ-values for all safe states:")
    inspect_all_nonterminal_q_values(Q)

    print("\nVisit counts for all safe states:")
    safe_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
    for state in safe_states:
        print(f"state {state}: {visit_counts[state]}")


if __name__ == "__main__":
    main()