import gymnasium as gym
import numpy as np

from src.q_learning import (
    render_policy,
    train_q_learning_with_visit_decay,
)

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}


def evaluate_with_action_override(Q, override_actions=None, episodes=5000, seed=None):
    """
    Evaluate a greedy policy with optional state-action overrides.

    Args:
        Q:
            Learned Q-table.

        override_actions:
            Dictionary mapping state -> forced action.
            Example:
                {9: 1}
            means:
                if the agent is at state 9, force action DOWN.

        episodes:
            Number of stochastic evaluation episodes.

        seed:
            Seed for reproducible environment randomness.

    Why this function matters:
        It lets us test causal hypotheses.
        If overriding one state's action greatly improves success rate,
        that state is likely a bottleneck.
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
            # If this state has a forced action, use it.
            # Otherwise use the learned greedy policy.
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


def print_policy(Q):
    """
    Print the learned greedy policy as a 4x4 grid.
    """
    policy = render_policy(Q)
    for row in policy:
        print(" ".join(row))


def run_single_state_intervention_sweep():
    """
    For a bad-performing learned policy, test all one-state action overrides.

    We train sqrt-decay Q-learning with seed 2, because this is the case
    whose success rate was only about 0.516.

    For each safe state:
        try forcing LEFT, DOWN, RIGHT, UP

    Then compare each intervention against the normal greedy policy.
    """
    seed = 4 #2
    episodes = 3000
    eval_seed = 1000 + seed

    Q, visit_counts = train_q_learning_with_visit_decay(
        episodes=episodes,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=seed,
    )

    safe_states = [0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]

    # baseline_rate = evaluate_with_action_override(
    #     Q,
    #     override_actions={},
    #     episodes=5000,
    #     seed=eval_seed,
    # )
    baseline_rate = evaluate_with_action_override(Q, {}, episodes=5000, seed=eval_seed)

    state0_left_rate = evaluate_with_action_override(
        Q,
        override_actions={0: 0},  # force LEFT at state 0
        episodes=5000,
        seed=eval_seed,
    )

    print("=" * 60)
    print("Visit decay seed 4")
    print(f"Baseline success rate: {baseline_rate:.3f}")
    print(f"State 0 left success rate: {state0_left_rate:.3f}")
    
    print("\nLearned policy:")
    print_policy(Q)

    print("\nSingle-state override results:")
    print("Only showing interventions that improve over baseline.")

    improvements = []

    for state in safe_states:
        learned_action = int(np.argmax(Q[state]))

        for action in range(4):
            if action == learned_action:
                continue

            rate = evaluate_with_action_override(
                Q,
                override_actions={state: action},
                episodes=5000,
                seed=eval_seed,
            )

            improvement = rate - baseline_rate

            if improvement > 0.02:
                improvements.append((improvement, state, learned_action, action, rate))

    # Sort by largest improvement first.
    improvements.sort(reverse=True)

    for improvement, state, learned_action, action, rate in improvements:
        print(
            f"state {state}: "
            f"{ACTION_NAMES[learned_action]} -> {ACTION_NAMES[action]}, "
            f"rate = {rate:.3f}, "
            f"improvement = {improvement:.3f}"
        )


if __name__ == "__main__":
    run_single_state_intervention_sweep()