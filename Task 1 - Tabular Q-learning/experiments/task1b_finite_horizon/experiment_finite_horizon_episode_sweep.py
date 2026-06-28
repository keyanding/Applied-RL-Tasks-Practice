import numpy as np

from eval.eval_finite_horizon import evaluate_finite_horizon_policy
from src.finite_horizon_q_learning import (
    print_policy_at_time,
    train_finite_horizon_q_learning,
)


def run_episode_sweep():
    """
    Compare finite-horizon Q-learning performance under different training budgets.

    Why this experiment matters:
        Our finite-horizon Q-table is much larger than the stationary Q-table:
            Q[t, state, action]

        That means 10,000 episodes may not be enough.

    We vary:
        training episodes

    We keep fixed:
        horizon
        alpha schedule
        epsilon schedule
        seeds
        evaluation episodes

    Goal:
        Check whether poor performance is mainly caused by insufficient training.
    """
    horizon = 100

    # Start with these. If 300k is too slow, stop at 100k first.
    episode_settings = [10_000, 50_000, 100_000, 300_000]

    # Use fewer seeds first to keep iteration fast.
    # Later, if a setting looks promising, we can run 5 or 10 seeds.
    seeds = [0, 1, 2]

    eval_episodes = 5000

    for train_episodes in episode_settings:
        rates = []

        print("=" * 80)
        print(f"Training episodes: {train_episodes}")

        for seed in seeds:
            Q, visit_counts = train_finite_horizon_q_learning(
                episodes=train_episodes,
                horizon=horizon,
                alpha0=0.5,
                gamma=1.0,
                epsilon_start=1.0,
                epsilon_end=0.05,
                seed=seed,
            )

            rate = evaluate_finite_horizon_policy(
                Q,
                episodes=eval_episodes,
                horizon=horizon,
                seed=1000 + seed,
            )

            rates.append(rate)

            print("-" * 60)
            print(f"Seed: {seed}")
            print(f"Success rate: {rate:.3f}")

            # Show the policy at key times for one seed only.
            # Otherwise the output becomes too long.
            if seed == 0:
                print("\nPolicy at t = 0:")
                print_policy_at_time(Q, 0)

                print("\nPolicy at t = 20:")
                print_policy_at_time(Q, 20)

                print("\nPolicy at t = 50:")
                print_policy_at_time(Q, 50)

                print("\nKey Q-values at t = 0, state 0:")
                print(Q[0, 0])

                print("\nKey Q-values at t = 20, state 9:")
                print(Q[20, 9])

                print("\nKey Q-values at t = 20, state 14:")
                print(Q[20, 14])

        rates = np.array(rates)

        print("-" * 60)
        print(f"Rates: {rates}")
        print(f"Mean/std: {rates.mean():.3f} / {rates.std():.3f}")


if __name__ == "__main__":
    run_episode_sweep()