import gymnasium as gym
import numpy as np

from src.q_learning import train_q_learning


def evaluate(Q, episodes=100):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    successes = 0

    for _ in range(episodes):
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


if __name__ == "__main__":
    Q = train_q_learning()
    success_rate = evaluate(Q)
    print("Success rate:", success_rate)