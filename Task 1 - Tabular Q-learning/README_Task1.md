# Task 1: Tabular Q-learning and Failure Analysis in Stochastic FrozenLake

## Goal

Implement tabular Q-learning from scratch and analyze how the learned policy behaves under deterministic and stochastic dynamics.

## Algorithm

Q-learning learns a state-action value table Q(s, a). The update rule is:

Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

The agent uses epsilon-greedy exploration during training and a greedy policy during evaluation.

## Environments

- FrozenLake-v1, is_slippery=False
- FrozenLake-v1, is_slippery=True

## Key Results

### Deterministic FrozenLake

The learned policy achieved:

Success rate: 1.0

### Stochastic FrozenLake

With constant alpha = 0.1, performance varied significantly across seeds:

[0.718, 0.739, 0.738, 0.728, 0.243]

The low-performing seed was caused by a small Q-value ranking error at state 9:

Q(9, LEFT) = 0.6032
Q(9, DOWN) = 0.6016

Although the difference was only about 0.0017, choosing LEFT instead of DOWN reduced the success rate from about 0.73 to about 0.24.

A causal intervention forcing state 9 -> DOWN restored performance:

Normal rate: 0.243
Intervention rate: 0.732

## Alpha Schedule Comparison

Constant alpha:

Mean/std = 0.633 / 0.195

Visit-count decay alpha = 1/N(s,a):

Mean/std = 0.296 / 0.127

Sqrt decay alpha = alpha0/sqrt(N(s,a)):

Mean/std = 0.686 / 0.085

Sqrt decay improved robustness and recovered the bad seed 4 case.

## Exact Policy Evaluation

For sqrt-decay seed 2, Monte Carlo evaluation suggested that forcing state 0 -> LEFT improved performance.

Finite-horizon exact policy evaluation confirmed this:

Original policy: 0.510904
State 0 -> LEFT intervention: 0.729766

This showed that the bottleneck was a risk-sensitive decision at the start state. The action LEFT appears counterintuitive, but under slippery dynamics it biases the agent toward the safer lower path.

## Lessons

1. Q-learning convergence theory does not guarantee good finite-sample performance under arbitrary practical settings.
2. Optimal stochastic policies may look counterintuitive.
3. A small Q-value estimation error can cause a large policy-level failure at safety-critical states.
4. Aggregate metrics are not enough; policy inspection and causal intervention are essential.
5. Evaluation objective matters: infinite-horizon and finite-horizon success probabilities can differ.