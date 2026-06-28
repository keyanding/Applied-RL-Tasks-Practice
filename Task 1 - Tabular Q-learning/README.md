# Tabular RL Failure Analysis: Q-learning and Finite-Horizon Control

This folder contains two related reinforcement-learning studies built around FrozenLake:

1. **Task 1: Stationary Tabular Q-learning**
2. **Task 1B: Finite-Horizon Q-learning**

The goal is not only to implement Q-learning, but also to practice a research-style workflow for diagnosing why learned policies fail.

---

## Project Structure

```text
Task 1 - Tabular Q-learning/
  README.md
  requirements.txt

  src/
    __init__.py
    q_learning.py
    finite_horizon_q_learning.py
    finite_horizon_q_learning_exploring_starts.py
    finite_horizon_dp.py

  eval/
    __init__.py
    eval.py
    eval_finite_horizon.py
    exact_policy_eval.py

  experiments/
    __init__.py

    task1_stationary/
      __init__.py
      experiment_slippery.py
      experiment_multiseed.py
      experiment_alpha_schedule.py
      experiment_three_alpha_schedules.py

    task1b_finite_horizon/
      __init__.py
      experiment_finite_horizon_episode_sweep.py
      experiment_exploring_starts.py

  diagnostics/
    __init__.py

    task1_stationary/
      __init__.py
      inspect_seeds.py
      inspect_sqrt_seed.py
      intervention_sweep.py

    task1b_finite_horizon/
      __init__.py
      occupancy_weighted_regret.py
      finite_horizon_targeted_intervention.py
      finite_horizon_window_intervention.py
      exploring_start_state_intervention.py
      analyze_state0_across_time.py
      analyze_exploring_starts_regret.py

  docs/
    README_Task1.md
    README_Task1B.md
    WORKFLOW_RL_DEBUGGING.md
    proof.md
```

---

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimum `requirements.txt`:

```text
gymnasium[toy-text]
numpy
```

Run all scripts from the project root.

Example:

```bash
python -m experiments.task1_stationary.experiment_multiseed
python -m experiments.task1b_finite_horizon.experiment_exploring_starts
python -m diagnostics.task1b_finite_horizon.occupancy_weighted_regret
```

---

## Task 1: Stationary Tabular Q-learning

Task 1 implements standard tabular Q-learning with a stationary Q-table:

```python
Q[state, action]
```

The learned policy is:

```python
action = argmax_a Q[state, a]
```

Main file:

```text
src/q_learning.py
```

Key ideas:

* Q-table
* epsilon-greedy exploration
* Bellman update
* deterministic vs stochastic environment
* multi-seed evaluation
* policy inspection
* Q-value inspection
* causal intervention

The Q-learning update is:

```python
if done:
    target = reward
else:
    target = reward + gamma * np.max(Q[next_state])

td_error = target - Q[state, action]
Q[state, action] += alpha * td_error
```

---

## Task 1 Key Results

In deterministic FrozenLake:

```text
Success rate: 1.0
```

In stochastic FrozenLake, performance varied significantly across random seeds.

With constant alpha:

```text
Success rates: [0.718, 0.739, 0.738, 0.728, 0.243]
```

The bad seed was caused by a tiny Q-value ranking error at a safety-critical state:

```text
state 9:
Q(LEFT) = 0.6032
Q(DOWN) = 0.6016
```

Although the difference was only about `0.0017`, choosing `LEFT` instead of `DOWN` reduced success from about `0.73` to about `0.24`.

A causal intervention confirmed the diagnosis:

```text
Normal rate:       0.2434
Intervention rate: 0.7322
```

This is a classic high-reliability failure mode: a small value-estimation error can cause a large policy-level failure at a safety-critical state.

---

## Task 1B: Finite-Horizon Q-learning

Task 1B studies finite-horizon Q-learning with a time-dependent Q-table:

```python
Q[t, state, action]
```

The learned policy is:

```python
action = argmax_a Q[t, state, a]
```

This is more expressive than stationary Q-learning because the best action can depend on how much time is left.

However, it also greatly increases sample complexity.

Stationary Q-table size:

```text
16 states × 4 actions = 64 entries
```

Finite-horizon Q-table size:

```text
101 time layers × 16 states × 4 actions = 6464 entries
```

The finite-horizon update is:

```python
if done:
    target = reward
else:
    target = reward + gamma * np.max(Q[t + 1, next_state])

td_error = target - Q[t, state, action]
Q[t, state, action] += alpha * td_error
```

The key difference is that the update bootstraps from the next time layer:

```python
Q[t + 1, next_state]
```

---

## Task 1B Key Results

Using dynamic programming with the known transition model `env.P`, the exact finite-horizon optimal value is:

```text
DP optimal success probability: 0.744190
```

Naive finite-horizon Q-learning from the true start state improved with more data:

```text
10k episodes:   0.197
50k episodes:   0.393
100k episodes:  0.475
300k episodes:  0.565
```

This shows that the algorithm is learning, but sample efficiency is poor.

---

## Exploring Starts

To improve coverage, we implemented exploring starts.

Instead of always starting from:

```text
t = 0
state = 0
```

some training episodes start from a random safe state and random time layer.

This improved performance substantially.

With 100k episodes:

```text
exploring_start_prob = 0.00: 0.462
exploring_start_prob = 0.25: 0.562
exploring_start_prob = 0.50: 0.592
exploring_start_prob = 0.75: 0.579
```

The best tested setting, `p=0.5`, outperformed naive 300k training using only one-third of the episodes.

This confirms that coverage of important `(t, state)` pairs was a major bottleneck.

---

## Failure Analysis Methods

This project uses several diagnostic tools:

### 1. Multi-seed evaluation

Single-seed results are not reliable in stochastic RL.

### 2. Policy visualization

A high aggregate success rate is not enough. We inspect the learned greedy policy.

### 3. Q-value inspection

We check whether action rankings are stable or nearly tied.

### 4. Exact policy evaluation

When a transition model is available, exact evaluation removes Monte Carlo noise.

### 5. Dynamic programming oracle

For small tabular MDPs, dynamic programming gives the ground-truth optimal policy.

### 6. Occupancy-weighted regret

Raw learned-vs-optimal disagreement can highlight unreachable states.

Occupancy-weighted regret prioritizes mistakes that are both reachable and costly:

```text
occupancy_weighted_regret(t, s)
=
Pr(policy reaches state s at time t)
×
[Q_opt(t,s,a_opt) - Q_opt(t,s,a_learned)]
```

### 7. Causal interventions

We test whether replacing selected learned actions with optimal actions improves value.

This distinguishes single-state failures from distributed time-state failures.

---

## Main Lessons

### Lesson 1: Correct representation can hurt sample efficiency

`Q[t, state, action]` is more expressive than `Q[state, action]`, but it is much harder to learn.

### Lesson 2: Aggregate metrics are not enough

Success rate alone does not explain why the agent succeeds or fails.

### Lesson 3: Small Q-value ranking errors can be catastrophic

In stochastic environments, two actions can have very similar Q-values but very different risk profiles.

### Lesson 4: Failure modes differ across representations

Task 1 had a local single-state failure.

Task 1B had distributed time-state ranking errors.

### Lesson 5: Coverage matters

Exploring starts improved sample efficiency by directly training under-visited time-state pairs.

### Lesson 6: Model-based oracles are valuable

When a model exists, dynamic programming or MPC can provide a strong benchmark for model-free RL.

---

## Connection to Robotics and Flight Control

The lessons transfer directly to robot control, aircraft control, and eVTOL autonomy.

Adding variables such as time-to-go, energy margin, altitude, distance-to-target, or mode can make a controller more expressive, but also increases the training coverage problem.

Practical high-reliability RL systems need:

```text
scenario coverage
targeted simulator resets
multi-seed evaluation
model-based oracle benchmarks
occupancy-weighted failure analysis
causal intervention tests
risk-sensitive evaluation
```

This project is a small tabular example of the same workflow.
