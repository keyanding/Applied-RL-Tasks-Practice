# Task 1B: Finite-Horizon Q-learning, Coverage, and Risk-Sensitive Failure Analysis

## 1. Goal

This task studies finite-horizon tabular Q-learning on the stochastic FrozenLake environment.

In Task 1, we trained a stationary Q-learning agent with a Q-table:

```python
Q[state, action]
```

This means the learned policy is stationary:

```python
action = argmax_a Q[state, a]
```

However, in a finite-horizon problem, the same physical state can require different actions depending on how much time is left. For example, being at the start state with 100 steps remaining is different from being at the start state with only 5 steps remaining.

Therefore, in Task 1B, we use a time-dependent Q-table:

```python
Q[t, state, action]
```

The corresponding policy is non-stationary:

```python
action = argmax_a Q[t, state, a]
```

The main goal is to understand both the benefit and cost of this more expressive finite-horizon representation.

---

## 2. Environment

We use:

```python
FrozenLake-v1, is_slippery=True
```

The map is:

```text
S F F F
F H F H
F F F H
H F F G
```

State indexing:

```text
0   1   2   3
4   5   6   7
8   9   10  11
12  13  14  15
```

Special states:

```text
S = start = state 0
G = goal  = state 15
H = holes = states 5, 7, 11, 12
```

Action encoding:

```python
0 = LEFT
1 = DOWN
2 = RIGHT
3 = UP
```

Because `is_slippery=True`, the intended action is not always the actual movement direction. This makes optimal behavior risk-sensitive. The shortest-looking path is not always the safest path.

---

## 3. Algorithm: Finite-Horizon Q-learning

The finite-horizon Q-table has shape:

```python
Q[horizon + 1, n_states, n_actions]
```

For this task:

```python
horizon = 100
n_states = 16
n_actions = 4
```

So the Q-table has:

```text
101 × 16 × 4 = 6464 entries
```

This is much larger than the stationary Q-table from Task 1:

```text
16 × 4 = 64 entries
```

The update rule is:

```python
if done:
    target = reward
else:
    target = reward + gamma * max_a Q[t + 1, next_state, a]

td_error = target - Q[t, state, action]
Q[t, state, action] += alpha * td_error
```

The key finite-horizon difference is that we bootstrap from the next time layer:

```python
Q[t + 1, next_state]
```

instead of using a time-independent value estimate.

We use a square-root visit-count learning-rate schedule:

```python
alpha = alpha0 / sqrt(N[t, state, action])
```

This is slower-decaying than `1/N` and worked better in Task 1.

---

## 4. Why Finite-Horizon Q-learning Is More Correct but Harder

The finite-horizon representation is more correct because the optimal action can depend on the remaining time.

For example:

```text
At state 0 with many steps left:
    A conservative boundary-using action may be optimal.

At state 0 with very few steps left:
    Waiting may no longer be useful.
```

However, the cost is much higher sample complexity.

Stationary Q-learning only needs to learn:

```python
Q[state, action]
```

Finite-horizon Q-learning must learn:

```python
Q[t, state, action]
```

This means the agent has to learn separate action values for the same physical state at many different time layers.

This creates a coverage problem: many `(t, state, action)` entries are rarely visited, especially later time layers or states that are only reachable under certain stochastic trajectories.

---

## 5. Dynamic Programming Oracle

Because FrozenLake exposes the transition model through `env.P`, we computed the exact finite-horizon optimal policy using backward dynamic programming.

The DP recurrence is:

```python
Q[t, s, a] = sum_p p * (reward + gamma * V[t + 1, next_state])
V[t, s] = max_a Q[t, s, a]
```

If a transition terminates, only the immediate reward counts:

```python
if terminated:
    contribution = prob * reward
else:
    contribution = prob * (reward + gamma * V[t + 1, next_state])
```

The DP optimal value from the start state is:

```text
DP optimal success probability: 0.744190
```

This serves as a ground-truth benchmark for evaluating learned finite-horizon policies.

---

## 6. Naive Finite-Horizon Q-learning Results

We first trained finite-horizon Q-learning from the true initial condition only:

```text
t = 0
state = 0
```

Episode sweep results:

```text
10k episodes:
Rates: [0.1954, 0.2084, 0.1870]
Mean/std: 0.197 / 0.009

50k episodes:
Rates: [0.4036, 0.3838, 0.3928]
Mean/std: 0.393 / 0.008

100k episodes:
Rates: [0.4904, 0.4768, 0.4578]
Mean/std: 0.475 / 0.013

300k episodes:
Rates: [0.5646, 0.5628, 0.5682]
Mean/std: 0.565 / 0.002
```

The policy improves monotonically with more data, which shows that the algorithm is learning. However, even after 300k episodes, it is still far below the DP optimal value:

```text
DP optimal:       0.744190
300k Q-learning:  0.565
Gap:              about 0.179
```

This suggests that the main difficulty is sample efficiency.

---

## 7. Diagnosis: Learned Policy vs DP Optimal Policy

For the 300k-episode learned policy, exact evaluation gave:

```text
DP optimal success probability:     0.744190
Learned policy success probability: 0.566090
Gap:                                0.178100
```

At `t=0, state=0`, the learned policy was already correct:

```text
state 0:
  learned action: LEFT
  optimal action: LEFT
  one-step regret: 0.000000
```

So the remaining gap was not caused by the initial action.

However, many other `(t, state)` entries were still under-learned. A raw learned-vs-optimal action disagreement list was not sufficient, because it included states that were impossible to reach at certain times, such as `t=0, state=13`.

Therefore, we used occupancy-weighted regret.

---

## 8. Occupancy-Weighted Regret

Raw one-step regret is:

```text
Q_opt[t, s, a_opt] - Q_opt[t, s, a_learned]
```

But this does not account for whether the learned policy actually reaches `(t, s)`.

So we computed:

```text
occupancy-weighted regret
=
Pr(learned policy reaches state s at time t)
×
[Q_opt[t,s,a_opt] - Q_opt[t,s,a_learned]]
```

This gives more meaningful failure analysis because it prioritizes reachable and high-impact mistakes.

For the 300k naive finite-horizon policy, the top occupancy-weighted mistakes included:

```text
t=43, state=4:  DOWN  -> LEFT
t=46, state=4:  DOWN  -> LEFT
t=47, state=4:  DOWN  -> LEFT
t=49, state=4:  DOWN  -> LEFT
t=49, state=9:  RIGHT -> DOWN
t=51, state=13: DOWN  -> RIGHT
t=51, state=8:  DOWN  -> UP
```

These mistakes were concentrated around the safe path:

```text
0 -> 4 -> 8 -> 9 -> 13 -> 14 -> G
```

The learned policy had learned the broad route, but it had not reliably learned the risk-sensitive action ranking along this route.

---

## 9. Causal Interventions

We tested whether fixing only a few high-regret `(t, state)` actions would recover the value gap.

Targeted fixes gave only small improvements:

```text
Baseline learned value: 0.566090

Top 1 targeted fix:
value:       0.567198
improvement: 0.001107

Top 8 targeted fixes:
value:       0.572186
improvement: 0.006096
```

This shows that Task 1B does not have a single critical-action failure like Task 1 seed 4. Instead, the gap is distributed across many time-state ranking errors.

We then tried broader interventions.

Time-window interventions:

```text
Replace t=00..19: value=0.566283, improvement=0.000192
Replace t=20..39: value=0.577763, improvement=0.011673
Replace t=40..59: value=0.590530, improvement=0.024439
Replace t=60..79: value=0.570404, improvement=0.004313
Replace t=80..99: value=0.566772, improvement=0.000681
```

The largest time-window improvement came from `t=40..59`, but it still explained only a small portion of the total gap.

State-set interventions were more informative:

```text
Replace states=[4, 8, 9, 13]:
value=0.585180
improvement=0.019089

Replace states=[4, 8, 9, 13, 14]:
value=0.619633
improvement=0.053542

Replace states=[0, 4, 8, 9, 13, 14]:
value=0.702560
improvement=0.136470

Replace states=[1, 2, 4, 8, 9, 10, 13, 14]:
value=0.649804
improvement=0.083713
```

The key result is:

```text
Replacing states [0, 4, 8, 9, 13, 14]
raises value from 0.566090 to 0.702560.
```

This explains most of the gap.

A deeper state-0 analysis showed that although `t=0, state=0` was correct, state 0 was still often wrong at later time layers:

```text
t=26: learned=RIGHT, optimal=LEFT
t=28: learned=DOWN,  optimal=LEFT
t=30: learned=DOWN,  optimal=LEFT
t=34: learned=DOWN,  optimal=LEFT
t=43: learned=RIGHT, optimal=LEFT
```

This means state 0 is not just the initial state. It is a recurring boundary state that the agent may revisit due to stochastic dynamics.

---

## 10. Exploring Starts

To improve coverage, we implemented exploring starts.

Instead of always starting from:

```text
t = 0
state = 0
```

we sometimes start from a random safe state and a random time layer:

```python
if random() < exploring_start_prob:
    t_start = random time in [0, horizon)
    state_start = random safe state
else:
    t_start = 0
    state_start = 0
```

This is a simulator-only technique, but it corresponds to common engineering practice in robot and flight-control training:

```text
randomized initial conditions
scenario coverage
domain randomization
targeted simulator resets
```

It directly improves coverage of under-visited time-state pairs such as:

```text
t=40, state=4
t=50, state=9
t=60, state=13
```

---

## 11. Exploring Starts Results

Using 100k episodes:

```text
Exploring start probability p=0.00:
Mean/std: 0.461869 / 0.011963
Gap to optimal: 0.282321

p=0.25:
Mean/std: 0.562484 / 0.018153
Gap to optimal: 0.181706

p=0.50:
Mean/std: 0.591884 / 0.018602
Gap to optimal: 0.152307

p=0.75:
Mean/std: 0.579352 / 0.012060
Gap to optimal: 0.164838
```

The best setting tested was:

```text
exploring_start_prob = 0.5
```

This result is important because:

```text
100k episodes + exploring starts p=0.5 -> about 0.592
300k episodes + naive start only       -> about 0.565
```

So exploring starts outperform naive training using only one-third as many episodes.

This confirms that coverage of important `(t, state)` pairs was a major bottleneck.

---

## 12. Remaining Errors After Exploring Starts

For `exploring_start_prob=0.5`, seed 0:

```text
Optimal value: 0.744190
Learned value: 0.617400
Gap:           0.126790
```

Top occupancy-weighted mistakes changed. They now included:

```text
t=42, state=8:  DOWN  -> UP
t=27, state=2:  RIGHT -> UP
t=31, state=2:  RIGHT -> UP
t=26, state=2:  RIGHT -> UP
t=13, state=0:  DOWN  -> LEFT
```

This suggests that exploring starts improved coverage, but the remaining failures were mostly fine-grained action-ranking errors in risk-sensitive states:

```text
state 0
state 2
state 8
```

State 2 is especially interesting because it lies on the upper path. The optimal action is often `UP`, which looks counterintuitive, but it uses the boundary to reduce the probability of entering dangerous regions near holes.

---

## 13. Exploring-Start Intervention Analysis

For the exploring-start policy:

```text
Baseline learned value: 0.617400
Optimal value:          0.744190
Gap:                    0.126790
```

State interventions:

```text
Replace states=[0]:
value=0.694707
improvement=0.077306
remaining_gap=0.049484

Replace states=[2]:
value=0.655318
improvement=0.037918
remaining_gap=0.088872

Replace states=[8]:
value=0.623403
improvement=0.006003
remaining_gap=0.120788

Replace states=[0, 2]:
value=0.713120
improvement=0.095720
remaining_gap=0.031071

Replace states=[0, 2, 8]:
value=0.726276
improvement=0.108875
remaining_gap=0.017915

Replace states=[0, 2, 8, 4, 9, 13, 14]:
value=0.738143
improvement=0.120743
remaining_gap=0.006047

Replace states=[0, 1, 2, 4, 8, 9, 10, 13, 14]:
value=0.743674
improvement=0.126274
remaining_gap=0.000516
```

The most important result is:

```text
Replacing states [0, 2, 8]
recovers about 86% of the remaining gap.
```

Calculation:

```text
0.108875 / 0.126790 ≈ 85.9%
```

So after improving coverage with exploring starts, most remaining error is concentrated in a small set of risk-sensitive physical states across time.

Time-window interventions also showed that the most important window was:

```text
t = 20..39
```

Results:

```text
Replace t=00..19:
improvement=0.009603

Replace t=20..39:
improvement=0.050651

Replace t=40..59:
improvement=0.027397

Replace t=60..79:
improvement=0.009575

Replace t=80..99:
improvement=0.000710
```

This matches the state-level analysis: the agent often visits state 0 and state 2 during the early-to-mid horizon, and small action-ranking errors there have large downstream effects.

---

## 14. Comparison with Task 1 Failure Mode

Task 1 had a stationary Q-table:

```python
Q[state, action]
```

Its failure mode was often local and causal. For example, in seed 4, state 9 had nearly tied Q-values:

```text
Q(9, LEFT) = 0.6032
Q(9, DOWN) = 0.6016
```

The greedy policy chose `LEFT`, but forcing state 9 to choose `DOWN` restored performance:

```text
Normal rate:       0.2434
Intervention rate: 0.7322
```

This was a single-state causal bug.

Task 1B is different. The finite-horizon Q-table is:

```python
Q[t, state, action]
```

Now the same physical state appears many times across the horizon. The policy must learn the correct action for each `(t, state)` pair.

Therefore, the failure mode becomes distributed:

```text
state 0 at t=13 may be wrong
state 0 at t=21 may be wrong
state 2 at t=27 may be wrong
state 8 at t=42 may be wrong
...
```

Fixing a single `(t, state)` action does not recover much value because the policy may still make related mistakes in nearby time layers or downstream states.

In other words:

```text
Task 1:
    one physical state -> one action ranking
    failure can be a single-state action flip

Task 1B:
    one physical state -> many time-dependent action rankings
    failure becomes distributed across time-state pairs
```

Also, earlier mistakes affect later learning and evaluation through occupancy. If the agent chooses a suboptimal action at state 0 or state 2 in the early-to-mid horizon, it changes which later states are visited. This can reduce coverage of the safe path and create downstream learning gaps.

---

## 15. Main Lessons

### Lesson 1: More expressive state representations can reduce sample efficiency

Adding time makes the representation more correct:

```python
Q[t, state, action]
```

But it increases the number of values to learn from 64 to 6464.

This creates a much harder coverage problem.

---

### Lesson 2: Exact DP is a useful oracle when a model exists

Because FrozenLake exposes `env.P`, we could compute the exact finite-horizon optimal policy.

This gave us a reliable benchmark:

```text
DP optimal value = 0.744190
```

Without this benchmark, it would be harder to know whether Q-learning was close to optimal or still far away.

---

### Lesson 3: Occupancy-weighted regret is better than raw disagreement

Raw learned-vs-optimal disagreement can highlight unreachable states.

Occupancy-weighted regret focuses on mistakes that are both:

```text
reachable under the learned policy
and costly under the optimal Q-function
```

This produces a much more useful failure analysis.

---

### Lesson 4: Local interventions distinguish failure modes

In Task 1, fixing one state could almost completely repair the policy.

In Task 1B, fixing a few individual time-state mistakes gave only small improvements. This revealed that the failure was distributed, not a single local bug.

---

### Lesson 5: Exploring starts improve coverage

Exploring starts substantially improved performance:

```text
100k naive training:             about 0.462
100k with exploring starts p=0.5: about 0.592
```

This shows that targeted initial condition sampling can be more effective than simply increasing the number of episodes from the true start state.

---

### Lesson 6: Remaining errors are risk-sensitive ranking errors

After exploring starts, the remaining gap was mostly explained by a few physical states:

```text
state 0
state 2
state 8
```

These are states where the optimal action is not necessarily the most direct action toward the goal. Instead, the optimal action often uses boundaries and stochastic transitions to reduce risk.

---

## 16. Connection to Robotics and Flight Control

This task has direct lessons for robot control, aircraft control, and eVTOL autonomy.

In high-reliability control, adding variables such as time-to-go, energy margin, distance-to-target, or remaining altitude can make the state representation more correct. But this also expands the learning problem.

For example, a controller may need to condition on:

```text
position
velocity
altitude
energy
time remaining
distance to target
wind disturbance
system mode
```

This can improve optimality, but only if the training process covers the relevant state-time combinations.

Therefore, practical systems need:

```text
scenario coverage
targeted simulator resets
model-based oracle benchmarks
occupancy-weighted failure analysis
causal intervention tests
risk-sensitive evaluation
```

This also suggests that pure model-free RL may not be the best first choice when a dynamics model is available. Model-based planning, MPC, dynamic programming, or hybrid model-based/model-free methods can be more sample-efficient and easier to validate.

---

## 17. Final Summary

Task 1B showed that finite-horizon Q-learning is conceptually more correct than stationary Q-learning for time-limited tasks, but it introduces a major sample-efficiency problem.

The main experimental findings were:

```text
DP optimal finite-horizon value:
0.744190

Naive finite-horizon Q-learning:
10k   -> 0.197
50k   -> 0.393
100k  -> 0.475
300k  -> 0.565

Exploring-start finite-horizon Q-learning with 100k episodes:
p=0.00 -> 0.462
p=0.25 -> 0.562
p=0.50 -> 0.592
p=0.75 -> 0.579
```

The best exploring-start policy still had a gap to optimal, but intervention analysis showed that most of the remaining gap came from a small number of risk-sensitive physical states across time.

Key intervention result:

```text
Baseline exploring-start value:
0.617400

Replace states [0, 2, 8]:
0.726276

Replace states [0, 2, 8, 4, 9, 13, 14]:
0.738143

DP optimal:
0.744190
```

The central lesson is:

```text
Finite-horizon representation improves correctness,
but reliable learning requires coverage, diagnostics, and risk-sensitive failure analysis.
```
