# Reusable Workflow: RL Debugging and Failure Analysis

This document summarizes a reusable workflow for debugging reinforcement-learning agents.

It was developed during Task 1 and Task 1B on stochastic FrozenLake, but the same workflow can be reused for later tasks such as DQN, PPO, robot control, aircraft control, and eVTOL simulation.

---

## 1. Start with the Smallest Working Agent

Before adding complexity, first build the smallest agent that can close the RL loop:

```text
observe state
choose action
step environment
receive reward and next state
update value or policy
repeat
```

For tabular Q-learning, this means implementing:

```text
Q-table
epsilon-greedy action selection
Bellman update
training loop
evaluation loop
```

The goal is not to optimize immediately. The goal is to make the learning loop observable and debuggable.

---

## 2. Separate Core Code, Experiments, and Diagnostics

A clean structure helps research iteration.

Recommended structure:

```text
src/
  core algorithms

eval/
  evaluation helpers

experiments/
  parameter sweeps and comparisons

diagnostics/
  failure analysis scripts

docs/
  writeups and notes
```

Do not mix core algorithm code with one-off diagnostic scripts.

A useful rule:

```text
src/ contains reusable code.
experiments/ asks "what happens if we vary X?"
diagnostics/ asks "why did this policy fail?"
docs/ explains what we learned.
```

---

## 3. Always Start with a Baseline

Run the simplest baseline first.

Example:

```text
deterministic FrozenLake
stationary Q-learning
constant alpha
single evaluation
```

Expected result:

```text
success rate should be high
policy should be sensible
```

If the baseline does not work, do not move to a harder setting yet.

---

## 4. Make the Environment Harder One Dimension at a Time

Avoid changing multiple things at once.

Example progression:

```text
deterministic environment
→ stochastic environment
→ multi-seed evaluation
→ learning-rate comparison
→ finite-horizon representation
→ exploring starts
```

This makes it possible to identify which change caused the new failure.

---

## 5. Do Not Trust One Seed

Stochastic RL can vary dramatically across random seeds.

A single seed can make an algorithm look either much better or much worse than it is.

Use multi-seed evaluation:

```python
seeds = [0, 1, 2, 3, 4]
```

Report:

```text
mean
standard deviation
min
max
individual seed results
```

Typical issue:

```text
One seed performs badly while others are good.
```

Useful diagnosis:

```text
Inspect the learned policy for the bad seed.
Compare Q-values at critical states.
Run causal interventions.
```

---

## 6. Inspect the Learned Policy

Aggregate success rate is not enough.

Always inspect what the agent actually learned.

For grid-world tasks, render a policy grid:

```text
S(←) ↑ ↑ ↑
← H ← H
↑ ↓ ← H
H → ↓ G
```

Look for:

```text
actions leading toward holes
default argmax artifacts
counterintuitive but risk-sensitive actions
states where the policy differs across seeds
```

Important warning:

```text
A strange-looking policy may still be optimal in a stochastic environment.
```

For example, choosing `LEFT` or `UP` near a boundary may be optimal because it uses the boundary to reduce risk.

---

## 7. Inspect Q-values, Not Just Actions

A greedy action can hide unstable value estimates.

For key states, print:

```python
print(Q[state])
```

or for finite-horizon tasks:

```python
print(Q[t, state])
```

Look for nearly tied values:

```text
Q(LEFT) = 0.6032
Q(DOWN) = 0.6016
```

A tiny difference can flip the greedy action.

In safety-critical settings, small ranking errors can produce large behavior differences.

---

## 8. Distinguish Evaluation Noise from Policy Failure

Monte Carlo evaluation has sampling noise.

If possible, increase evaluation episodes:

```python
evaluate(policy, episodes=5000)
```

If the environment exposes a transition model, use exact policy evaluation.

Monte Carlo answers:

```text
How did this policy perform under sampled rollouts?
```

Exact evaluation answers:

```text
What is the true expected value of this fixed policy under the model?
```

Use exact evaluation when available to avoid confusing sampling noise with policy failure.

---

## 9. Use an Oracle Benchmark When Available

For small tabular MDPs, dynamic programming can compute the optimal policy exactly.

This gives a ground-truth benchmark:

```text
DP optimal value
learned policy value
gap
```

Without an oracle, it is hard to know whether the learned policy is close to optimal or still far away.

In larger systems, possible oracle or reference baselines include:

```text
dynamic programming
MPC
LQR
PID controller
expert policy
rule-based safety controller
offline planner
```

---

## 10. Diagnose Raw Disagreement Carefully

Comparing learned policy against optimal policy is useful, but raw disagreement can be misleading.

Example:

```text
learned action differs from optimal at t=0, state=13
```

But if the agent starts at state 0, then `t=0, state=13` is unreachable.

Raw disagreement may highlight states that do not matter.

Therefore, prefer occupancy-weighted analysis.

---

## 11. Use Occupancy-Weighted Regret

Occupancy means:

```text
Pr(policy reaches state s at time t)
```

One-step regret means:

```text
Q_opt[t,s,a_opt] - Q_opt[t,s,a_learned]
```

Occupancy-weighted regret is:

```text
occupancy(t,s) × one_step_regret(t,s)
```

This prioritizes mistakes that are:

```text
reachable
costly
relevant to the learned policy's actual behavior
```

Use it to identify high-impact failure locations.

---

## 12. Run Causal Interventions

After identifying a suspected failure point, test it.

For stationary policies:

```python
override_actions = {
    state: forced_action
}
```

For finite-horizon policies:

```python
override_actions = {
    (t, state): forced_action
}
```

Then compare:

```text
baseline value
intervened value
improvement
remaining gap
```

This answers:

```text
Is this suspected mistake actually causal?
```

Important interpretation:

```text
If fixing one action recovers most value:
    failure is local and causal.

If fixing one action gives little improvement:
    failure is distributed or downstream errors remain.
```

---

## 13. Distinguish Local Failure from Distributed Failure

Task 1 showed a local failure:

```text
One critical state had the wrong greedy action.
Fixing that one state restored performance.
```

Task 1B showed a distributed failure:

```text
Many time-state pairs had small ranking errors.
Fixing one or a few pairs gave little improvement.
Fixing key physical states across all time layers recovered most value.
```

This distinction matters.

For local failures, targeted patches may work.

For distributed failures, you need better coverage, better representation, better training, or model-based guidance.

---

## 14. Check for Representation-Induced Sample Complexity

A more expressive state representation may be theoretically correct but practically harder to learn.

Example:

```python
Q[state, action]
```

versus:

```python
Q[t, state, action]
```

The second representation can express time-dependent behavior, but it increases the number of values to learn.

Before concluding that the algorithm is bad, check:

```text
How many entries does the value function have?
How many are actually visited?
Are key time-state-action pairs under-covered?
Does performance improve with more episodes?
```

---

## 15. Run Episode Sweeps Before Drawing Conclusions

If performance is poor, first test whether more training helps.

Example:

```text
10k episodes
50k episodes
100k episodes
300k episodes
```

Possible interpretations:

```text
Performance improves steadily:
    algorithm is learning, but sample efficiency is low.

Performance stays flat:
    possible bug, bad exploration, bad reward, or wrong objective.

Performance gets worse:
    possible instability, learning-rate issue, or policy extraction issue.
```

---

## 16. Check the Training Distribution

An RL agent only learns from the states it visits.

If the training process rarely visits important states, the policy may fail there.

Useful diagnostics:

```text
visit counts
state occupancy
time-state occupancy
coverage heatmaps
```

Typical issue:

```text
The agent starts from the same initial state every episode.
Many important mid-horizon states are rarely visited.
```

Possible fix:

```text
exploring starts
randomized initial states
curriculum learning
scenario-based resets
prioritized replay
targeted data collection
```

---

## 17. Use Exploring Starts When the Simulator Allows It

Exploring starts means some episodes begin from randomized states or time layers.

Example:

```python
if random() < exploring_start_prob:
    t_start = random time
    state_start = random safe state
else:
    t_start = 0
    state_start = 0
```

This improves coverage.

In robotics or flight control, this corresponds to:

```text
randomized initial conditions
scenario library sampling
domain randomization
simulator reset to edge cases
targeted coverage of known weak regions
```

But do not overuse it.

If exploring-start probability is too high, training distribution may become too different from the true evaluation distribution.

---

## 18. Align Training Objective and Evaluation Objective

Check whether the agent is trained for the same objective used in evaluation.

Potential mismatch:

```text
Training: infinite-horizon return
Evaluation: finite-horizon success
```

or:

```text
Training: final success
Evaluation: final success + safety constraints + time limit
```

If the task is finite-horizon, consider including time or remaining horizon in the state:

```python
Q[t, state, action]
```

or for function approximation:

```python
Q(state, remaining_time, action)
```

---

## 19. Watch for Default Argmax Artifacts

When Q-values are all zero:

```python
np.argmax([0, 0, 0, 0])
```

returns:

```text
0 = LEFT
```

A policy grid full of `LEFT` may not mean the agent learned to go left. It may mean the Q-table entries were never updated.

Always check visit counts before interpreting policy visualization.

---

## 20. Common Problems and Fixes

### Problem 1: Success rate is low

Check:

```text
Is the environment stochastic?
Is reward sparse?
Are episodes long enough?
Is exploration sufficient?
Are terminal rewards handled correctly?
```

Fixes:

```text
increase episodes
increase evaluation episodes
inspect learned policy
inspect Q-values
try multi-seed evaluation
```

---

### Problem 2: One seed is much worse than others

Check:

```text
policy for the bad seed
Q-values at critical states
near ties between actions
```

Fixes:

```text
causal intervention
different alpha schedule
more seeds
more training
risk-sensitive diagnostics
```

---

### Problem 3: Training longer does not monotonically improve performance

Possible causes:

```text
constant alpha keeps Q-values noisy
stochastic transitions
greedy policy sensitive to small ranking errors
single-seed variance
```

Fixes:

```text
multi-seed runs
slower learning-rate decay
Q-value inspection
policy exact evaluation
```

---

### Problem 4: `alpha = 1/N` performs badly

Possible cause:

```text
learning rate decays too fast before sparse reward has propagated
early inaccurate targets get locked in
```

Fixes:

```text
use constant alpha
use alpha0 / sqrt(N)
try slower decay
increase exploration
```

---

### Problem 5: Finite-horizon Q-learning performs much worse than stationary Q-learning

Possible causes:

```text
Q-table is much larger
time-state-action coverage is sparse
reward signal propagates slowly across time layers
many time layers are never visited
```

Fixes:

```text
episode sweep
visit-count diagnostics
dynamic programming benchmark
exploring starts
time-window intervention
state-set intervention
```

---

### Problem 6: Learned policy differs from optimal policy in many places

Check:

```text
Are those states reachable?
What is their occupancy?
What is their one-step regret?
```

Fix:

```text
use occupancy-weighted regret
ignore unreachable disagreements
focus on reachable high-regret mistakes
```

---

### Problem 7: Fixing top one-step mistakes barely improves value

Possible cause:

```text
failure is distributed
downstream learned policy still makes mistakes
single-action fix does not change the whole trajectory distribution enough
```

Fixes:

```text
replace a time window
replace a physical state across all time layers
replace a set of key states
compare causal improvements
```

---

### Problem 8: Exploring starts help, but not enough

Possible cause:

```text
coverage improved, but risk-sensitive action ranking remains difficult
```

Fixes:

```text
analyze remaining occupancy-weighted regret
try targeted exploring starts around high-regret states
improve learning-rate schedule
increase training
use model-based planning or hybrid methods
```

---

## 21. Recommended Debugging Order

Use this order for future RL tasks:

```text
1. Run baseline.
2. Confirm the learning loop works.
3. Evaluate with multiple seeds.
4. Inspect learned policy.
5. Inspect Q-values or action scores.
6. Increase evaluation episodes or use exact evaluation.
7. Compare against an oracle if available.
8. Compute occupancy / visit counts.
9. Compute occupancy-weighted regret.
10. Run causal interventions.
11. Run episode sweeps.
12. Improve coverage through exploring starts or scenario sampling.
13. Re-run diagnostics.
14. Write down the failure mode and lesson.
```

---

## 22. Research-Style Summary Template

For every RL task, write results in this format:

```text
Environment:
    ...

Algorithm:
    ...

Representation:
    ...

Training setup:
    ...

Evaluation setup:
    ...

Baseline result:
    ...

Main failure:
    ...

Diagnostics used:
    ...

Causal evidence:
    ...

Fix attempted:
    ...

Result after fix:
    ...

Remaining gap:
    ...

Main lesson:
    ...
```

This forces the project to move beyond “the score is low” toward a clear scientific explanation of what failed and why.

---

## 23. Core Mindset

Do not stop at:

```text
The agent failed.
```

Ask:

```text
Where did it fail?
Was the failure reachable?
Was the mistake costly?
Was it caused by one action or many?
Was it caused by insufficient coverage?
Was it caused by wrong representation?
Was it caused by objective mismatch?
Can an intervention prove the diagnosis?
```

This is the workflow that turns RL coding practice into research practice.
