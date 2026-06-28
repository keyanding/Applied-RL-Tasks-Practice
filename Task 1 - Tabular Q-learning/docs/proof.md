# Comprehensive Guide to Q-Learning: Algorithm and Convergence

## 1. The Q-Learning Algorithm

Q-learning is a model-free reinforcement learning algorithm used to find the optimal action-value function.

### A. Initialization

Before the agent interacts with the environment, estimates must be initialized:

* Let $S$ be the set of all states and $A$ be the set of all actions.
* Initialize the Q-table, $Q(s,a)$, arbitrarily for all $s \in S$ and $a \in A$ (e.g., setting all to $0$).
* If the environment has terminal states, initialize $Q(\text{terminal}, \cdot) = 0$.
* Set hyperparameters:
* **Learning rate ($`\alpha \in (0, 1]`$):** Determines how much new information overrides old information.
* **Discount factor ($`\gamma \in [0, 1)`$):** Weights the importance of future rewards.


* **Exploration rate ($`\epsilon \in (0, 1)`$):** The probability of taking a random action to explore.



### B. The Main Loop (Interaction and Update)

For each episode (or continuous trajectory), repeat the following steps:

1. **Observe State:** Observe the current state $s$.
2. **Choose an Action:** Select an action $a$ using an exploration strategy (e.g., $\epsilon$-greedy):
* With probability $1 - \epsilon$: **Exploit** ( $a = \arg\max_{a'} Q(s, a')$ ).
* With probability $\epsilon$: **Explore** (choose a random action $a \in A$).


3. **Execute Action:** Take action $a$ in the environment, and observe the immediate reward $r$ and the next state $s'$.
4. **Mathematical Update (Temporal Difference):** Update the Q-value for $(s,a)$ using the stochastic approximation of the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$


5. **Transition:** Update the current state: $s \leftarrow s'$.

### C. Termination Condition

The algorithm repeats until:

1. **Convergence:** The maximum update step across the table is smaller than a tiny threshold $\delta$: $\max_{s,a} |\Delta Q(s,a)| < \delta$.
2. **Fixed Budget:** Reaches a predefined maximum number of episodes or steps.
3. **Policy Stability:** The greedy policy $\pi(s) = \arg\max_a Q(s,a)$ stops changing.

---

## 2. Rigorous Proof of Q-Learning Convergence

For Q-learning to converge to a unique optimal policy, we rely on the Contraction Mapping Theorem and Stochastic Approximation theory.

### Part 1: Environment Conditions

The environment must satisfy the following constraints:

1. **Finite Spaces:** State space $S$ and action space $A$ are discrete and finite.


2. **Discount Factor:** Governed by $\gamma \in [0,1)$.


3. **Bounded Rewards:** Rewards are bounded, $r_t \in [0, R_{max}]$.



Under these conditions, "there always exists a stationary and deterministic policy that is optimal for all starting states simultaneously"  (Proof: Puterman'94, Thm 6.2.7).

### Part 2: The Bellman Optimality Operator

The optimal action-value function is defined by the Bellman Optimality Equation:


$$Q^{* }(s,a) = R(s,a) + \gamma \mathbb{E}_{{s'} \sim P(s,a)} [\max_{{a'} \in A} Q^{* }({s'}, {a'})]$$


We define the **Bellman Optimality Operator**, $\mathcal{T}$:


$$(\mathcal{T}Q)(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s'|s,a) \max_{a' \in A} Q(s',a')$$


The true optimal $`Q^*`$ is a **fixed point** of this operator, meaning $`\mathcal{T}Q^* = Q^*`$.

### Part 3: Contraction Mapping

To prove convergence to a unique fixed point, we must prove $\mathcal{T}$ is a $\gamma$-contraction mapping under the infinity norm, $`\Vert Q \Vert_\infty = \max_{s,a} |Q(s,a)|`$. For any arbitrary $Q_1$ and $Q_2$:


$$\Vert \mathcal{T}Q_1 - \mathcal{T}Q_2 \Vert_\infty = \max_{s,a} \left| \gamma \sum_{s'} P(s'|s,a) \left( \max_{a'} Q_1(s',a') - \max_{a'} Q_2(s',a') \right) \right|$$

Because $\max_{a'} Q_1 - \max_{a'} Q_2 \le \max_{a'} |Q_1 - Q_2|$, and the transition probabilities sum to 1:


$$\Vert \mathcal{T}Q_1 - \mathcal{T}Q_2 \Vert_\infty \le \gamma \Vert Q_1 - Q_2 \Vert_\infty$$

Because $\gamma \in [0,1)$, $\mathcal{T}$ is strictly a contraction mapping. By **Banach's Fixed-Point Theorem**, there is exactly one unique fixed point $`Q^*`$, and applying $`\mathcal{T}`$ infinitely many times converges to $`Q^*`$.

### Part 4: Stochastic Approximation Requirements

Because Q-learning samples the environment rather than doing exact dynamic programming, the stochastic update rule converges to the deterministic fixed point $Q^*$ if:

1. **Infinite Exploration:** Every state-action pair is visited infinitely often as $t \to \infty$. (We must "Assume states & actions are visited uniformly", which requires exploration ).


2. **Robbins-Monro Learning Rates:** The learning rate $\alpha_t$ must decay such that $\sum_{t=0}^{\infty} \alpha_t = \infty$ and $\sum_{t=0}^{\infty} \alpha_t^2 < \infty$.

If these hold, the Q-values mathematically converge to $`Q^*`$, yielding the optimal policy $`\pi^*(s) = \arg\max_{a\in A} Q^*(s,a)`$.

---

## 3. Equivalence of Update Formulas

In reinforcement learning literature, the Q-learning update is written in two seemingly different ways. This is an algebraic proof showing that the Temporal Difference (TD) form and the Stochastic Approximation (Moving Average) form are mathematically identical.

**1. The TD Form (Error Correction):**


$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha_t \left[ r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t) \right]$$

**2. Algebraic Expansion:**
Distribute the learning rate ($\alpha_t$) into the brackets:


$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha_t \left( r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') \right) - \alpha_t Q_t(s_t, a_t)$$

**3. Factoring:**
Group the two terms containing $Q_t(s_t, a_t)$ together:


$$Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) (1 - \alpha_t) + \alpha_t \left( r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') \right)$$

**4. The Stochastic Approximation Form:**
Rearranging gives the exact moving average representation:


$$Q_{t+1}(s_t, a_t) = (1 - \alpha_t)Q_t(s_t, a_t) + \alpha_t \left( r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') \right)$$