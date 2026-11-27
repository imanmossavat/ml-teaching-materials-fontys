# MDP environment: the standard and all the code

This page holds **all the code** for the MDP design workshop: the standard, a template for a bigger
environment, and the tools to test and train on it. The *ideas* (what makes a good MDP, common
mistakes, example scenarios) are in [mdp-design-workshop.md](mdp-design-workshop.md).

**Status: optional.** You do not have to follow it. It is a convention that makes your environment easy
to test, to compare and to reuse in the later weeks of the course.

**You do not need Gym.** Do not install or import `gym` or `gymnasium`. The standard borrows only the
*style* of a [Gymnasium](https://gymnasium.farama.org/) environment: the same names for the same
things. Your environment is plain Python (plus `numpy` and `matplotlib` if you like). If you ever want
to plug it into Gymnasium later, the step from this style to the real thing is small.

Contents:
1. [The standard in one look](#1-the-standard-in-one-look)
2. [A small complete example](#2-a-small-complete-example)
3. [Template for a bigger environment](#3-template-for-a-bigger-environment)
4. [Test it: random baseline and a checker](#4-test-it-random-baseline-and-a-checker)
5. [Train it: Q-learning and one comparison plot](#5-train-it-q-learning-and-one-comparison-plot)
6. [Later: planning with the model](#6-later-planning-with-the-model-optional)
7. [Good habits](#7-good-habits)

The code blocks build on each other: run them in order in one file or notebook.

---

## 1. The standard in one look

```python
env = MyMDP()
state = env.reset()                              # start a new episode, returns the start state
state, reward, done, info = env.step(action)     # take one action
env.actions                                      # the list of possible actions
```

| Part | What it must do |
|---|---|
| `env.actions` | A list of all actions, for example `["left", "right"]` or `[0, 1, 2]`. |
| `env.reset()` | Starts a new episode and returns the start state. |
| `env.step(action)` | Applies one action. Returns four things: `next_state`, `reward`, `done`, `info`. |
| state | A plain, **hashable** value: an `int`, a `tuple` or a `namedtuple`, for example `(energy, days_left)`. Not a list and not a dict, because you will use it as a key in a Q-table. |
| `done` | `True` when the episode is over (a terminal state is reached, or a step limit is hit). Never `True` by accident. |
| `info` | A dict for extras that the agent does not use. By convention: `info["parts"]` = the reward components, `info["done_reason"]` = why the episode ended. May be empty. |

### How this differs from real Gymnasium

| | This standard | Gymnasium |
|---|---|---|
| `reset()` returns | `state` | `(observation, info)` |
| `step()` returns | `(next_state, reward, done, info)` | `(observation, reward, terminated, truncated, info)` |
| Spaces | a list `env.actions` | `action_space`, `observation_space` objects |
| Import needed | no | yes |

Gymnasium splits `done` into `terminated` (a real end of the task) and `truncated` (a time limit).
Here, `info["done_reason"] == "step_limit"` plays the role of `truncated`.

---

## 2. A small complete example

Two states, two actions. Work once, rest to recover. The model lives in one method, `transitions`,
and `step` just draws one outcome from it. That way the two cannot disagree, and the same method
can be used for planning later (section 6).

```python
import random

class TinyMDP:
    def __init__(self, seed=None, max_steps=20):
        self.actions = ["work", "rest"]
        self.rng = random.Random(seed)       # your own random generator, so runs can be repeated
        self.max_steps = max_steps
        self.state = None
        self.t = 0

    def transitions(self, state, action):
        """All outcomes of one state-action pair: a list of (probability, next_state, reward)."""
        if state == "tired" and action == "rest":
            return [(0.8, "fresh", 0.0), (0.2, "tired", 0.0)]
        if state == "fresh" and action == "work":
            return [(1.0, "tired", 1.0)]
        return [(1.0, state, 0.0)]           # work while tired, rest while fresh

    def reset(self):
        self.state = "tired"
        self.t = 0
        return self.state

    def step(self, action):
        assert action in self.actions, f"unknown action {action}"
        outcomes = self.transitions(self.state, action)
        _, s_next, reward = self.rng.choices(outcomes, weights=[o[0] for o in outcomes])[0]
        self.t += 1
        self.state = s_next
        done = self.t >= self.max_steps
        info = {"done_reason": "step_limit"} if done else {}
        return s_next, reward, done, info

# the standard loop works with any environment that follows the standard
env = TinyMDP(seed=1)
s, done, total = env.reset(), False, 0.0
while not done:
    a = random.choice(env.actions)
    s, r, done, info = env.step(a)
    total += r
print("return of one random episode:", total)
```

---

## 3. Template for a bigger environment

When you have 3 to 6 state variables, split the work into small functions, each doing one thing:

`transition → events → reward → termination`

- **`transition`**: the effect of the action. Deterministic and simple. This is the most important
  part of your design.
- **`events`**: optional randomness, applied after the action.
- **`reward`**: returns the total *and* its named parts.
- **`terminal`**: returns the reason the episode ends, or `None`.

This is a structure to **adapt, not to copy**. The scenario (a worker with energy and stress) is only
a placeholder.

```python
from collections import namedtuple

# the state: everything the agent needs to decide. A namedtuple is hashable, so it can be a Q-table key
State = namedtuple("State", ["energy", "progress", "stress"])

# named constants instead of magic numbers
MAX_ENERGY    = 5
GOAL_PROGRESS = 6
STRESS_PROB   = 0.1      # chance per step that stress goes up
PROGRESS_R    = 1.0      # reward per unit of progress
ENERGY_W      = 0.1      # weight of the energy change in the reward
GOAL_R        = 10.0
BURNOUT_R     = -5.0

class ExampleMDP:
    def __init__(self, seed=None, max_steps=25):
        self.actions = ["work", "rest"]
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.state, self.t = None, 0

    def reset(self):
        self.state, self.t = State(energy=MAX_ENERGY, progress=0, stress=2), 0
        return self.state

    def transition(self, s, action):
        if action == "work":
            cost = 3 if s.stress >= 4 else 2          # working is tiring, and more so when stressed
            return s._replace(progress=s.progress + 1, energy=s.energy - cost)
        if action == "rest":
            return s._replace(energy=min(s.energy + 1, MAX_ENERGY))
        raise ValueError(f"unknown action {action}")

    def events(self, s):
        if self.rng.random() < STRESS_PROB:
            s = s._replace(stress=min(s.stress + 1, 5))
        return s

    def reward(self, s_prev, action, s_next):
        parts = {
            "progress": PROGRESS_R * (s_next.progress - s_prev.progress),
            "energy":   ENERGY_W * (s_next.energy - s_prev.energy),
            "goal":     GOAL_R if s_next.progress >= GOAL_PROGRESS else 0.0,
            "burnout":  BURNOUT_R if s_next.energy <= 0 else 0.0,
        }
        return sum(parts.values()), parts

    def terminal(self, s):
        if s.progress >= GOAL_PROGRESS:
            return "goal"
        if s.energy <= 0:
            return "burnout"
        return None

    def step(self, action):
        s_next = self.events(self.transition(self.state, action))
        reward, parts = self.reward(self.state, action, s_next)
        self.t += 1
        self.state = s_next
        reason = self.terminal(s_next)
        if reason is None and self.t >= self.max_steps:
            reason = "step_limit"
        return s_next, reward, reason is not None, {"parts": parts, "done_reason": reason}
```

Log the reason for every reward when you debug. It shows what the agent is really paid for:

```python
env = ExampleMDP(seed=0)
s = env.reset()
s, r, done, info = env.step("work")
print(s, r, info["parts"])
```

---

## 4. Test it: random baseline and a checker

**Always test before you train.** A random policy shows whether the environment is stable and what
"bad" scores look like. A trained agent is only impressive compared with this baseline.

```python
import statistics

def random_policy(env, state):
    return random.choice(env.actions)

def run_episode(env, policy):
    """Plays one episode. Returns (total reward, length, reason the episode ended)."""
    s, done, total, n, reason = env.reset(), False, 0.0, 0, None
    while not done:
        s, r, done, info = env.step(policy(env, s))
        total += r
        n += 1
        reason = info.get("done_reason")
    return total, n, reason

def evaluate(make_env, policy, episodes=500):
    """Many episodes: mean and SD of the return, success rate, mean length."""
    results = [run_episode(make_env(seed), policy) for seed in range(episodes)]
    returns = [r[0] for r in results]
    return {
        "mean_return": statistics.mean(returns),
        "sd_return": statistics.stdev(returns),
        "success_rate": sum(r[2] == "goal" for r in results) / episodes,
        "mean_length": statistics.mean(r[1] for r in results),
    }

print(evaluate(lambda seed: ExampleMDP(seed=seed), random_policy))
```

`success_rate` assumes an end reason called `"goal"`. Rename it to fit your environment.
One episode is an anecdote: report the mean and the SD over many episodes.

A quick checker for the standard. If `check_env` passes, your environment follows the rules above:

```python
def check_env(make_env, episodes=20, max_steps=10_000):
    env = make_env(0)
    assert isinstance(env.actions, list) and env.actions, "env.actions must be a non-empty list"
    for _ in range(episodes):
        s, n, done = env.reset(), 0, False
        hash(s)                                        # raises TypeError if the state is not hashable
        while not done:
            out = env.step(random.choice(env.actions))
            assert len(out) == 4, "step must return (state, reward, done, info)"
            s, r, done, info = out
            hash(s)
            assert isinstance(r, (int, float)), "reward must be a number"
            assert isinstance(info, dict), "info must be a dict"
            n += 1
            assert n <= max_steps, "episode never ends: add a terminal state or a step limit"

    def episode_log(seed):                             # same seed + same actions = same episode
        e, pick, log, done = make_env(seed), random.Random(1), [], False
        e.reset()
        while not done:
            out = e.step(pick.choice(e.actions))
            log.append(out[:3])
            done = out[2]
        return log
    assert episode_log(7) == episode_log(7), "same seed, different episode: use self.rng, not the global random"

    env.reset()
    try:
        env.step("not_an_action")
    except Exception:
        pass
    else:
        raise AssertionError("an unknown action must be rejected")
    print("OK: the environment follows the standard")

check_env(lambda seed: ExampleMDP(seed=seed))
```

---

## 5. Train it: Q-learning and one comparison plot

An agent that learns a table `Q[state][action]` from experience. This is a good next step once your
environment works.

```python
def epsilon_greedy(Q, s, actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(actions)
    return max(actions, key=lambda a: Q[s][a])

def q_learning(make_env, episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.1, seed=0):
    rng = random.Random(seed)
    env = make_env(seed)
    Q = {}
    def row(s):                                        # make a row the first time we see a state
        return Q.setdefault(s, {a: 0.0 for a in env.actions})
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            row(s)
            a = epsilon_greedy(Q, s, env.actions, epsilon, rng)
            s2, r, done, info = env.step(a)
            row(s2)
            terminated = done and info.get("done_reason") != "step_limit"
            target = r if terminated else r + gamma * max(Q[s2].values())
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
    return Q

Q = q_learning(lambda seed: ExampleMDP(seed=seed))
greedy = lambda env, s: max(env.actions, key=lambda a: Q.get(s, {}).get(a, 0.0))
print("random :", evaluate(lambda seed: ExampleMDP(seed=1000 + seed), random_policy))
print("trained:", evaluate(lambda seed: ExampleMDP(seed=1000 + seed), greedy))
```

Two details that are easy to get wrong:
- At a **terminal** state there is no future, so the target is just `r`. At a **step limit** the
  story did not really end, so the target still looks ahead. That is why the code tests the reason.
- `Q[s][a] = average(observed reward)` is **not** Q-learning. It is only reward statistics.
  Q-learning bootstraps: it uses `max Q[s2]`, the value of the next state.

One comparison plot (random versus trained) usually makes a report much stronger:

```python
import matplotlib.pyplot as plt

def returns_of(policy, n=300):
    return [run_episode(ExampleMDP(seed=2000 + i), policy)[0] for i in range(n)]

plt.hist(returns_of(random_policy), bins=20, alpha=0.6, label="random")
plt.hist(returns_of(greedy), bins=20, alpha=0.6, label="trained (Q-learning)")
plt.xlabel("episode return"); plt.ylabel("episodes"); plt.legend()
plt.show()
```

Report at least: episode return, success rate and episode length, for both policies.

---

## 6. Later: planning with the model (optional)

Planning methods such as value iteration need the model itself, not only samples from `step`.
That is why section 2 has a `transitions(state, action)` method that returns
`(probability, next_state, reward)` for every outcome, with probabilities adding up to 1. It works
only when you can list all the states, so a small discrete environment is best.

```python
def value_iteration(env, states, gamma=0.9, theta=1e-8):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            best = max(
                sum(p * (r + gamma * V[s2]) for p, s2, r in env.transitions(s, a))
                for a in env.actions
            )
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            return V

print(value_iteration(TinyMDP(), ["tired", "fresh"]))
```

Compare this with what Q-learning finds for the same environment. They should agree.

---

## 7. Good habits

- **One place for the model.** Keep the transition rule and the reward rule in clearly separated
  parts of your code, so that you can explain each one separately.
- **Repeatable runs.** Give the environment its own random generator with a `seed`, and do not use
  the global one. Then the same seed gives the same episode.
- **Always an end.** Either a terminal state or a step limit, so an episode cannot run forever.
- **Everything in the state.** The state must hold what the agent needs to decide. If the future
  depends on something that is not in the state, the model is not Markov. (A step limit makes the
  time step matter too. For a short, clear task that is fine; if it bothers you, add `t` to the state.)
- **A reward you can explain.** Prefer a few named parts (`info["parts"]`) over one unexplained number.
- **Name the end.** Store *why* an episode ended (`info["done_reason"]`), not only that it did.
- **Check the actions.** Reject an unknown action instead of silently ignoring it.
- **Named constants** (`GOAL_R = 10`) instead of magic numbers inside functions.
