# A coding standard for your MDP environment (optional)

**Status: optional.** You do not have to follow it. It is a convention that makes your environment easy
to test, to compare and to reuse in the later weeks of the course.

**You do not need Gym.** Do not install or import `gym` or `gymnasium`. The standard borrows only the
*style* of a [Gymnasium](https://gymnasium.farama.org/) environment: the same names for the same
things. Your environment is plain Python (plus `numpy` if you like). If you ever want to plug it into
Gymnasium later, the step from this style to the real thing is small.

## The standard in one look

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
| state | A plain, **hashable** value: an `int` or a `tuple`, for example `(energy, days_left)`. Not a list and not a dict, because you will use it as a key in a Q-table. |
| `done` | `True` when the episode is over (a terminal state is reached, or a step limit is hit). Never `True` by accident. |
| `info` | A dict for extras that the agent does not use, for example the parts of the reward. May be empty. |

## How this differs from real Gymnasium

| | This standard | Gymnasium |
|---|---|---|
| `reset()` returns | `state` | `(observation, info)` |
| `step()` returns | `(next_state, reward, done, info)` | `(observation, reward, terminated, truncated, info)` |
| Spaces | a list `env.actions` | `action_space`, `observation_space` objects |
| Import needed | no | yes |

Gymnasium splits `done` into `terminated` (a real end of the task) and `truncated` (a time limit).
You can do the same in `info` if you want, for example `info["truncated"] = True`.

## A small complete example

Two states, two actions. Work once, rest to recover.

```python
import random

class TinyMDP:
    def __init__(self, seed=None, max_steps=20):
        self.actions = ["work", "rest"]
        self.rng = random.Random(seed)       # your own random generator, so runs can be repeated
        self.max_steps = max_steps
        self.state = None
        self.t = 0

    def reset(self):
        self.state = "tired"
        self.t = 0
        return self.state

    def step(self, action):
        assert action in self.actions, f"unknown action {action}"
        s = self.state
        if s == "tired" and action == "rest":
            s_next = "fresh" if self.rng.random() < 0.8 else "tired"
            reward = 0.0
        elif s == "fresh" and action == "work":
            s_next = "tired"
            reward = 1.0
        else:                                # work while tired, rest while fresh
            s_next = s
            reward = 0.0
        self.t += 1
        self.state = s_next
        done = self.t >= self.max_steps
        return s_next, reward, done, {}

# the standard loop works with any environment that follows the standard
env = TinyMDP(seed=1)
s, done, total = env.reset(), False, 0.0
while not done:
    a = random.choice(env.actions)
    s, r, done, info = env.step(a)
    total += r
print("return of one random episode:", total)
```

## Good habits

- **One place for the model.** Keep the transition rule and the reward rule in clearly separated
  parts of your code, so that you can explain each one separately.
- **Repeatable runs.** Give the environment its own random generator with a `seed`, and do not use
  the global one. Then the same seed gives the same episode.
- **Always an end.** Either a terminal state or a step limit, so an episode cannot run forever.
- **Everything in the state.** The state must hold what the agent needs to decide. If the future
  depends on something that is not in the state, the model is not Markov.
- **A reward you can explain.** Prefer a few named parts (`info["parts"]`) over one unexplained number.
- **Check the actions.** Reject an unknown action instead of silently ignoring it.

## Later in the course (optional, not needed now)

Planning methods such as value iteration need the model itself, not only samples from `step`. If you
want to prepare, add a method that lists the possible outcomes of one state-action pair:

```python
def transitions(self, state, action):
    """Return a list of (probability, next_state, reward). The probabilities add up to 1."""
    ...
```

Then `step` can simply draw one outcome from this list, and the two cannot disagree.
