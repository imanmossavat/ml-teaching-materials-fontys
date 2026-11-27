# Designing your own MDP environment

Workshop reader · Iman Mossavat · 2026

All the code is on one page: [mdp-environment-standard.md](mdp-environment-standard.md). This reader holds
the ideas. When it says "see code", look there.

## Goal of this workshop

In the previous assignment you worked with a fixed environment and learned how an agent can learn a
policy using Q-learning. In this workshop the focus shifts: **you design the environment itself.**

Your goal is to define a simple, clear Markov Decision Process (MDP) that an RL agent could use.

### What you need to specify

| Part | Question it answers |
|---|---|
| **State** | What can the agent observe? |
| **Actions** | What can the agent do? |
| **Transition** | How does the state change after an action? |
| **Reward** | What is the agent optimising? |

The emphasis is not on complex code, but on a **clear and interpretable model**.

## Design principles

1. **Keep it small.** At most 5 to 6 state variables and 4 to 5 actions.
2. **Fully observable state.** Everything that matters is in the state. No hidden variable affects the outcome.
3. **Clear transitions.** Each action has a simple, understandable effect. Avoid complicated interactions.
   The transition function is the most important part of the design.
4. **Explicit reward.** Split the reward into components, and make each one interpretable.
5. **Include a trade-off.** At least one meaningful one, for example short-term versus long-term
   reward, progress versus energy, or risk versus reward.

## The shape of the code

Code is in [section 2 and 3 of the standard](mdp-environment-standard.md#3-template-for-a-bigger-environment).
In short:

```
env.reset()        -> start state
env.step(action)   -> next_state, reward, done, info
```

and inside `step`: **transition → events (randomness) → reward (components) → termination (with a reason)**.
The state holds everything the agent needs to decide. The template is a structure to adapt, not to copy.

## Common design patterns

1. **Separation of concerns.** One small function per job: state → transition → events → reward → done.
2. **Reward decomposition.** Instead of one number, give named parts (`goal_reward`, `cost`,
   `risk_penalty`) and add them up. This helps debugging and prevents black-box rewards.
3. **Explicit termination reasons.** Do not only return `done = True`; also store why
   (`"burnout"`, `"fired"`, `"goal"`).
4. **Baseline first.** Run a random policy, check that the environment is stable, and only then train.
5. **Named constants.** `GOAL_REWARD = 10` instead of a bare `10` buried in a function.

## Common mistakes to avoid

| Mistake | What goes wrong |
|---|---|
| **Reward mismatch** | The reward does not match the goal, for example paying for risky actions without a penalty. |
| **Hidden or missing state variables** | If the agent cannot observe something that affects the outcome, the MDP is incomplete (not Markov). |
| **Overly complex transitions** | If one action changes many variables at once, you can no longer explain it, and learning becomes unstable. |
| **Reward hacking** | The agent exploits the reward definition instead of solving the intended task. |
| **No baseline** | If you only train an agent and never compare with random behaviour, the results are hard to interpret. |
| **Calling averaged reward a Q-table** | `Q[s][a] = average(observed_reward)` is only reward statistics. Q-learning also looks ahead (`max Q[s']`). |

### Flaws seen in earlier student designs

These are real kinds of mistakes from earlier student environments, written generally. Use them as a
checklist for your own design and for peer review.

| Flaw | Example | Question to ask |
|---|---|---|
| **Reward uses the wrong variable** | A city game meant to reward growth in money, but the code rewards the change in pollution. | Does each reward part measure what I *intended*? Print `info["parts"]` for a few steps. |
| **Hidden state** | A cleaning robot cannot see how dirty each room is, but dirt decides the reward. | Can the agent observe everything that affects the outcome? |
| **Missing state variable** | A grid game where enemies move, but their positions are not in the state. | If I know the state, is the future fully determined (apart from the dice)? |
| **Reward hacking** | Bonuses for shooting or for spending ammo, so the agent just shoots. | What is the cheapest way to collect reward, and is that what I want? |
| **Values pile up at the bounds** | Every variable is clipped to 0 to 10, so the agent sits at 10 or 0. | What happens at the edges of my ranges? |
| **Quantity that grows without bound** | Dirt that only increases: there is nothing steady to learn about. | Does the system settle down, or does it just drift? |
| **A goal that is almost never reached** | Four conditions must all hold at once. | How often does a random policy reach the goal? If never, the agent has nothing to learn from. |
| **One run treated as the truth** | A "policy" read from a single 30-step run, or conclusions from one seed. | Did I run many episodes and report mean ± SD? |
| **A horizon that cuts the story** | Episodes of 30 steps stop before a startup runs out of cash. | Does the step limit hide the thing I want to study? |

**Exercise: spot the issue.** Take a classmate's design, ignoring the code. Write only its state,
actions, transition and reward. Then find: (1) a non-Markov state, (2) a reward that does not match
its intent, (3) something that never settles, (4) a conclusion based on a single run.

## How to nudge the design forward

Start simple, then add one new idea at a time:

1. a deterministic environment
2. add terminal conditions
3. add reward decomposition
4. add randomness
5. test a random policy
6. train a simple Q-learning agent
7. compare random versus trained

Two helpers (code in the standard):
- **Log the reason for every reward** (`print(info["parts"])`). It shows what the agent is optimising.
- **Keep one comparison plot** of random versus trained: episode return, success rate, episode length.
  This usually makes the report much stronger.

The code for the random baseline, the checker and Q-learning is in
[sections 4 and 5 of the standard](mdp-environment-standard.md#4-test-it-random-baseline-and-a-checker).
Q-learning is a good next step once your environment works.

## Example MDP ideas

These are reusable **design templates**, not full environments. Apply the same pattern to any
scenario (games, robotics, business, simulations).

| Template | Pattern | State | Actions | Good next steps |
|---|---|---|---|---|
| **Resource management** | competing limited resources over time | resource_1, resource_2, capacity, progress, risk | invest, consume, replenish, explore | resource decay; diminishing returns; failure when a resource hits zero; greedy versus long-term planning |
| **Queue / workload** | incoming tasks versus processing capacity | queue_size, processing_power, energy, latency, quality | process_fast, process_safe, delay, accept_task | random task arrivals; deadline penalties; quality versus speed; stability under a random policy |
| **Maintenance / stability** | the system degrades unless maintained | health, wear, efficiency, cost, stress | operate, repair, upgrade, shutdown | slow degradation; a failure threshold; preventive versus reactive maintenance; short-term gain versus survival |
| **Exploration / uncertainty** | known reward versus unknown regions | knowledge, uncertainty, reward_estimate, risk, time | exploit, explore, gather_info, wait | random rewards; information gain in the reward; exploration versus exploitation; fixed policy versus learner |

### Starter scenarios

Small enough to check by hand (3 to 12 states, 2 to 3 actions):

| Scenario | The trade-off | Watch for |
|---|---|---|
| **Machine maintenance**: produce now or maintain? | output today versus breakdowns later | when does "always produce" stop paying off? |
| **Exam countdown**: study or relax, with days left and your preparation level | free time now versus pass chance later | is "start tomorrow" ever the best plan? |
| **App under load**: accept all users or throttle? | revenue versus crashing | the tipping point where more users hurts |
| **Cleaning robot**: clean, charge or wait while dirt appears at random | clean room versus empty battery | does the reward pay for a clean room, or for busy cleaning? |

### What to aim for

Strong designs usually have:
- one clear **bottleneck variable** (energy, capacity, risk, …)
- one clear **trade-off axis**
- at least one **random element**
- a **random baseline**
- optional: a simple learning agent (Q-learning)

> A good MDP is not defined by its domain, but by whether the agent must repeatedly choose between
> short-term reward and long-term stability under constraints.

## What good usually looks like

- clear state variables
- clear action effects
- separate transition, event, reward and termination logic
- reward components that are easy to explain
- a random baseline
- a trained policy, if possible
- a small evaluation section with plots or summary statistics

## Your task

1. **Define your own scenario.** For example a student balancing study and social life, a robot
   navigating a grid, or an agent managing resources.
2. **Build a first version** of your environment.
3. **Modify and improve** your model: adjust state variables, refine transitions, improve the reward design.
4. **Test it** with a simple random policy.
5. **Train it** with Q-learning (or Deep Q-learning, advanced).

## Sanity checks

Before you finish, make sure that:
- you can explain your model clearly in **two sentences**
- the reward function makes sense
- there is a clear goal
- there is at least one trade-off
- the environment behaves reasonably under random actions
- `check_env` from the standard passes (optional)
