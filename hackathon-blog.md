Building a Multi-Agent Fraud Detection RL Environment

Hackathon Blog:

During this hackathon, we worked on building an OpenEnv-compatible multi-agent
reinforcement learning environment for fraud detection. The project simulates a
financial fraud ecosystem where two intelligent agents interact with each other:
a defender and a fraudster.

The defender represents a fraud detection system. Its goal is to identify risky
accounts, suspicious merchants, and abnormal transaction behavior while avoiding
false positives. The fraudster represents an adaptive adversary. Its goal is to
move money, hide suspicious patterns, rotate mule accounts, exploit refund flows,
and eventually cash out without being detected.

Instead of treating fraud detection as a static classification problem, we
framed it as an interactive decision-making problem. Real fraud is not a single
row in a dataset. It evolves over time. Fraudsters change strategies when they
notice pressure, and defenders must respond with limited information. That is
why we chose a multi-agent reinforcement learning environment as our problem
statement.


Why We Chose This Problem

Fraud detection is a high-impact real-world problem. Banks, payment companies,
marketplaces, and fintech platforms constantly fight against adversaries who
adapt to detection rules. Traditional fraud systems often rely on static models,
thresholds, or rules. These approaches are useful, but they can struggle when
fraudsters deliberately change their behavior.

We wanted to build something closer to the real world:

- A defender that does not know the full hidden state.
- A fraudster that actively tries to avoid detection.
- Multiple fraud patterns instead of one fixed task.
- A reward system that balances fraud prevention against user friction.
- A training environment where agents can improve through repeated interaction.

This made the problem both technically interesting and practically meaningful.
It also matched the spirit of the hackathon: building an environment where
agents can learn by acting, failing, adapting, and improving.


What We Built

We built a fraud detection reinforcement learning environment with two agents:

1. Defender Agent

The defender sees a partial and noisy view of the financial system. It can take
actions such as monitoring an account, challenging a user, freezing an account,
holding a transaction, blocking a merchant, investigating a connected group, or
doing nothing.

The defender is rewarded for stopping fraud, identifying suspicious behavior
early, holding fraudulent transactions, and blocking colluding merchants. It is
penalized for false positives, customer friction, unnecessary investigations,
and missed fraud.

2. Fraudster Agent

The fraudster sees an operational view of the fraud route. It can split
payments, rotate mule accounts, switch merchants, rotate devices, delay activity,
abuse refund workflows, attempt cashout, or do nothing.

The fraudster is rewarded for successful cashout, staying undetected, and using
evasive behavior when detection pressure is high. It is penalized when accounts
are frozen, merchants are blocked, routes are disrupted, or cashout fails.

The environment supports multiple fraud families:

- Refund abuse
- Mule cashout
- Merchant collusion
- Account takeover
- Random task selection across fraud families

Each episode starts from a generated hidden world state. The defender and
fraudster receive different partial observations from that world. This was
important because real-world fraud detection systems rarely have perfect
information. They work with incomplete signals, noisy risk scores, delayed
alerts, and indirect patterns.


How We Made It

We designed the project as a modular environment rather than one large script.
The main components include:

- A scenario generator that creates different fraud worlds.
- A hidden world state that stores the true underlying fraud situation.
- An observation generator that gives partial views to each agent.
- An action processor that validates and applies defender and fraudster actions.
- A transition engine that advances the world after each step.
- A reward engine that gives step-level rewards.
- A termination engine that decides when an episode ends.
- A grading engine that calculates final episode metrics.
- A FastAPI server that exposes the environment through an OpenEnv-style API.
- PPO and GRPO training scripts for learning policies.
- An inference script for running LLM-based agents.

The environment was built around episodes. At the start of an episode, the
system resets with a selected fraud family and seed. Then, step by step, the
agents act in the environment. The defender tries to reduce fraud losses while
minimizing false positives. The fraudster tries to keep routes alive and cash
out before being stopped.

We also added grading metrics such as:

- Total fraud prevented
- Total fraud escaped
- False positive count
- False positive rate
- Detection delay
- Customer friction score
- Merchant disruption score
- Defender score
- Fraudster score

These metrics help us evaluate not only whether the defender catches fraud, but
also whether it does so responsibly.



Training Approach

We explored two training directions.

First, we included PPO-style training for policy networks. This gives a more
traditional reinforcement learning setup where agents learn from the rewards
generated by the environment.

Second, we worked on GRPO training for LLM agents. The idea was to let an LLM
act as either the defender or fraudster. The LLM receives an observation,
generates a structured action, the environment executes that action, and the
episode reward is used for learning.

For GRPO, the reward functions included:

- Whether the model produced valid structured JSON.
- Whether the selected action was legal in the current state.
- The final episode reward.
- For the fraudster, an additional evasion reward based on final alert level.

This helped shape the model toward outputs that are both syntactically valid and
strategically useful.

logs for trained model:
![alt text](image.png)
![alt text](<WhatsApp Image 2026-04-26 at 3.06.51 PM.jpeg>)

What We Learned

This hackathon taught us several important lessons.

First, fraud detection is naturally adversarial. A model that only learns from
static examples may not be enough when the opponent adapts. Multi-agent
simulation gives us a way to study that adaptation.

Second, partial observability matters. The defender should not see ground-truth
fraud labels directly, and the fraudster should not see the defender's complete
internal state. Designing the observation contract carefully made the environment
more realistic.

Third, reward design is difficult. If the defender is rewarded only for freezing
accounts, it may freeze too aggressively. If it is punished too much for false
positives, it may become passive. Balancing fraud prevention, user friction, and
merchant disruption was one of the most important design decisions.

Finally, GPU memory is a real engineering constraint. A training configuration
that looks correct on paper may fail at the optimizer step because optimizer
states, gradients, and activations take much more memory than expected.


Final Outcome

By the end of the hackathon, we had built a working multi-agent fraud detection
environment with:

- Multiple fraud scenarios.
- Defender and fraudster action spaces.
- Partial observations for both agents.
- Step-level rewards and final grading metrics.
- Server and client support.
- PPO-style training.
- LLM inference support.
- GRPO training support with custom rollout logic.

The project demonstrates how fraud detection can be modeled as an adaptive
agent-vs-agent problem rather than just a classification task. It also gives a
foundation for future work, such as stronger LLM policies, richer fraud
scenarios, better evaluation dashboards, and more scalable training.


Conclusion

This project was a mix of environment design, reinforcement learning, LLM
integration, API debugging, and GPU optimization. We chose fraud detection
because it is a meaningful real-world problem where adversarial behavior matters.
Through the hackathon, we learned how to turn that idea into a simulation where
agents can interact, compete, and improve.

The biggest takeaway is that building intelligent agents is not only about the
model. It is also about the environment, the rewards, the observations, the
constraints, and the engineering details that make learning possible.
