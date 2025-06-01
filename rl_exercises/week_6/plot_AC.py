import os

import gymnasium as gym
import matplotlib.pyplot as plt
from actor_critic import ActorCriticAgent, set_seed

env_names = ["CartPole-v1", "LunarLander-v3"]
baselines = ["none", "avg", "value", "gae"]
total_steps = 100000
eval_interval = 5000
eval_episodes = 5
results = {}

for env_name in env_names:
    for baseline in baselines:
        print(f"\nTraining {env_name} with baseline: {baseline} ")
        env = gym.make(env_name)
        set_seed(env, seed=0)

        agent = ActorCriticAgent(
            env=env,
            baseline_type=baseline,
            gamma=0.99,
            gae_lambda=0.95,
            baseline_decay=0.9,
            seed=0,
            lr_actor=5e-4,
            lr_critic=1e-3,
            hidden_size=128,
        )

        agent.train(
            total_steps=total_steps,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )

        results[(env_name, baseline)] = agent.eval_returns


output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for env_name in env_names:
    plt.figure(figsize=(10, 6))
    for baseline in baselines:
        steps, returns = zip(*results[(env_name, baseline)])
        plt.plot(steps, returns, label=f"{baseline}")
    plt.title(f"{env_name} - Average Return vs. Steps")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.legend(title="Baseline")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{env_name}_baseline_comparison.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()
