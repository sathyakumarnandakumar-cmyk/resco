import gym
import torch
import pfrl
import optuna
from pfrl import agents, explorers, replay_buffers, experiments, q_functions

def objective(trial):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    
    # 2. Setup Environment
    env = gym.make("CartPole-v1")
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 3. Setup PFRL components (DQN)
    q_func = torch.nn.Sequential(
        torch.nn.Linear(obs_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_actions),
        q_functions.DiscreteActionValueHead(),
    )
    
    optimizer = torch.optim.Adam(q_func.parameters(), lr=lr)
    rb = replay_buffers.ReplayBuffer(capacity=10**4)
    explorer = explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)

    agent = agents.DQN(
        q_func, optimizer, rb, gamma=gamma, explorer=explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100
    )

    # 4. Define Pruning Callback (Step Hook)
    # This reports the average reward to Optuna every 'step'
    def step_hook(env, agent, step):
        if step % 1000 == 0:
            # We use the average rewards or evaluations for pruning
            # For simplicity, we'll report the current cumulative stats
            current_score = agent.get_statistics()[0][1] # Example: getting 'average_q' or custom metric
            trial.report(current_score, step)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # 5. Run Training
    experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=5000,
        eval_n_steps=None,
        eval_n_episodes=10,
        eval_interval=1000,
        outdir="results",
        train_max_episode_len=200,
        step_hooks=[step_hook] # Attaching the pruner here
    )

    # Return the final evaluation metric
    # Note: If no evaluation happened, stats might be empty or old.
    stats = agent.get_statistics()
    return stats[0][1] if stats else 0.0

if __name__ == "__main__":
    # Create a study with a SQLite backend for the dashboard
    study = optuna.create_study(
        study_name="pfrl_dqn_optimization",
        storage="sqlite:///pfrl_optuna.db",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # Standard median pruner
    )
    
    study.optimize(objective, n_trials=20)
    
    print("Best Trial:")
    print(study.best_params)