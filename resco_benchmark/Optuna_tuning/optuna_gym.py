import gym
import torch
from torch import nn
import pfrl
from pfrl import agents, replay_buffers
import optuna
import warnings

# Suppress gym warnings to keep Optuna logs clean
warnings.filterwarnings("ignore")

def make_env():
    return gym.make("Humanoid-v3")

# Initialize environment once to get dimensions dynamically
temp_env = make_env()
obs_size = temp_env.observation_space.low.size
action_size = temp_env.action_space.low.size
temp_env.close()

def objective(trial):
    env = make_env()
    
    # 1. Define the Hyperparameter Search Space
    # We sample learning rates on a log scale, and network/batch sizes categorically
    actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    
    # 2. Dynamic Network Architecture based on Optuna suggestions
    def squashed_diagonal_gaussian_head(x):
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        var = torch.exp(log_std * 2)
        base_dist = torch.distributions.Normal(mean, var.sqrt())
        return torch.distributions.transformed_distribution.TransformedDistribution(
            base_dist, [torch.distributions.transforms.TanhTransform()]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size * 2),
        pfrl.nn.Lambda(squashed_diagonal_gaussian_head),
    )

    def make_q_func():
        return nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    q_func1 = make_q_func()
    q_func2 = make_q_func()

    opt_p = torch.optim.Adam(policy.parameters(), lr=actor_lr)
    opt_q1 = torch.optim.Adam(q_func1.parameters(), lr=critic_lr)
    opt_q2 = torch.optim.Adam(q_func2.parameters(), lr=critic_lr)

    # Reduced replay buffer size for memory efficiency during tuning
    rbuf = replay_buffers.ReplayBuffer(10**5) 

    agent = agents.SAC(
        policy, q_func1, q_func2, opt_p, opt_q1, opt_q2,
        rbuf, gamma=0.99, replay_start_size=1000,
        batch_size=batch_size, tau=0.005, entropy_target=-action_size,
    )

    # 3. Custom Training Loop with Pruning Integration
    # We restrict the trial to 50 episodes so Optuna can iterate through configurations
    max_episodes = 50 
    max_episode_len = 1000
    scores = []
    
    for i in range(1, max_episodes + 1):
        # Compatible with Gym >= 0.26 API
        obs, _ = env.reset()
        R = 0 
        t = 0 
        
        while True:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            R += reward
            t += 1
            
            done = terminated or truncated
            reset = t == max_episode_len
            
            # The agent observes the transition and updates its networks
            agent.observe(obs, reward, done, reset)
            
            if done or reset:
                break
                
        scores.append(R)
        
        # Calculate moving average to smooth out the noisy RL reward signal
        current_metric = sum(scores[-5:]) / min(len(scores), 5)
        
        # 4. Report back to Optuna and check for Pruning
        trial.report(current_metric, i)
        if trial.should_prune():
            env.close()
            raise optuna.exceptions.TrialPruned()

    env.close()
    
    # Optuna will attempt to maximize this returned value
    return current_metric

if __name__ == "__main__":
    # 5. Execute the Study
    # MedianPruner aggressively stops unpromising trials after 10 warmup steps
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    print("Executing Optuna Study. Watch the metrics...")
    
    # Running 20 trials. For production workloads, you can increase this and add n_jobs=-1
    study.optimize(objective, n_trials=20)
    
    print("\n--- Optimization Finished ---")
    print(f"Best Trial Score: {study.best_value}")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")