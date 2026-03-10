"""
main-o.py — Optuna-powered hyperparameter tuning for RESCO traffic simulation.

Wraps the existing training loop with Optuna for automated hyperparameter search
and pruning. Logs every trial to MLflow. Currently supports IDQN and IMA2C.

Usage:
    python main-o.py --agent IDQN --n_trials 20 --n_jobs 4
    python main-o.py --agent IDQN --n_trials 50 --n_jobs 4 --pruner hyperband
"""

import argparse
import copy
import multiprocessing as mp
import os
import sys

# Ensure parent directory is in sys.path for imports from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import shutil
from datetime import datetime

import mlflow
import numpy as np
import optuna
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rewards as reward_module
from config.agent_config import agent_configs
from config.map_config import map_configs
from config.mdp_config import mdp_configs
from multi_signal import MultiSignal
from utils.mlflow_logger import log_metrics, log_model_artifact
from utils.time_utils import convert_time_range_to_seconds

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

START_TIME = datetime.now().strftime("%d_%m_%H_%M")

# Map reward name strings to actual functions
REWARD_MAP = {
    "wait": reward_module.wait,
    "wait_norm": reward_module.wait_norm,
    "pressure": reward_module.pressure,
    "queue_maxwait": reward_module.queue_maxwait,
    "queue_maxwait_neighborhood": reward_module.queue_maxwait_neighborhood,
}


# ---------------------------------------------------------------------------
# Optuna Callback: save best model whenever a trial beats the previous best
# ---------------------------------------------------------------------------
class SaveBestModelCallback:
    """Optuna callback that archives the best model zip after each completed trial."""

    def __init__(self, optuna_models_dir: str):
        self.best_value = float("inf")  # minimize delay → lower is better
        self.optuna_models_dir = optuna_models_dir

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.value is None:
            return  # pruned or failed
        if trial.value < self.best_value:
            self.best_value = trial.value
            # The best model zip is saved inside the trial's results dir
            best_zip = trial.user_attrs.get("best_model_zip")
            if best_zip and os.path.exists(best_zip + ".zip"):
                dest = os.path.join(self.optuna_models_dir, f"best_trial_{trial.number}.zip")
                shutil.copy2(best_zip + ".zip", dest)
                print(
                    f"\n★ NEW BEST — Trial {trial.number} "
                    f"(delay={trial.value:.4f}) → saved to {dest}\n"
                )


# ---------------------------------------------------------------------------
# MLflow logging helper — initializes a per-trial MLflow run
# ---------------------------------------------------------------------------
def init_mlflow_optuna_run(args, env, trial, suggested_reward, PARAMS_ALGORITHM):
    """
    Initialize an MLflow run for a single Optuna trial.

    Similar to init_mlflow_run in utils/mlflow_logger.py but adds
    Optuna-specific parameters and tags.
    """
    main_dir = os.path.dirname(os.path.abspath(__file__))

    experiment_name = f"{args.agent}-sumo-{args.map}-optuna"
    if args.experiment_suffix:
        experiment_name += f"_{args.experiment_suffix}"

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(main_dir, 'mlflow.db')}")
    mlflow.set_experiment(experiment_name)

    tags = {
        "environment": "sumo-v0",
        "agent": args.agent,
        "net": args.net,
        "activation": args.activation,
        "framework": "stable-baselines3",
        "validation_period": args.validation_period,
        "optuna": "true",
        "optuna_study": args.study_name,
        "optuna_trial": str(trial.number),
    }
    if args.group_tag:
        tags["group_tag"] = args.group_tag

    run_description = (
        args.description
        or f"Optuna trial {trial.number}: {args.agent} on {args.map}"
    )
    run_name = (
        f"{args.agent}-optuna-trial{trial.number}-{args.map}"
        f"_net-{args.net}"
        f"_act-{args.activation}"
        + (f"_ns-{args.negative_slope}" if args.activation == "leaky_relu" else "")
        + f"_rw-{suggested_reward}"
        f"_seed-{args.seed}"
        f"_{START_TIME}"
    )

    run = mlflow.start_run(
        run_name=run_name,
        description=run_description,
        tags=tags,
    )
    mlflow.log_params(PARAMS_ALGORITHM)
    return run


# ---------------------------------------------------------------------------
# Model selection helpers (reused from main.py)
# ---------------------------------------------------------------------------
def choose_best_validation_model(dict_with_agents: dict):
    max_count_of_vehicles = max(
        [
            model_info["count_of_vehicles_completing_journey"]
            for model_info in dict_with_agents.values()
        ]
    )
    list_of_eps_numbers_max_count_of_vehicles = [
        eps_number
        for eps_number, model_info in dict_with_agents.items()
        if model_info["count_of_vehicles_completing_journey"] == max_count_of_vehicles
    ]
    helper_list_total_average_delays = [
        model_info["total_average_delays_of_all_vehicles_from_all_routes"]
        for eps_number, model_info in dict_with_agents.items()
        if eps_number in list_of_eps_numbers_max_count_of_vehicles
    ]
    best_eps_for_count_of_vehicles_completing_journey = [
        eps
        for _, eps in sorted(
            zip(
                helper_list_total_average_delays,
                list_of_eps_numbers_max_count_of_vehicles,
            )
        )
    ][0]
    return best_eps_for_count_of_vehicles_completing_journey


def make_archive(dict_with_agents: dict, valid_models_dir: str):
    best_validation_model = choose_best_validation_model(
        dict_with_agents=dict_with_agents
    )
    zipped_best_model = os.path.join(valid_models_dir, best_validation_model)
    shutil.make_archive(zipped_best_model, "zip", zipped_best_model)
    return best_validation_model, zipped_best_model


# ---------------------------------------------------------------------------
# Optuna objective — one trial = one full training run with suggested HPs
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, args):
    """
    Optuna objective function for IDQN hyperparameter tuning.

    Suggests hyperparams, runs a full training loop on BB5B,
    reports validation metrics for pruning, and logs everything to MLflow.
    """

    # ---- 1. Suggest hyperparameters ----
    if args.agent == "IDQN":
        suggested_batch_size = trial.suggest_categorical("BATCH_SIZE", [32, 64, 128, 256])
        suggested_gamma = trial.suggest_categorical("GAMMA", [0.9, 0.95, 0.98, 0.99])
        suggested_eps_end = trial.suggest_float("EPS_END", 0.0, 0.05)
        suggested_eps_decay = trial.suggest_categorical("EPS_DECAY", [100, 220, 500])
        suggested_target_update = trial.suggest_int("TARGET_UPDATE", 500, 5000, step=500)
        suggested_lr = trial.suggest_float("LR", 1e-5, 1e-3, log=True)  # default: 1e-3
        suggested_reward = trial.suggest_categorical("REWARD", ["wait_norm", "queue_maxwait", "pressure"])
        suggested_buffer_size = trial.suggest_categorical("REPLAY_BUFFER_SIZE", [10000, 50000, 100000])  # default: 50000

        print(
            f"\n{'='*60}\n"
            f"OPTUNA TRIAL {trial.number} (IDQN)\n"
            f"  BATCH_SIZE={suggested_batch_size}, GAMMA={suggested_gamma}, "
            f"EPS_END={suggested_eps_end:.4f},\n"
            f"  EPS_DECAY={suggested_eps_decay}, TARGET_UPDATE={suggested_target_update},\n"
            f"  LR={suggested_lr:.6f}, REPLAY_BUFFER_SIZE={suggested_buffer_size},\n"
            f"  REWARD={suggested_reward}\n"
            f"{'='*60}"
        )
    elif args.agent == "IMA2C":
        suggested_lr = trial.suggest_float("LR", 1e-5, 1e-3, log=True)
        suggested_gamma = trial.suggest_categorical("GAMMA", [0.95, 0.96, 0.99])
        suggested_entropy_coef = trial.suggest_float("ENTROPY_COEF", 1e-4, 1e-1, log=True)
        suggested_value_coef = trial.suggest_float("VALUE_COEF", 0.1, 1.0)
        suggested_num_hidden = trial.suggest_categorical("NUM_HIDDEN", [64, 128, 256])
        suggested_num_gru = trial.suggest_categorical("NUM_GRU", [32, 64, 128])
        suggested_batch_size = trial.suggest_categorical("BATCH_SIZE", [64, 120, 256])
        suggested_reward_norm = trial.suggest_categorical("REWARD_NORM", [1000, 2000, 5000])
        suggested_reward = trial.suggest_categorical("REWARD", ["wait_norm", "queue_maxwait", "pressure", "queue_maxwait_neighborhood"])

        print(
            f"\n{'='*60}\n"
            f"OPTUNA TRIAL {trial.number} (IMA2C)\n"
            f"  LR={suggested_lr:.6f}, GAMMA={suggested_gamma}, ENTROPY_COEF={suggested_entropy_coef:.4f},\n"
            f"  VALUE_COEF={suggested_value_coef:.2f}, NUM_HIDDEN={suggested_num_hidden}, NUM_GRU={suggested_num_gru},\n"
            f"  BATCH_SIZE={suggested_batch_size}, REWARD_NORM={suggested_reward_norm}, REWARD={suggested_reward}\n"
            f"{'='*60}"
        )
    else:
        raise ValueError(f"Unsupported agent for Optuna tuning: {args.agent}")

    # ---- 2. Deep-copy configs to avoid cross-trial contamination ----
    agt_config = copy.deepcopy(agent_configs[args.agent])

    # Override with suggested hyperparameters
    if args.agent == "IDQN":
        agt_config["BATCH_SIZE"] = suggested_batch_size
        agt_config["GAMMA"] = suggested_gamma
        agt_config["EPS_END"] = suggested_eps_end
        agt_config["EPS_DECAY"] = suggested_eps_decay
        agt_config["TARGET_UPDATE"] = suggested_target_update
        agt_config["LR"] = suggested_lr
        agt_config["REPLAY_BUFFER_SIZE"] = suggested_buffer_size
    elif args.agent == "IMA2C":
        agt_config["lr_init"] = suggested_lr
        agt_config["gamma"] = suggested_gamma
        agt_config["entropy_coef"] = suggested_entropy_coef
        agt_config["value_coef"] = suggested_value_coef
        agt_config["num_hidden"] = suggested_num_hidden
        agt_config["num_gru"] = suggested_num_gru
        agt_config["batch_size"] = suggested_batch_size
        agt_config["reward_norm"] = suggested_reward_norm
    
    agt_config["reward"] = REWARD_MAP[suggested_reward]

    alg = agt_config["agent"]

    # ---- 3. MDP config ----
    # Ensure global mdp_configs is updated so state functions (like states.ma2c) can see it
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    # IMA2C specific handling (matches logic in main.py)
    if args.agent == 'IMA2C':
        ma2c_mdp = mdp_configs.get('MA2C')
        if ma2c_mdp is not None:
            ma2c_map = ma2c_mdp.get(args.map)
            if ma2c_map is not None:
                mdp_configs['MA2C'] = ma2c_map

    agt_config["mdp"] = mdp_configs.get(args.agent)
    if agt_config["mdp"] is not None:
        management = agt_config["mdp"].get("management")
        if management is not None:
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            agt_config["mdp"]["supervisors"] = supervisors

    # ---- 4. Map config ----
    map_config = copy.deepcopy(map_configs[args.map])
    num_steps_eps = int(
        (map_config["end_time"] - map_config["start_time"]) / map_config["step_length"]
    )
    if args.map == "BB5B":
        start_time, end_time = convert_time_range_to_seconds(args.validation_period)
        map_config["start_time"] = start_time
        map_config["end_time"] = end_time

    route = map_config["route"]
    if route is not None:
        route = os.path.join(args.pwd, route)

    # ---- 5. Create environment with unique trial ID in run_name ----
    # Include optuna trial number to avoid SUMO TraCI label collisions
    trial_id = f"optuna{trial.number}"
    run_name = (
        alg.__name__
        + "-net" + args.net
        + "-activ" + args.activation
        + (f"-neg_slope{args.negative_slope}" if args.activation == "leaky_relu" else "")
        + "-seed" + str(args.seed)
        + "-" + trial_id
    )

    env = MultiSignal(
        run_name,
        args.map,
        os.path.join(args.pwd, map_config["net"]),
        agt_config["state"],
        suggested_reward,
        validation_day_directory_name=args.validation_day,
        validation_period_file_name=args.validation_period,
        route=route,
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        start_time=map_config["start_time"],
        end_time=map_config["end_time"],
        max_distance=agt_config["max_distance"],
        lights=map_config["lights"],
        gui=args.gui,
        log_dir=args.log_dir,
        libsumo=args.libsumo,
        warmup=map_config["warmup"],
        seed=args.seed,
    )

    # ---- 6. Configure agent ----
    agt_config["episodes"] = int(
        args.eps_val * args.validation_interval * 0.8
    )
    agt_config["steps"] = agt_config["episodes"] * num_steps_eps
    agt_config["log_dir"] = os.path.join(args.log_dir, env.connection_name)
    agt_config["models_dir"] = os.path.join(args.models_dir, env.connection_name)
    agt_config["models_for_visualization"] = os.path.join(
        args.models_dir, "models_for_visualization"
    )
    agt_config["num_lights"] = len(env.all_ts_ids)
    agt_config["load"] = args.load

    if not os.path.exists(agt_config["models_dir"]):
        os.makedirs(agt_config["models_dir"], exist_ok=True)
    if not os.path.exists(agt_config["models_for_visualization"]):
        os.makedirs(agt_config["models_for_visualization"], exist_ok=True)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [
            env.obs_shape[key],
            2 if key in env.phases else None,
        ]

    agent = alg(
        agt_config,
        obs_act,
        args.map,
        0,  # trial/thread number for the agent
        net=args.net,
        activation=args.activation,
        negative_slope=args.negative_slope,
    )

    # ---- 7. MLflow run (tagged "optuna") ----
    run = None
    if args.map == "BB5B":
        PARAMS_ALGORITHM = {
            "action_frequency": env.step_length,
            "algorithm": args.agent,
            "number_of_training_episodes": args.eps_val * args.validation_interval
            - args.eps_val,
            "number_of_validation_episodes": args.eps_val,
            "map": args.map,
            "net": args.net,
            "reward": args.reward_type,
            "activation": args.activation,
            "validation_day": args.validation_day,
            "validation_period": args.validation_period,
            "seed": args.seed,
            "phases": str({
                connection_name: len(phases)
                for connection_name, phases in env.phases.items()
            }),
            # Optuna-specific params
            "optuna_trial": trial.number,
            "optuna_study": args.study_name,
            "REWARD_CHOSEN": suggested_reward,
        }
        if args.agent == "IDQN":
            PARAMS_ALGORITHM.update({
                "BATCH_SIZE": suggested_batch_size,
                "GAMMA": suggested_gamma,
                "EPS_END": suggested_eps_end,
                "EPS_DECAY": suggested_eps_decay,
                "TARGET_UPDATE": suggested_target_update,
                "LR": suggested_lr,
                "REPLAY_BUFFER_SIZE": suggested_buffer_size,
            })
        elif args.agent == "IMA2C":
            PARAMS_ALGORITHM.update({
                "lr_init": suggested_lr,
                "gamma": suggested_gamma,
                "entropy_coef": suggested_entropy_coef,
                "value_coef": suggested_value_coef,
                "num_hidden": suggested_num_hidden,
                "num_gru": suggested_num_gru,
                "batch_size": suggested_batch_size,
                "reward_norm": suggested_reward_norm,
            })
        if args.activation == "leaky_relu":
            PARAMS_ALGORITHM["negative_slope"] = args.negative_slope

        run = init_mlflow_optuna_run(args, env, trial, suggested_reward, PARAMS_ALGORITHM)

    # ---- 8. Training loop with pruning ----
    mode = "training"
    dict_with_agents = {}
    validation_step_counter = 0  # for Optuna reporting

    try:
        for i in range(1, args.eps_val * args.validation_interval + 1):
            if args.map == "BB5B":
                if i % args.validation_interval != 0:
                    mode = "training"
                    agent.set_mode("training")
                else:
                    mode = "validation"
                    agent.set_mode("validation")
                env.mode = mode

            obs = env.reset()
            done = False
            while not done:
                act = agent.act(obs)
                obs, rew, done, eps, info = env.step(act)
                if args.map == "BB5B" and run is not None:
                    log_metrics(buf_infos=info, done=done, mode=mode)
                agent.observe(obs, rew, done, info)

            # ---- Validation checkpoint: report to Optuna for pruning ----
            if mode == "validation" and args.map == "BB5B":
                validation_eps_number = int(i / args.validation_interval - 1)

                avg_delay = info["total_average_delays_of_all_vehicles_from_all_routes"]
                vehicles_completed = info["count_of_vehicles_completing_journey"]

                dict_with_agents[f"eps_{validation_eps_number}"] = {
                    "total_average_delays_of_all_vehicles_from_all_routes": avg_delay,
                    "count_of_vehicles_completing_journey": vehicles_completed,
                }

                # Log both metrics to MLflow
                if run is not None:
                    mlflow.log_metric("optuna/avg_delay", avg_delay, step=validation_step_counter)
                    mlflow.log_metric("optuna/vehicles_completed", vehicles_completed, step=validation_step_counter)

                # Report to Optuna (we minimize avg_delay)
                trial.report(avg_delay, validation_step_counter)
                validation_step_counter += 1

                print(
                    f"  Trial {trial.number} | Validation {validation_eps_number}: "
                    f"avg_delay={avg_delay:.4f}, vehicles={vehicles_completed}"
                )

                # Check if Optuna wants to prune this trial
                if trial.should_prune():
                    print(f"  ✂ Trial {trial.number} PRUNED at validation step {validation_step_counter}")
                    if run is not None:
                        mlflow.set_tag("optuna_status", "pruned")
                        mlflow.end_run()
                    env.close()
                    raise optuna.exceptions.TrialPruned()

                # Save validation models
                valid_models_subdir = os.path.join(
                    agt_config["models_dir"], f"eps_{validation_eps_number}"
                )
                if not os.path.exists(valid_models_subdir):
                    os.mkdir(valid_models_subdir)
                for model_name, model in agent.agents.items():
                    model.save(os.path.join(valid_models_subdir, model_name))

        # ---- 9. Training complete — archive best model ----
        env.close()

        final_avg_delay = avg_delay  # last validation value
        final_vehicles = vehicles_completed

        if dict_with_agents:
            best_validation_model, zipped_best_model = make_archive(
                dict_with_agents=dict_with_agents,
                valid_models_dir=agt_config["models_dir"],
            )
            trial.set_user_attr("best_model_zip", zipped_best_model)
            trial.set_user_attr("best_validation_model", best_validation_model)
            trial.set_user_attr("vehicles_completed", final_vehicles)

            trial.set_user_attr("final_avg_delay", final_avg_delay)

            if run is not None:
                log_model_artifact(f"{zipped_best_model}.zip")
                mlflow.set_tag("optuna_status", "completed")
                mlflow.log_metric("optuna/final_avg_delay", final_avg_delay)
                mlflow.log_metric("optuna/final_vehicles_completed", final_vehicles)

        if run is not None:
            mlflow.end_run()

        # Cleanup models dir to save disk space
        if os.path.exists(agt_config["models_dir"]):
            shutil.rmtree(agt_config["models_dir"])

        print(
            f"  ✓ Trial {trial.number} COMPLETED: "
            f"avg_delay={final_avg_delay:.4f}, vehicles={final_vehicles}"
        )

        # Return the metric to Optuna (minimize avg_delay)
        return final_avg_delay

    except optuna.exceptions.TrialPruned:
        raise  # re-raise so Optuna handles it
    except Exception as e:
        # Cleanup on error
        print(f"  ✗ Trial {trial.number} FAILED: {e}")
        try:
            env.close()
        except Exception:
            pass
        if run is not None:
            mlflow.set_tag("optuna_status", "failed")
            mlflow.set_tag("optuna_error", str(e))
            mlflow.end_run()
        raise


# ---------------------------------------------------------------------------
# Post-study plotting
# ---------------------------------------------------------------------------
def plot_study_results(study, args):
    """Generate summary plots after the Optuna study completes."""
    plots_dir = os.path.join(os.path.dirname(__file__), "reports", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(complete_trials) < 2:
        print("Not enough completed trials to generate plots.")
        return

    # ---- Plot 1: Optimization History (avg_delay per trial) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    trial_nums = [t.number for t in complete_trials]
    delays = [t.value for t in complete_trials]
    best_so_far = []
    current_best = float('inf')
    for d in delays:
        current_best = min(current_best, d)
        best_so_far.append(current_best)

    ax.bar(trial_nums, delays, color='#e94560', alpha=0.7, label='Trial Avg Delay')
    ax.plot(trial_nums, best_so_far, color='#0f3460', linewidth=2.5,
            marker='o', markersize=6, label='Best So Far')
    ax.set_xlabel('Trial Number', color='white', fontsize=12)
    ax.set_ylabel('Avg Delay (s)', color='white', fontsize=12)
    ax.set_title('Optuna Optimization History — Avg Delay', color='white', fontsize=14)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(plots_dir, "optuna_optimization_history.png")
    plt.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path1}")

    # ---- Plot 2: Vehicles vs Delay scatter ----
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    vehicles = []
    trial_delays = []
    labels = []
    for t in complete_trials:
        v = t.user_attrs.get("vehicles_completed")
        d = t.value
        if v is not None:
            vehicles.append(v)
            trial_delays.append(d)
            labels.append(f"T{t.number}")

    if vehicles:
        scatter = ax.scatter(trial_delays, vehicles, c=vehicles, cmap='RdYlGn',
                             s=120, edgecolors='white', linewidths=0.8, zorder=5)
        plt.colorbar(scatter, ax=ax, label='Vehicles Completed')

        # Annotate best trial
        best_idx = trial_delays.index(min(trial_delays))
        ax.annotate(f'Best: {labels[best_idx]}', xy=(trial_delays[best_idx], vehicles[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    color='#e94560', fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#e94560'))

        ax.set_xlabel('Avg Delay (s) — Lower is Better →', color='white', fontsize=12)
        ax.set_ylabel('Vehicles Completed — Higher is Better →', color='white', fontsize=12)
        ax.set_title('Vehicles Completed vs Avg Delay (per trial)', color='white', fontsize=14)
        ax.tick_params(colors='white')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path2 = os.path.join(plots_dir, "optuna_vehicles_vs_delay.png")
        plt.savefig(path2, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved: {path2}")

    # ---- Plot 3: Parameter Importance ----
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1a1a2e')
            ax.set_facecolor('#16213e')

            params = list(importances.keys())
            values = list(importances.values())
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))

            ax.barh(params, values, color=colors, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Importance', color='white', fontsize=12)
            ax.set_title('Hyperparameter Importance (for Avg Delay)', color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.invert_yaxis()
            plt.tight_layout()
            path3 = os.path.join(plots_dir, "optuna_param_importance.png")
            plt.savefig(path3, dpi=150, facecolor=fig.get_facecolor())
            plt.close()
            print(f"  Saved: {path3}")
    except Exception as e:
        print(f"  Could not generate param importance plot: {e}")

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print(f"{'Trial':>6} | {'Delay (s)':>10} | {'Vehicles':>10} | {'LR':>10} | {'Batch':>6} | {'Buffer':>7} | {'Reward':>14} | {'Gamma':>5}")
    print(f"{'-'*80}")
    for t in sorted(complete_trials, key=lambda x: x.value):
        v = t.user_attrs.get("vehicles_completed", "?")
        p = t.params
        print(
            f"  T{t.number:<4} | {t.value:>10.4f} | {v:>10} | "
            f"{p.get('LR', '?'):>10.6f} | {p.get('BATCH_SIZE', '?'):>6} | "
            f"{p.get('REPLAY_BUFFER_SIZE', '?'):>7} | {p.get('REWARD', '?'):>14} | "
            f"{p.get('GAMMA', '?'):>5}"
        )
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for RESCO traffic simulation"
    )

    # ---- Same args as main.py ----
    ap.add_argument("--agent", type=str, default="IDQN", choices=["IDQN", "IMA2C"])
    ap.add_argument("--eps_val", type=int, default=10)
    ap.add_argument("--validation_interval", type=int, default=6)
    ap.add_argument(
        "--map", type=str, default="BB5B",
        choices=[
            "grid4x4", "arterial4x4", "ingolstadt1", "ingolstadt7",
            "ingolstadt21", "cologne1", "cologne3", "cologne8", "BB5B",
        ],
    )
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument(
        "--log_dir", type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "results" + os.sep),
    )
    ap.add_argument(
        "--models_dir", type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "models" + os.sep),
    )
    ap.add_argument("--gui", type=bool, default=False)
    ap.add_argument("--load", type=bool, default=False)
    ap.add_argument("--net", type=str, default="default")
    ap.add_argument("--activation", type=str, default="relu")
    ap.add_argument("--validation_day", type=str, default="26NovFull")
    ap.add_argument("--validation_period", type=str, default="7-8am")
    ap.add_argument("--negative_slope", type=float, default=0.01)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument(
        "--reward-type", type=str, default="queue_maxwait",
        choices=["wait", "wait_norm", "pressure", "queue_maxwait", "queue_maxwait_neighborhood"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_tag", type=str, default=None, help="Tag to group experiments in MLflow")
    ap.add_argument("--description", type=str, default=None, help="Description for the MLflow run")
    ap.add_argument("--experiment_suffix", type=str, default=None, help="String to append to the MLflow experiment name")

    # ---- Optuna-specific args ----
    ap.add_argument("--n_trials", type=int, default=20,
                    help="Number of Optuna trials to run")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Number of parallel trials (each uses 1 SUMO instance)")
    ap.add_argument("--study_name", type=str, default="resco_optuna",
                    help="Optuna study name")
    ap.add_argument("--pruner", type=str, default="hyperband",
                    choices=["median", "hyperband"],
                    help="Pruning strategy (hyperband is more aggressive)")
    ap.add_argument("--storage", type=str, default="sqlite:///optuna_resco.db",
                    help="Optuna storage backend (SQLite for dashboard)")

    args = ap.parse_args()

    # Validate
    if args.libsumo and "LIBSUMO_AS_TRACI" not in os.environ:
        raise EnvironmentError(
            "Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo"
        )

    if not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir, exist_ok=True)

    # Create dedicated Optuna models directory
    optuna_models_dir = os.path.join(args.models_dir, "optuna")
    if not os.path.exists(optuna_models_dir):
        os.makedirs(optuna_models_dir, exist_ok=True)

    # Seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Create Optuna study ----
    if args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        # MedianPruner: don't prune until 3 validation checkpoints have been seen
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
        )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",  # minimize avg_delay
        pruner=pruner,
        load_if_exists=True,
    )

    # Callback to save best model
    best_model_callback = SaveBestModelCallback(optuna_models_dir)

    print(
        f"\n{'='*60}\n"
        f"OPTUNA STUDY: {args.study_name}\n"
        f"  Best models directory: {optuna_models_dir}\n"
        f"  Agent: {args.agent} | Map: {args.map}\n"
        f"  Trials: {args.n_trials} | Parallel jobs: {args.n_jobs}\n"
        f"  Episodes per trial: {args.eps_val * args.validation_interval} "
        f"({args.eps_val} val × {args.validation_interval} interval)\n"
        f"  Pruner: {args.pruner}\n"
        f"  Direction: minimize avg_delay\n"
        f"  Storage: {args.storage}\n"
        f"{'='*60}\n"
    )

    # ---- Run the study ----
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        callbacks=[best_model_callback],
        show_progress_bar=True,
    )

    # ---- Print results ----
    print(f"\n{'='*60}")
    print("OPTUNA STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Number of finished trials: {len(study.trials)}")

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Pruned:    {len(pruned_trials)}")
    print(f"  Completed: {len(complete_trials)}")

    if complete_trials:
        print(f"\nBest trial (lowest avg_delay):")
        best = study.best_trial
        print(f"  Value (avg_delay): {best.value:.4f}")
        if "vehicles_completed" in best.user_attrs:
            print(f"  Vehicles completed: {best.user_attrs['vehicles_completed']}")
        print(f"  Params:")
        for key, value in best.params.items():
            print(f"    {key}: {value}")

    # ---- Generate plots ----
    plot_study_results(study, args)


if __name__ == "__main__":
    main()
