import argparse
import multiprocessing as mp
import os
import random
import shutil

import neptune.new as neptune
import numpy as np
import torch

from config.agent_config import agent_configs
from config.map_config import map_configs
from config.mdp_config import mdp_configs
from multi_signal import MultiSignal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment_name",
        type=str,
        required=True
    )
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "results" + os.sep),
    )
    ap.add_argument(
        "--models_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "models" + os.sep),
    )
    ap.add_argument("--gui", type=bool, default=True)
    ap.add_argument("--load", type=bool, default=True)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument(
        "--tr", type=int, default=0
    )  # Can't multi-thread with libsumo, provide a trial number
    args = ap.parse_args()

    if args.libsumo and "LIBSUMO_AS_TRACI" not in os.environ:
        raise EnvironmentError(
            "Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo"
        )
    
    run = neptune.init_run(
        with_id=args.experiment_name,
        mode="read-only"
    )
    agent, map, episodes, net, activation, seed, negative_slope, validation_day = fetch_experiment_data(run)

    np.random.seed(seed)
    random.seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr, run, 
                  agent=agent, 
                  map=map, 
                  episodes=episodes, 
                  net=net, 
                  activation=activation, 
                  seed=seed, 
                  negative_slope=negative_slope, 
                  validation_day=validation_day)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials + 1):
            pool.apply_async(run_trial, args=(args, trial, run), 
                             kwds={"agent": agent, 
                                   "map": map, 
                                   "episodes": episodes, 
                                   "net": net, 
                                   "activation": activation, 
                                   "seed": seed, 
                                   "negative_slope": negative_slope,
                                   "validation_day": validation_day})
        pool.close()
        pool.join()


def run_trial(args, trial, run, **kwargs):
    agent = kwargs.get("agent")
    map = kwargs.get("map")
    episodes = kwargs.get("episodes")
    net = kwargs.get("net")
    activation = kwargs.get("activation")
    seed = kwargs.get("seed")
    negative_slope = kwargs.get("negative_slope")
    validation_day = kwargs.get("validation_day")
    
    mdp_config = mdp_configs.get(agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[agent] = mdp_config

    agt_config = agent_configs[agent]
    alg = agt_config["agent"]

    if mdp_config is not None:
        agt_config["mdp"] = mdp_config
        management = agt_config["mdp"].get("management")
        if management is not None:  # Save some time and precompute the reverse mapping
            supervisors = dict()
            for manager in management:
                workers = management[manager]
                for worker in workers:
                    supervisors[worker] = manager
            mdp_config["supervisors"] = supervisors

    map_config = map_configs[map]
    num_steps_eps = int(
        (map_config["end_time"] - map_config["start_time"]) / map_config["step_length"]
    )
    route = map_config["route"]
    if route is not None:
        route = os.path.join(args.pwd, route)
    if map == "grid4x4" or map == "arterial4x4":
        if not os.path.exists(route):
            raise EnvironmentError(
                "You must decompress environment files defining traffic flow"
            )

    env = MultiSignal(
        alg.__name__
        + "-net"
        + net
        + "-activ"
        + activation
        + (f"-neg_slope{negative_slope}" if activation == "leaky_relu" else "")
        + "-seed"
        + str(seed)
        + "-tr"
        + str(trial),
        map,
        os.path.join(args.pwd, map_config["net"]),
        agt_config["state"],
        agt_config["reward"],
        validation_day_directory_name=validation_day,
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
        seed=seed,
    )

    agt_config["episodes"] = int(episodes * 0.8)  # schedulers decay over 80% of steps
    agt_config["steps"] = episodes * num_steps_eps
    agt_config["log_dir"] = os.path.join(args.log_dir, env.connection_name)
    agt_config["models_dir"] = os.path.join(args.models_dir, env.connection_name)
    agt_config["models_for_visualization"] = os.path.join(args.models_dir, "models_for_visualization" + os.sep)
    agt_config["num_lights"] = len(env.all_ts_ids)
    agt_config["load"] = args.load
    
    if not os.path.exists(args.models_dir):
        os.mkdir(args.models_dir)
    if not os.path.exists(agt_config["models_for_visualization"]):
        os.mkdir(agt_config["models_for_visualization"])

    download_models(run, agt_config["models_for_visualization"])

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [
            env.obs_shape[key],
            2 if key in env.phases else None,
        ]
    if agent == "IDQN" or args.agent == "IPPO":
        agent = alg(agt_config, obs_act, map, trial, 
                    net=net, 
                    activation=activation,
                    negative_slope=negative_slope)
    else:
        agent = alg(agt_config, obs_act, map, trial)
    remove_files(agt_config["models_for_visualization"])
    mode = "validation"
    env.mode = mode
    obs = env.reset()

    done = False
    while not done:
        act = agent.act(obs)
        obs, rew, done, eps, info = env.step(act)
        agent.observe(obs, rew, done, info)
        
    env.close()


def download_models(run: neptune.Run, path_to_models: str):
    if not run.exists("models"):
        raise FileNotFoundError("Directory 'models' does not exists.")
    model_name = list(run.get_structure().get("models").keys())[0]
    run[f"models/{model_name}"].download(path_to_models)
    
    shutil.unpack_archive(filename=os.path.join(path_to_models, f"{model_name}.zip"),
                          extract_dir=path_to_models,
                          format="zip")


def fetch_experiment_data(run: neptune.Run):
    params = run["parameters"].fetch()
    agent = params.get("algorithm")
    map = params.get("map")
    episodes = params.get("number_episodes")
    net = params.get("net")
    activation = params.get("activation")
    seed = params.get("seed")
    negative_slope = params.get("negative_slope", 0.01)
    validation_day = params.get("validation_day", "26NovFull")
    return agent, map, episodes, net, activation, seed, negative_slope, validation_day


def remove_files(path: str):
    for file in os.listdir(path):
        if not file.startswith("."):
            try:
                shutil.rmtree(path=os.path.join(path, file))
            except NotADirectoryError:
                os.remove(path=os.path.join(path, file))


if __name__ == "__main__":
    main()
