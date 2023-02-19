import pathlib
import os
import multiprocessing as mp
import neptune.new as neptune
from neptune.new import Run

from multi_signal import MultiSignal
import argparse
from config.agent_config import agent_configs
from config.map_config import map_configs
from config.mdp_config import mdp_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent",
        type=str,
        default="STOCHASTIC",
        choices=[
            "STOCHASTIC",
            "MAXWAVE",
            "MAXPRESSURE",
            "IDQN",
            "IPPO",
            "MPLight",
            "MA2C",
            "FMA2C",
            "MPLightFULL",
            "FMA2CFull",
            "FMA2CVAL",
        ],
    )
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument(
        "--map",
        type=str,
        default="ingolstadt1",
        choices=[
            "grid4x4",
            "arterial4x4",
            "ingolstadt1",
            "ingolstadt7",
            "ingolstadt21",
            "cologne1",
            "cologne3",
            "cologne8",
            "BB5B",
        ],
    )
    ap.add_argument("--pwd", type=str, default=os.path.dirname(__file__))
    ap.add_argument(
        "--log_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "results" + os.sep),
    )
    ap.add_argument("--gui", type=bool, default=True)
    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument(
        "--tr", type=int, default=0
    )  # Can't multi-thread with libsumo, provide a trial number
    args = ap.parse_args()

    if args.libsumo and "LIBSUMO_AS_TRACI" not in os.environ:
        raise EnvironmentError(
            "Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo"
        )

    if args.procs == 1 or args.libsumo:
        run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials + 1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()


def run_trial(args, trial):
    mdp_config = mdp_configs.get(args.agent)
    if mdp_config is not None:
        mdp_map_config = mdp_config.get(args.map)
        if mdp_map_config is not None:
            mdp_config = mdp_map_config
        mdp_configs[args.agent] = mdp_config

    agt_config = agent_configs[args.agent]
    # Te linijki niezbyt mają sens
    # agt_map_config = agt_config.get(args.map)
    # if agt_map_config is not None:
    #     agt_config = agt_map_config
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

    map_config = map_configs[args.map]
    num_steps_eps = int(
        (map_config["end_time"] - map_config["start_time"]) / map_config["step_length"]
    )
    route = map_config["route"]
    if route is not None:
        route = os.path.join(args.pwd, route)
    if args.map == "grid4x4" or args.map == "arterial4x4":
        if not os.path.exists(route):
            raise EnvironmentError(
                "You must decompress environment files defining traffic flow"
            )

    env = MultiSignal(
        alg.__name__ + "-tr" + str(trial),
        args.map,
        os.path.join(args.pwd, map_config["net"]),
        agt_config["state"],
        agt_config["reward"],
        route=route,
        step_length=map_config["step_length"],
        yellow_length=map_config["yellow_length"],
        step_ratio=map_config["step_ratio"],
        end_time=map_config["end_time"],
        max_distance=agt_config["max_distance"],
        lights=map_config["lights"],
        gui=args.gui,
        log_dir=args.log_dir,
        libsumo=args.libsumo,
        warmup=map_config["warmup"],
    )

    agt_config["episodes"] = int(args.eps * 0.8)  # schedulers decay over 80% of steps
    agt_config["steps"] = agt_config["episodes"] * num_steps_eps
    agt_config["log_dir"] = os.path.join(args.log_dir, env.connection_name)
    agt_config["num_lights"] = len(env.all_ts_ids)

    # Get agent id's, observation shapes, and action sizes from env
    obs_act = dict()
    for key in env.obs_shape:
        obs_act[key] = [
            env.obs_shape[key],
            len(env.phases[key]) if key in env.phases else None,
        ]
    agent = alg(agt_config, obs_act, args.map, trial)
    run = None
    if args.map == "BB5B":
        PARAMS_ALGORITHM = {
            "algorithm": args.agent,
            "number_episodes": args.eps,
            "map": args.map
        }  # TODO
        run = neptune.init_run(
            api_token=None,
            project="pgora/Malaysia2",
            name=f"{args.agent}-sumo-v0",
            description=f"Apply {args.agent} algorithm to the sumo-v0 environment",
            tags=[
                "sumo-v0",
                f"{args.agent}",
                "stable-baselines3",
                # "10 episodes - train, 1 episode - validation on new own generated file",
                "4 phases for PBB_Junc and SIRIM_Junc, 3 phases for INFMain_Junc - Full",
                "no new vehicles after 1 hour",
                f"Reward: {agent_configs[args.agent]['reward']}",
                # "5e5 steps",
                "7-8 am",
                # "aggregating data from lanes on the same road",
            ],
        )
        run["parameters"] = PARAMS_ALGORITHM

    for _ in range(args.eps):
        obs = env.reset()
        done = False
        while not done:
            act = agent.act(obs)
            obs, rew, done, eps, info = env.step(act)
            if args.map == "BB5B":
                log_metrics(buf_infos=info, run=run, done=done, mode="train")
            agent.observe(obs, rew, done, info)
    env.close()


def log_metrics(buf_infos: dict, run: Run, done: bool, mode: str):

    if not done:
        # run["metrics/" + mode + "/learning_rate"].log(model.learning_rate)           #ignore LR for now
        run["metrics/" + mode + "/observation"].log(str(buf_infos["observation"]))
        run["metrics/" + mode + "/action_dict"].log(str(buf_infos["action"]))
        run["metrics/" + mode + "/action/INFMain_Junc"].log(str(buf_infos["action"]['INFMain_Junc']))
        run["metrics/" + mode + "/action/PBB_Junc"].log(str(buf_infos["action"]['PBB_Junc']))
        run["metrics/" + mode + "/action/SIRIM_Junc"].log(str(buf_infos["action"]['SIRIM_Junc']))
        run["metrics/" + mode + "/reward_dict"].log(str(buf_infos['reward']))
        run["metrics/" + mode + "/reward/INFMain_Junc"].log(str(buf_infos['reward']['INFMain_Junc']))
        run["metrics/" + mode + "/reward/PBB_Junc"].log(str(buf_infos['reward']['PBB_Junc']))
        run["metrics/" + mode + "/reward/SIRIM_Junc"].log(str(buf_infos['reward']['SIRIM_Junc']))
        run["metrics/" + mode + "/current_number_of_vehicles"].log(
            buf_infos["current_number_of_vehicles"]
        )
        run[
            "metrics/"
            + mode
            + "/number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"
        ].log(
            buf_infos[
                "number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"
            ]
        )
        run[
            "metrics/"
            + mode
            + "/number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
        ].log(
            buf_infos[
                "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
            ]
        )
        run[
            "metrics/"
            + mode
            + "/waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
        ].log(
            buf_infos["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"]
        )
        run["metrics/" + mode + "/calculate_average_delta_of_delays_after_action"].log(
            buf_infos["calculate_average_delta_of_delays_after_action"]
        )
        run["metrics/" + mode + "/number_of_vehicles_that_passed_through_the_intersections_in_last_steps"].log(
            buf_infos["number_of_vehicles_that_passed_through_the_intersections_in_last_steps"]
        )
        run[
            "metrics/" + mode + "/current_average_delays_of_all_vehicles_in_simulation"
        ].log(buf_infos["current_average_delays_of_all_vehicles_in_simulation"])
    else:
        # run["metrics/" + mode + "/learning_rate"].log(model.learning_rate)             #ignore LR for now
        run["metrics/" + mode + "/observation"].log(str(buf_infos["observation"]))
        run["metrics/" + mode + "/action_dict"].log(str(buf_infos["action"]))
        run["metrics/" + mode + "/action/INFMain_Junc"].log(str(buf_infos["action"]['INFMain_Junc']))
        run["metrics/" + mode + "/action/PBB_Junc"].log(str(buf_infos["action"]['PBB_Junc']))
        run["metrics/" + mode + "/action/SIRIM_Junc"].log(str(buf_infos["action"]['SIRIM_Junc']))
        run["metrics/" + mode + "/reward_dict"].log(str(buf_infos['reward']))
        run["metrics/" + mode + "/reward/INFMain_Junc"].log(str(buf_infos['reward']['INFMain_Junc']))
        run["metrics/" + mode + "/reward/PBB_Junc"].log(str(buf_infos['reward']['PBB_Junc']))
        run["metrics/" + mode + "/reward/SIRIM_Junc"].log(str(buf_infos['reward']['SIRIM_Junc']))
        run["metrics/" + mode + "/count_of_all_vehicles_in_simulation"].log(
            buf_infos["count_of_all_vehicles_in_simulation"]
        )
        run[
            "metrics/"
            + mode
            + "/number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"
        ].log(
            buf_infos[
                "number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"
            ]
        )
        run[
            "metrics/"
            + mode
            + "/number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
        ].log(
            buf_infos[
                "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"
            ]
        )
        run[
            "metrics/"
            + mode
            + "/waiting_time_all_vehicles_for_the_last_time_step_in_simulation"
        ].log(
            buf_infos["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"]
        )
        # run[
        #     "metrics/"
        #     + mode
        #     + "/total_waiting_time_all_vehicles_in_simulation_in_episode"
        # ].log(
        #     buf_infos["total_waiting_time_all_vehicles_in_simulation_in_episode"]
        # )  # not present in Pawel's metrics
        run[
            "metrics/" + mode + "/total_waiting_time_on_the_incoming_lanes_in_episode"
        ].log(buf_infos["total_waiting_time_on_the_incoming_lanes_in_episode"])
        run[
            "metrics/" + mode + "/total_waiting_time_on_the_incoming_lanes_in_episode2"
            ].log(buf_infos["total_waiting_time_on_the_incoming_lanes_in_episode2"])
        run["metrics/" + mode + "/count_of_vehicles_completing_journey"].log(
            buf_infos["count_of_vehicles_completing_journey"]
        )
        run["metrics/" + mode + "/total_time_of_journey"].log(
            buf_infos["total_time_of_journey"]
        )
        run["metrics/" + mode + "/average_time_of_journey"].log(
            buf_infos["average_time_of_journey"]
        )
        run[
            "metrics/" + mode + "/total_sum_delays_of_all_vehicles_from_all_routes"
        ].log(buf_infos["total_sum_delays_of_all_vehicles_from_all_routes"])
        run[
            "metrics/" + mode + "/total_average_delays_of_all_vehicles_from_all_routes"
        ].log(buf_infos["total_average_delays_of_all_vehicles_from_all_routes"])
        run["metrics/" + mode + "/total_average_delays_real_times_by_ideal_times"].log(
            buf_infos["total_average_delays_real_times_by_ideal_times"]
        )
        run["metrics/" + mode + "/total_average_delays_with_weights"].log(
            buf_infos["total_average_delays_with_weights"]
        )
        run[
            "metrics/"
            + mode
            + "/total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey"
        ].log(
            buf_infos[
                "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey"
            ]
        )
        chosen_routes = [
                "Infout-HLin",
                "PBBN-FMin",
                "PBBN-SirimS",
                "PBBN-SirimW",
                "PBBN-SKE",
                "PBBW-FMin",
                "PBBW-SKE",
                "SirimE-HLin",
                "SirimS-HLin",
                "SirimS-PBBN",
                "SirimW-HLin",
                "SirimW-SirimE",
                "SKE-HLin",
                "SKE-PBBN",
                ]
        for route_id in buf_infos['routes'].keys():
            if route_id in chosen_routes:
                run["metrics/" + mode + "/routes/" + route_id + "/length"].log(buf_infos['routes'][route_id]['length'])
                run["metrics/" + mode + "/routes/" + route_id + "/total_number_of_all_vehicles_generated-ThruPut_Scheduled"].log(buf_infos['routes'][route_id]['total_number_of_all_vehicles_generated-ThruPut_Scheduled'])
                run["metrics/" + mode + "/routes/" + route_id + "/total_number_of_all_vehicles_completing_journey-ThruPut_Actual"].log(buf_infos['routes'][route_id]['total_number_of_all_vehicles_completing_journey-ThruPut_Actual'])
                run["metrics/" + mode + "/routes/" + route_id + "/throughput_of_the_route-ThruPut_Idx"].log(buf_infos['routes'][route_id]['throughput_of_the_route-ThruPut_Idx'])
                run["metrics/" + mode + "/routes/" + route_id + "/total_travel_time_of_all_vehicles"].log(str(buf_infos['routes'][route_id]['total_travel_time_of_all_vehicles']))
                run["metrics/" + mode + "/routes/" + route_id + "/total_average_travel_time_of_all_vehicles"].log(buf_infos['routes'][route_id]['total_average_travel_time_of_all_vehicles'])
                run["metrics/" + mode + "/routes/" + route_id + "/total_delays_of_all_vehicles"].log(str(buf_infos['routes'][route_id]['total_delays_of_all_vehicles']))
                run["metrics/" + mode + "/routes/" + route_id + "/total_average_delays_of_all_vehicles-Delay_Idx_Average"].log(buf_infos['routes'][route_id]['total_average_delays_of_all_vehicles-Delay_Idx_Average'])
                run["metrics/" + mode + "/routes/" + route_id + "/Delay_Idx_StDev"].log(buf_infos['routes'][route_id]['Delay_Idx_StDev'])
                run["metrics/" + mode + "/routes/" + route_id + "/total_average_delays_of_all_vehicles_with_weights"].log(buf_infos['routes'][route_id]['total_average_delays_of_all_vehicles_with_weights'])
                for veh_type in buf_infos['routes'][route_id]['vehicle_type']:
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/ideal/travel_time"].log(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['ideal']['travel_time'])
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/real/number_of_vehicles"].log(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['number_of_vehicles'])
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/real/total_travel_time"].log(str(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['total_travel_time']) if len(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['vehicle_id']) != 0 else "0")
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/real/average_travel_time"].log(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['average_travel_time'])
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/real/delays/total"].log(str(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['delays']['total']) if len(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['vehicle_id']) != 0 else "0")
                    run["metrics/" + mode + "/routes/" + route_id + "/vehicle_type/" + veh_type + "/real/delays/average"].log(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['delays']['average'] if len(buf_infos['routes'][route_id]['vehicle_type'][veh_type]['real']['vehicle_id']) != 0 else 0)


if __name__ == "__main__":
    main()
