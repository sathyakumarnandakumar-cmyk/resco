"""
MLflow logging utilities for RESCO BB5B experiments.

This module handles all MLflow metric logging, run initialization,
and artifact saving for the BB5B traffic signal control experiments.
"""

import os

import mlflow


# Step counters for MLflow metric logging (per mode)
_metric_step = {"training": 0, "validation": 0}


def init_mlflow_run(args, env, main_dir: str):
    """
    Initialize an MLflow run with experiment parameters, tags, and description.

    Args:
        args: Parsed command-line arguments
        env: MultiSignal environment instance
        main_dir: Absolute path to the directory containing main.py

    Returns:
        The MLflow run object
    """
    from datetime import datetime

    START_TIME = datetime.now().strftime("%d_%m_%H_%M")

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
    }
    if args.activation == "leaky_relu":
        PARAMS_ALGORITHM["negative_slope"] = args.negative_slope

    experiment_name = f"{args.agent}-sumo-{args.map}"
    if args.experiment_suffix:
        experiment_name += f"_{args.experiment_suffix}"

    # Use absolute path so the DB is always in the main.py directory
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(main_dir, 'mlflow.db')}")

    mlflow.set_experiment(experiment_name)

    tags = {
        "environment": "sumo-v0",
        "agent": args.agent,
        "net": args.net,
        "activation": args.activation,
        "framework": "stable-baselines3",
        "phase_config": "4 phases for PBB_Junc and SIRIM_Junc, 3 phases for INFMain_Junc - Full",
        "traffic_note": "no new vehicles after 1 hour",
        "validation_period": args.validation_period,
    }
    if args.group_tag:
        tags["group_tag"] = args.group_tag

    run_description = args.description or f"Apply {args.agent} algorithm to the sumo-v0 environment on {args.map}"
    run_name = (
        f"{args.agent}-{args.map}"
        f"_net-{args.net}"
        f"_act-{args.activation}"
        + (f"_ns-{args.negative_slope}" if args.activation == "leaky_relu" else "")
        + f"_rw-{args.reward_type}"
        f"_seed-{args.seed}"
        f"_{START_TIME}"
    )

    run = mlflow.start_run(
        run_name=run_name,
        description=run_description,
        tags=tags,
    )
    mlflow.log_params(PARAMS_ALGORITHM)

    # Reset step counters for this run
    _metric_step["training"] = 0
    _metric_step["validation"] = 0

    return run


def end_mlflow_run():
    """End the current MLflow run."""
    mlflow.end_run()


def log_model_artifact(zipped_model_path: str):
    """Upload a zipped model file as an MLflow artifact."""
    mlflow.log_artifact(zipped_model_path)


def log_metrics(buf_infos: dict, done: bool, mode: str):
    """
    Log metrics to MLflow for the current simulation step.

    Only episode-end metrics are logged (when done=True) to keep the DB lean.
    Per-step metrics are commented out but preserved for easy re-enabling.

    Args:
        buf_infos: Dictionary of simulation info/metrics from the environment
        done: Whether the episode has finished
        mode: "training" or "validation"
    """
    step = _metric_step[mode]
    _metric_step[mode] += 1

    if not done:
        # --- Per-step metrics commented out to reduce DB size ---
        # Uncomment these if you need fine-grained per-step tracking:
        # mlflow.log_metric(f"metrics/{mode}/action/INFMain_Junc", buf_infos["action"]["INFMain_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/action/PBB_Junc", buf_infos["action"]["PBB_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/action/SIRIM_Junc", buf_infos["action"]["SIRIM_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/reward/INFMain_Junc", buf_infos["reward"]["INFMain_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/reward/PBB_Junc", buf_infos["reward"]["PBB_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/reward/SIRIM_Junc", buf_infos["reward"]["SIRIM_Junc"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/current_number_of_vehicles", buf_infos["current_number_of_vehicles"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes", buf_infos["number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/number_of_all_halting_vehicles_for_the_last_time_step_in_simulation", buf_infos["number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/waiting_time_all_vehicles_for_the_last_time_step_in_simulation", buf_infos["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/calculate_average_delta_of_delays_after_action", buf_infos["calculate_average_delta_of_delays_after_action"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/number_of_vehicles_that_passed_through_the_intersections_in_last_steps", buf_infos["number_of_vehicles_that_passed_through_the_intersections_in_last_steps"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/current_average_delays_of_all_vehicles_in_simulation", buf_infos["current_average_delays_of_all_vehicles_in_simulation"], step=step)
        pass
    else:
        # === Episode-level metrics (logged once per episode) ===
        mlflow.log_metric(
            f"metrics/{mode}/count_of_all_vehicles_in_simulation",
            buf_infos["count_of_all_vehicles_in_simulation"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/count_of_vehicles_completing_journey",
            buf_infos["count_of_vehicles_completing_journey"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/average_time_of_journey",
            buf_infos["average_time_of_journey"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/total_sum_delays_of_all_vehicles_from_all_routes",
            buf_infos["total_sum_delays_of_all_vehicles_from_all_routes"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/total_average_delays_of_all_vehicles_from_all_routes",
            buf_infos["total_average_delays_of_all_vehicles_from_all_routes"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/total_average_delays_with_weights",
            buf_infos["total_average_delays_with_weights"],
            step=step,
        )
        mlflow.log_metric(
            f"metrics/{mode}/total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey",
            buf_infos[
                "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey"
            ],
            step=step,
        )
        # --- Commented out less critical episode-end metrics ---
        # mlflow.log_metric(f"metrics/{mode}/number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes", buf_infos["number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/number_of_all_halting_vehicles_for_the_last_time_step_in_simulation", buf_infos["number_of_all_halting_vehicles_for_the_last_time_step_in_simulation"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/waiting_time_all_vehicles_for_the_last_time_step_in_simulation", buf_infos["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/total_waiting_time_on_the_incoming_lanes_in_episode", buf_infos["total_waiting_time_on_the_incoming_lanes_in_episode"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/total_waiting_time_on_the_incoming_lanes_in_episode2", buf_infos["total_waiting_time_on_the_incoming_lanes_in_episode2"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/total_time_of_journey", buf_infos["total_time_of_journey"], step=step)
        # mlflow.log_metric(f"metrics/{mode}/total_average_delays_real_times_by_ideal_times", buf_infos["total_average_delays_real_times_by_ideal_times"], step=step)

        # === Per-route metrics (only key ones: throughput + average delay) ===
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
        for route_id in buf_infos["routes"].keys():
            if route_id in chosen_routes:
                mlflow.log_metric(
                    f"metrics/{mode}/routes/{route_id}/throughput_of_the_route-ThruPut_Idx",
                    buf_infos["routes"][route_id]["throughput_of_the_route-ThruPut_Idx"],
                    step=step,
                )
                mlflow.log_metric(
                    f"metrics/{mode}/routes/{route_id}/total_average_delays_of_all_vehicles-Delay_Idx_Average",
                    buf_infos["routes"][route_id][
                        "total_average_delays_of_all_vehicles-Delay_Idx_Average"
                    ],
                    step=step,
                )
                # --- Commented out other per-route metrics ---
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/length", buf_infos["routes"][route_id]["length"], step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_number_of_all_vehicles_generated-ThruPut_Scheduled", buf_infos["routes"][route_id]["total_number_of_all_vehicles_generated-ThruPut_Scheduled"], step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_number_of_all_vehicles_completing_journey-ThruPut_Actual", buf_infos["routes"][route_id]["total_number_of_all_vehicles_completing_journey-ThruPut_Actual"], step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_travel_time_of_all_vehicles", sum(buf_infos["routes"][route_id]["total_travel_time_of_all_vehicles"]), step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_average_travel_time_of_all_vehicles", buf_infos["routes"][route_id]["total_average_travel_time_of_all_vehicles"], step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_delays_of_all_vehicles", sum(buf_infos["routes"][route_id]["total_delays_of_all_vehicles"]), step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/Delay_Idx_StDev", buf_infos["routes"][route_id]["Delay_Idx_StDev"], step=step)
                # mlflow.log_metric(f"metrics/{mode}/routes/{route_id}/total_average_delays_of_all_vehicles_with_weights", buf_infos["routes"][route_id]["total_average_delays_of_all_vehicles_with_weights"], step=step)

                # Text artifacts: per-vehicle lists (still saved for detailed analysis)
                mlflow.log_text(
                    str(buf_infos["routes"][route_id]["total_travel_time_of_all_vehicles"]),
                    f"routes/{route_id}/{mode}_total_travel_time_list_step_{step}.txt",
                )
                mlflow.log_text(
                    str(buf_infos["routes"][route_id]["total_delays_of_all_vehicles"]),
                    f"routes/{route_id}/{mode}_total_delays_list_step_{step}.txt",
                )
                """
                for veh_type in buf_infos["routes"][route_id]["vehicle_type"]:
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/ideal/travel_time",
                        buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                            "ideal"
                        ]["travel_time"],
                        step=step,
                    )
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/real/number_of_vehicles",
                        buf_infos["routes"][route_id]["vehicle_type"][veh_type]["real"][
                            "number_of_vehicles"
                        ],
                        step=step,
                    )
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/real/total_travel_time",
                        float(str(
                            buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                                "real"
                            ]["total_travel_time"]
                        ))
                        if len(
                            buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                                "real"
                            ]["vehicle_id"]
                        )
                        != 0
                        else 0,
                        step=step,
                    )
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/real/average_travel_time",
                        buf_infos["routes"][route_id]["vehicle_type"][veh_type]["real"][
                            "average_travel_time"
                        ],
                        step=step,
                    )
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/real/delays/total",
                        float(str(
                            buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                                "real"
                            ]["delays"]["total"]
                        ))
                        if len(
                            buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                                "real"
                            ]["vehicle_id"]
                        )
                        != 0
                        else 0,
                        step=step,
                    )
                    mlflow.log_metric(
                        f"metrics/{mode}/routes/{route_id}/vehicle_type/{veh_type}/real/delays/average",
                        buf_infos["routes"][route_id]["vehicle_type"][veh_type]["real"][
                            "delays"
                        ]["average"]
                        if len(
                            buf_infos["routes"][route_id]["vehicle_type"][veh_type][
                                "real"
                            ]["vehicle_id"]
                        )
                        != 0
                        else 0,
                        step=step,
                    )
                """
