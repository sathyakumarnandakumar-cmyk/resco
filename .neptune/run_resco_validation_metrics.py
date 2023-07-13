import csv
import time

import neptune.new as neptune

start_time = time.time()

# define the list of Neptune project IDs
project_id_list = ["MAL2-342", "MAL2-341", "MAL2-326"]

data_from_neptune_dict = {}

metrics_from_neptune = [
    "number_of_validation_episodes",
    "total_average_delays_of_all_vehicles_from_all_routes",
    "total_average_delays_real_times_by_ideal_times",
    "total_sum_delays_of_all_vehicles_from_all_routes",
    "count_of_vehicles_completing_journey",
    "total_waiting_time_on_the_incoming_lanes_in_episode",
    "total_waiting_time_on_the_incoming_lanes_in_episode2",
    "total_time_of_journey",
    "average_time_of_journey",
    "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey",
]

filename_save = "_".join(project_id_list)

for project_id in project_id_list:
    run = neptune.init_run(
        project="pgora/Malaysia2", api_token=None, with_id=project_id
    )

    algorithm_name = run["parameters/algorithm"].fetch()

    total_average_delays_of_all_vehicles_from_all_routes = (
        run["metrics/validation/total_average_delays_of_all_vehicles_from_all_routes"]
        .fetch_values()
        .value.values
    )
    total_average_delays_real_times_by_ideal_times = (
        run["metrics/validation/total_average_delays_real_times_by_ideal_times"]
        .fetch_values()
        .value.values
    )
    total_sum_delays_of_all_vehicles_from_all_routes = (
        run["metrics/validation/total_sum_delays_of_all_vehicles_from_all_routes"]
        .fetch_values()
        .value.values
    )
    count_of_vehicles_completing_journey = (
        run["metrics/validation/count_of_vehicles_completing_journey"]
        .fetch_values()
        .value.values
    )
    total_waiting_time_on_the_incoming_lanes_in_episode = (
        run["metrics/validation/total_waiting_time_on_the_incoming_lanes_in_episode"]
        .fetch_values()
        .value.values
    )
    total_waiting_time_on_the_incoming_lanes_in_episode2 = (
        run["metrics/validation/total_waiting_time_on_the_incoming_lanes_in_episode2"]
        .fetch_values()
        .value.values
    )
    total_time_of_journey = (
        run["metrics/validation/total_time_of_journey"].fetch_values().value.values
    )
    average_time_of_journey = (
        run["metrics/validation/average_time_of_journey"].fetch_values().value.values
    )
    total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey = (
        run[
            "metrics/validation/total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey"
        ]
        .fetch_values()
        .value.values
    )

    data_from_neptune_dict[project_id] = {
        "algorithm_name": algorithm_name,
        "number_of_validation_episodes": len(total_time_of_journey),
        "total_average_delays_of_all_vehicles_from_all_routes_first": round(
            total_average_delays_of_all_vehicles_from_all_routes[0], 5
        ),
        "total_average_delays_of_all_vehicles_from_all_routes_last": round(
            total_average_delays_of_all_vehicles_from_all_routes[-1], 5
        ),
        "total_average_delays_of_all_vehicles_from_all_routes_min": round(
            min(total_average_delays_of_all_vehicles_from_all_routes), 5
        ),
        "total_average_delays_real_times_by_ideal_times_first": round(
            total_average_delays_real_times_by_ideal_times[0], 5
        ),
        "total_average_delays_real_times_by_ideal_times_last": round(
            total_average_delays_real_times_by_ideal_times[-1], 5
        ),
        "total_average_delays_real_times_by_ideal_times_min": round(
            min(total_average_delays_real_times_by_ideal_times), 5
        ),
        "total_sum_delays_of_all_vehicles_from_all_routes_first": round(
            total_sum_delays_of_all_vehicles_from_all_routes[0], 5
        ),
        "total_sum_delays_of_all_vehicles_from_all_routes_last": round(
            total_sum_delays_of_all_vehicles_from_all_routes[-1], 5
        ),
        "total_sum_delays_of_all_vehicles_from_all_routes_min": round(
            min(total_sum_delays_of_all_vehicles_from_all_routes), 5
        ),
        "count_of_vehicles_completing_journey_first": round(
            count_of_vehicles_completing_journey[0], 5
        ),
        "count_of_vehicles_completing_journey_last": round(
            count_of_vehicles_completing_journey[-1], 5
        ),
        "count_of_vehicles_completing_journey_max": round(
            max(count_of_vehicles_completing_journey), 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode_first": round(
            total_waiting_time_on_the_incoming_lanes_in_episode[0], 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode_last": round(
            total_waiting_time_on_the_incoming_lanes_in_episode[-1], 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode_min": round(
            min(total_waiting_time_on_the_incoming_lanes_in_episode), 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode2_first": round(
            total_waiting_time_on_the_incoming_lanes_in_episode2[0], 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode2_last": round(
            total_waiting_time_on_the_incoming_lanes_in_episode2[-1], 5
        ),
        "total_waiting_time_on_the_incoming_lanes_in_episode2_min": round(
            min(total_waiting_time_on_the_incoming_lanes_in_episode2), 5
        ),
        "total_time_of_journey_first": round(total_time_of_journey[0], 5),
        "total_time_of_journey_last": round(total_time_of_journey[-1], 5),
        "total_time_of_journey_min": round(min(total_time_of_journey), 5),
        "average_time_of_journey_first": round(average_time_of_journey[0], 5),
        "average_time_of_journey_last": round(average_time_of_journey[-1], 5),
        "average_time_of_journey_min": round(min(average_time_of_journey), 5),
        "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_first": round(
            total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey[
                0
            ],
            5,
        ),
        "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_last": round(
            total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey[
                -1
            ],
            5,
        ),
        "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_min": round(
            min(
                total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey
            ),
            5,
        ),
    }

print("--- %s seconds ---" % (time.time() - start_time))

try:
    with open("resco_validation_metrics/" + filename_save + ".csv", "w") as file:
        writer = csv.writer(
            file,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writerow(
            [
                "algorithm_name",
                "number_of_validation_episodes",
                "total_average_delays_of_all_vehicles_from_all_routes_first",
                "total_average_delays_of_all_vehicles_from_all_routes_last",
                "total_average_delays_of_all_vehicles_from_all_routes_min",
                "total_average_delays_real_times_by_ideal_times_first",
                "total_average_delays_real_times_by_ideal_times_last",
                "total_average_delays_real_times_by_ideal_times_min",
                "total_sum_delays_of_all_vehicles_from_all_routes_first",
                "total_sum_delays_of_all_vehicles_from_all_routes_last",
                "total_sum_delays_of_all_vehicles_from_all_routes_min",
                "count_of_vehicles_completing_journey_first",
                "count_of_vehicles_completing_journey_last",
                "count_of_vehicles_completing_journey_max",
                "total_waiting_time_on_the_incoming_lanes_in_episode_first",
                "total_waiting_time_on_the_incoming_lanes_in_episode_last",
                "total_waiting_time_on_the_incoming_lanes_in_episode_min",
                "total_waiting_time_on_the_incoming_lanes_in_episode2_first",
                "total_waiting_time_on_the_incoming_lanes_in_episode2_last",
                "total_waiting_time_on_the_incoming_lanes_in_episode2_min",
                "total_time_of_journey_first",
                "total_time_of_journey_last",
                "total_time_of_journey_min",
                "average_time_of_journey_first",
                "average_time_of_journey_last",
                "average_time_of_journey_min",
                "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_first",
                "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_last",
                "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_min",
            ]
        )

        row = []
        for project_id in data_from_neptune_dict:
            row = [
                data_from_neptune_dict[project_id]["number_of_validation_episodes"],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_from_all_routes_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_from_all_routes_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_from_all_routes_min"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_real_times_by_ideal_times_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_real_times_by_ideal_times_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_real_times_by_ideal_times_min"
                ],
                data_from_neptune_dict[project_id][
                    "total_sum_delays_of_all_vehicles_from_all_routes_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_sum_delays_of_all_vehicles_from_all_routes_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_sum_delays_of_all_vehicles_from_all_routes_min"
                ],
                data_from_neptune_dict[project_id][
                    "count_of_vehicles_completing_journey_first"
                ],
                data_from_neptune_dict[project_id][
                    "count_of_vehicles_completing_journey_last"
                ],
                data_from_neptune_dict[project_id][
                    "count_of_vehicles_completing_journey_max"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode_min"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode2_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode2_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_waiting_time_on_the_incoming_lanes_in_episode2_min"
                ],
                data_from_neptune_dict[project_id]["total_time_of_journey_first"],
                data_from_neptune_dict[project_id]["total_time_of_journey_last"],
                data_from_neptune_dict[project_id]["total_time_of_journey_min"],
                data_from_neptune_dict[project_id]["average_time_of_journey_first"],
                data_from_neptune_dict[project_id]["average_time_of_journey_last"],
                data_from_neptune_dict[project_id]["average_time_of_journey_min"],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_first"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_last"
                ],
                data_from_neptune_dict[project_id][
                    "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey_min"
                ],
            ]
            writer.writerow(row)
except:
    print("Failed to write Delay Idx")

print("Finish")
