from enum import Enum
from dataclasses import dataclass
from typing import Literal
import neptune
from neptune import Run
import logging
import difflib
import pprint

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resco_benchmark")


def compare_dicts(d1, d2):
    return "\n" + "\n".join(
        difflib.ndiff(pprint.pformat(d1).splitlines(), pprint.pformat(d2).splitlines())
    )


BB5B_metrics = [
    "count_of_all_vehicles_in_simulation",
    "number_of_halting_vehicles_for_the_last_time_step_on_the_incoming_lanes",
    "number_of_all_halting_vehicles_for_the_last_time_step_in_simulation",
    "waiting_time_all_vehicles_for_the_last_time_step_in_simulation",
    "total_waiting_time_on_the_incoming_lanes_in_episode",
    "total_waiting_time_on_the_incoming_lanes_in_episode2",
    "count_of_vehicles_completing_journey",
    "total_time_of_journey",
    "average_time_of_journey",
    "total_sum_delays_of_all_vehicles_from_all_routes",
    "total_average_delays_of_all_vehicles_from_all_routes",
    "total_average_delays_of_all_vehicles_completing_journey_and_not_completing_journey",
    "total_average_delays_real_times_by_ideal_times",
    "total_average_delays_with_weights",
    "current_number_of_vehicles",
    "current_average_delays_of_all_vehicles_in_simulation",
    "calculate_average_delta_of_delays_after_action",
    "number_of_vehicles_that_passed_through_the_intersections_in_last_steps",
]

BB5B_routes_metrics = [
    "total_number_of_all_vehicles_generated-ThruPut_Scheduled",
    "total_number_of_all_vehicles_completing_journey-ThruPut_Actual",
    "throughput_of_the_route-ThruPut_Idx",
    "total_travel_time_of_all_vehicles",
    "total_average_travel_time_of_all_vehicles",
    "total_delays_of_all_vehicles",
    "total_average_delays_of_all_vehicles-Delay_Idx_Average",
    "total_average_delays_of_all_vehicles_with_weights",
    "total_ideal_travel_time_of_all_vehicles",
]


class ComparisonMode(Enum):
    NONE = 0
    LAST_VALUE = 1
    ALL_VALUES = 2


@dataclass
class ComparisonConfiguration:
    general: ComparisonMode
    routes: ComparisonMode


def _access_nested(d: dict, key: str):
    keys = key.split("/")
    for k in keys:
        d = d[k]
    return d


def compare_BB5B_runs_results_for_single_mode(
    first: Run,
    second: Run,
    config: ComparisonConfiguration,
    mode: Literal["training", "validation"],
):
    logger.info("Comparing mode: %s", mode)
    logger.info("General comparison: %s", config.general.name)

    if config.general == ComparisonMode.ALL_VALUES:
        for metric in BB5B_metrics:
            key = f"metrics/{mode}/{metric}"
            f_m = first[key].fetch_values()
            s_m = second[key].fetch_values()
            try:
                if (f_m["value"] != s_m["value"]).any():
                    print("Different results for", metric)
            except ValueError:
                logger.error(f"Error comparing '{metric}'", exc_info=True)

    logger.info("Routes comparison: %s", config.routes.name)

    if config.routes == ComparisonMode.ALL_VALUES:
        key_prefix = f"metrics/{mode}/routes"
        routes = _access_nested(first.get_structure(), key_prefix).keys()
        for route in routes:
            for metric in BB5B_routes_metrics:
                key = f"{key_prefix}/{route}/{metric}"
                f_m = first[key].fetch_values()
                s_m = second[key].fetch_values()
                try:
                    if (f_m["value"] != s_m["value"]).any():
                        print("Different results for", metric)
                except ValueError:
                    logger.error(f"Error comparing '{metric}'", exc_info=True)


def compare_BB5B_runs_results(
    first: Run,
    second: Run,
    *,
    training: ComparisonConfiguration,
    validation: ComparisonConfiguration,
):
    f_params = first["parameters"].fetch()
    s_params = second["parameters"].fetch()

    if f_params != s_params:
        logger.warning("Different parameters. The results may differ.")
        print(compare_dicts(f_params, s_params))

    compare_BB5B_runs_results_for_single_mode(first, second, training, "training")
    compare_BB5B_runs_results_for_single_mode(first, second, validation, "validation")


def test_BB5B_IDQN_3eps():
    model_run = neptune.init_run(with_id="MAL-1264", mode="read-only")
    model_run2 = neptune.init_run(with_id="MAL-1264", mode="read-only")
    compare_BB5B_runs_results(
        model_run,
        model_run2,
        training=ComparisonConfiguration(
            general=ComparisonMode.ALL_VALUES, routes=ComparisonMode.ALL_VALUES
        ),
        validation=ComparisonConfiguration(
            general=ComparisonMode.ALL_VALUES, routes=ComparisonMode.ALL_VALUES
        ),
    )


if __name__ == "__main__":
    test_BB5B_IDQN_3eps()
