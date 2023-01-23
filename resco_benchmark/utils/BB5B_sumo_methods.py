import codecs
from pathlib import Path

import traci
import yaml as yaml


def get_the_routes_info():
    routes_info = {}
    # load available routes in the environment
    with codecs.open(
        str(Path(__file__).parent.parent.joinpath("environments", "BB5B", "routes", "default.yaml")),
        "r",
        "utf-8",
    ) as file:
        routes_description = yaml.safe_load(file)
    # load the available types of vehicles in the environment
    with codecs.open(
        str(
            Path(__file__).parent.parent.joinpath("environments", "BB5B", "vehicles", "default.yaml")
        ),
        "r",
        "utf-8",
    ) as file:
        vehicles_description = yaml.safe_load(file)

    for route_id in routes_description.keys():
        routes_info[route_id] = {
            "edges": routes_description[route_id]["config"]["edges"],
            "length": traci.simulation.findRoute(
                routes_description[route_id]["config"]["edges"].split()[0],
                routes_description[route_id]["config"]["edges"].split()[-1],
            ).length,
            "total_number_of_all_vehicles_generated-ThruPut_Scheduled": 0,
            "total_number_of_all_vehicles_completing_journey-ThruPut_Actual": 0,
            "throughput_of_the_route-ThruPut_Idx": 0,
            "total_travel_time_of_all_vehicles": [],
            "total_ideal_travel_time_of_all_vehicles": [],
            "total_average_travel_time_of_all_vehicles": 0,
            "total_delays_of_all_vehicles": [],
            "total_average_delays_of_all_vehicles-Delay_Idx_Average": 0,
            "Delay_Idx_StDev": 0,
            "vehicle_type": {
                veh_type: {
                    "ideal": {
                        "travel_time": traci.simulation.findRoute(
                            routes_description[route_id]["config"]["edges"].split()[0],
                            routes_description[route_id]["config"]["edges"].split()[-1],
                            vType=veh_type,
                        ).length
                        / vehicles_description[veh_type]["config"]["maxSpeed"],
                        "max_speed": vehicles_description[veh_type]["config"]["maxSpeed"],
                    },
                    "real": {
                        "vehicle_id": [],
                        "number_of_vehicles": 0,
                        "total_travel_time": [],
                        "average_travel_time": 0,
                        "delays": {"total": [], "average": []},
                    },
                }
                for veh_type in vehicles_description.keys()
            },
        }
    return routes_info
