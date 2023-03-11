from __future__ import annotations

import codecs
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)
from xml.etree import ElementTree

import numpy as np
import xmltodict

from resco_benchmark.utils.mytl.generators.base import Generator
from resco_benchmark.utils.mytl.typing import SumoRouteSetting, SumoVehicleSetting

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T", bound="RoutesGenerator")


class RoutesGenerator(Generator):
    """SUMO routes configuration files generator.

    Parameters
    ----------
    path_to_save_rou : str or pathlib.Path
        Path where the created XML file will be saved.

    routes : dict_like
        Mapping where key is an object id and value is the SUMO
        setting for routes.

    vehicles : dict_like
        Mapping where key is an object id and value is the SUMO
        setting for vehicles.

    n_steps : int
        Number of steps in the environment.

    force : bool, optional (default=False)
        - If True, existing file under the `path_to_save` indicated above will
          be overwritten.
        - If False, unless the `path_to_save` points to an unallocated location,
          exception will be thrown.

    begin : int
       The beginning of the duration of the specified time period.

    end : int
        The end of the specified period of time.

    total_time : int
        Total duration of the generated file

    Attributes
    ----------
    root

    data : xml.etree.ElementTree.Element
        Root node in the XML tree representation from which the other
        sub-elements will branch.

    """

    def __init__(
            self,
            path_to_save_rou: Union[str, Path],
            *,
            routes: Mapping[str, SumoRouteSetting],
            vehicles: Mapping[str, SumoVehicleSetting],
            force: bool = False,
            begin: int,
            end: int,
            total_time: int,
    ):
        super().__init__(path=path_to_save_rou, force=force, begin=begin, end=end, total_time=total_time)

        self.routes = routes
        self.vehicles = vehicles

    @property
    def root(self) -> str:
        """Name of the root tag onto which other tags are created."""
        return "routes"

    def make(self, path_from_original_rou: Path) -> T:
        """Write routes' subelements to the XML tree.

        Returns
        -------
        self : object
            Returns the instance itself.

        """

        # load all parameters of available vehicles (e.g. type, length, maxSpeed)
        # and the probability of the appearance of a vehicle type
        vehicles, vehicles_probas = self._parse(elements=self.vehicles)
        self._set(name="vType", elements=vehicles)

        # load all id of available routes and their edges
        routes, routes_probas = self._parse(elements=self.routes)
        self._set(name="route", elements=routes)

        # load all available vehicle types
        vehicles_types = [veh_type for veh_type in self.vehicles.keys()]

        times_of_the_appearance_of_the_vehicle_from_file_deltas = defaultdict(list)
        times_of_the_appearance_of_the_vehicle_from_file = defaultdict(list)
        generated_vehicles_on_routes = dict()

        # load the original traffic data from a file
        with codecs.open(str(path_from_original_rou), mode="r", encoding="utf-8") as file:
            xml = xmltodict.parse(file.read())

        # load the time of the vehicle appearance in the environment from the XML file
        for vehicle in xml['routes']['vehicle']:
            times_of_the_appearance_of_the_vehicle_from_file[vehicle["@route"]].append(int(vehicle["@depart"]))

        # calculate deltas between the time of the vehicle appearance in the environment from the XML file
        for key, values in times_of_the_appearance_of_the_vehicle_from_file.items():
            times_of_the_appearance_of_the_vehicle_from_file_deltas[key].extend(list(np.diff(sorted(values))))

        # randomly generating vehicles on chosen routes within a specified time period
        for route, deltas in times_of_the_appearance_of_the_vehicle_from_file_deltas.items():
            if len(deltas) != 0:
                deltas = np.asarray(deltas)
                mean = deltas.mean()
                size = (self.end - self.begin) // mean
                # generate of vehicle appearance times on a chosen route according to the Poisson distribution
                time_the_vehicle_appeared_on_the_route = np.cumsum(np.around(np.random.exponential(scale=deltas.mean(), size=int(size * 1.5))).astype(int)) if len(times_of_the_appearance_of_the_vehicle_from_file_deltas[route]) != 0 else []
                # move all items in the list by 7 hours, so 25200 s
                time_the_vehicle_appeared_on_the_route = [x + self.begin for x in time_the_vehicle_appeared_on_the_route]
                # take into consideration only those appearance times that are within the specified time period
                time_the_vehicle_appeared_on_the_route = np.array([element for element in time_the_vehicle_appeared_on_the_route if element <= self.end])
                # random choice of a vehicle types on chosen route
                generated_vehicle_types = np.random.choice(vehicles_types, size=len(time_the_vehicle_appeared_on_the_route), replace=True, p=vehicles_probas)
                # save the generated vehicles in a dictionary where the route id is the key
                generated_vehicles_on_routes[route] = {"time_the_vehicle_appeared_on_the_route": time_the_vehicle_appeared_on_the_route,
                                                       "vehicles_type": generated_vehicle_types
                                                       }

        idx = 0
        # save the generated data in a training file XML
        for step in range(self.begin, self.end + 1):
            for route in generated_vehicles_on_routes.keys():
                if step in generated_vehicles_on_routes[route]['time_the_vehicle_appeared_on_the_route']:
                    elem = ElementTree.SubElement(self.data, "vehicle")
                    elem.set("id", route + f"_{idx}")
                    elem.set("type", generated_vehicles_on_routes[route]['vehicles_type'][list(generated_vehicles_on_routes[route]['time_the_vehicle_appeared_on_the_route']).index(step)])
                    elem.set("route", route)
                    elem.set("depart", str(step))
                    elem.set("speedDev", "0")
                    idx += 1

        return self

    def _set(self, name: str, elements: List[Mapping[str, str]]) -> None:
        for element in elements:
            elem = ElementTree.SubElement(self.data, name)
            for attribute, value in element.items():
                elem.set(attribute, value)

    @staticmethod
    def _parse(
            elements: Mapping[str, Union[SumoRouteSetting, SumoVehicleSetting]],
    ) -> Tuple[List[Mapping[str, str]], List[float]]:
        elements_parsed = []
        weights = []
        for element_id, setting in elements.items():
            config = {
                **{"id": element_id},
                **{
                    key: str(value) for key, value in setting["config"].items()
                },
            }
            elements_parsed.append(config)
            weights.append(float(setting["weight"]))

        probas = weights / np.linalg.norm(weights, ord=1)

        return elements_parsed, probas
