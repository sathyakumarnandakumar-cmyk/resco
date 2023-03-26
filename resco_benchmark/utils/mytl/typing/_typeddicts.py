from __future__ import annotations

from typing import Union

from typing_extensions import TypedDict


class CodecsOpenParams(TypedDict, total=False):
    mode: str
    encoding: str
    errors: str
    buffering: int


class SumoRouteConfig(TypedDict, total=False):
    edges: str


class SumoRouteSetting(TypedDict):
    weight: Union[float, int]
    config: SumoRouteConfig


class SumoVehicleConfig(TypedDict, total=False):
    accel: Union[float, int]
    decel: Union[float, int]
    sigma: Union[float, int]
    length: Union[float, int]
    minGap: Union[float, int]
    maxSpeed: Union[float, int]
    vClass: str
    guiShape: str


class SumoVehicleSetting(TypedDict):
    weight: Union[float, int]
    config: SumoVehicleConfig




