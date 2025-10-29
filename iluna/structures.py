from typing import TypedDict

from iluna.basic_types import (
    DurationValue,
    FluxValue,
    MassValue,
    NonnegativeValue,
    NormalizedValue,
    RadiusValue,
)


class TransitFitParameters(TypedDict):
    period: DurationValue
    time_transit: float
    MWD: MassValue
    RWD: RadiusValue
    RBD: RadiusValue
    impact_param: NonnegativeValue
    baseline_WD_brightness: FluxValue
    baseline_BD_nightside_brightness: FluxValue
    q0: NormalizedValue
    q1: NormalizedValue


class NonEclipseFitParameters(TypedDict):
    period: DurationValue
    time_transit: float
    MWD: MassValue
    RWD: RadiusValue
    RBD: RadiusValue
    impact_param: NonnegativeValue
    baseline_WD_brightness: FluxValue
    baseline_BD_nightside_brightness: FluxValue
    q0: NormalizedValue
    q1: NormalizedValue
