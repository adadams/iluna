from dataclasses import dataclass
from typing import TypeAlias

from basic_types import NonnegativeValue, PositiveValue

"""
TransitOrbit is used for the transit fitting, but it is an incomplete description of the system.
Versus the canonical eclipsing structure in jaxoplanet which is a combination of .starry Surface, orbit, Ylm, and .orbits.keplerian Central, Body.
"""

Duration: TypeAlias = NonnegativeValue


@dataclass
class TransitOrbitParameters:
    period: Duration
    duration: Duration
    time_transit: Duration
    impact_param: NonnegativeValue
    radius_ratio: PositiveValue
