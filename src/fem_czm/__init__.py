"""FEM/CZM fracture-fatigue constitutive core.

Version 0.1 establishes the shared monotonic/fatigue physics.  The source
population is reusable: there is no available-site counter, source exhaustion,
or crack-advance source refresh.
"""

from .parameters import MATERIALS, get_material
from .front import FrontConfig, CrackFront
from .fatigue import FatigueConfig, FatigueIntegrator

__all__ = [
    "MATERIALS",
    "get_material",
    "FrontConfig",
    "CrackFront",
    "FatigueConfig",
    "FatigueIntegrator",
]

__version__ = "0.1.0"
