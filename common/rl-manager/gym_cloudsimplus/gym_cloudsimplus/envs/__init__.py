from .singledc import GrpcSingleDC as SingleDC
from .multidc import GrpcMultiDC as MultiDC

import gymnasium

# Register the gymnasium environments
gymnasium.register(
    id="SingleDC-v0",
    entry_point=SingleDC,
)

gymnasium.register(
    id="MultiDC-v0",
    entry_point=MultiDC,
)
