from .singledc import GrpcSingleDC
from .multidc import GrpcMultiDC

import gymnasium

# Register the gymnasium environments
gymnasium.register(
    id="SingleDC-v0",
    entry_point=GrpcSingleDC,
)

gymnasium.register(
    id="MultiDC-v0",
    entry_point=GrpcMultiDC,
)
