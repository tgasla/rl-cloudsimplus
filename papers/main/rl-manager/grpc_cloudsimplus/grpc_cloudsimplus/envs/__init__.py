from .grpc_singledc import GrpcSingleDC

import gymnasium

# Register the gymnasium environment
gymnasium.register(
    id="GrpcSingleDC-v0",
    entry_point=GrpcSingleDC,
)
