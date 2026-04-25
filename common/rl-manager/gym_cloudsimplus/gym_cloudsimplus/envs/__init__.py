from .base import CloudSimBaseEnv
from .vm_management import VmManagementEnv  # noqa: F401
from .job_placement import JobPlacementEnv  # noqa: F401

import gymnasium

# Register new domain-named environments
gymnasium.register(
    id="VmManagement-v0",
    entry_point=VmManagementEnv,
)

gymnasium.register(
    id="JobPlacement-v0",
    entry_point=JobPlacementEnv,
)

# Backwards-compatible aliases (legacy registrations — old file-based classes)
gymnasium.register(
    id="SingleDC-v0",
    entry_point="gym_cloudsimplus.gym_cloudsimplus.envs.singledc:GrpcSingleDC",
)

gymnasium.register(
    id="MultiDC-v0",
    entry_point="gym_cloudsimplus.gym_cloudsimplus.envs.multidc:GrpcMultiDC",
)