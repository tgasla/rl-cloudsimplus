from .base import CloudSimBaseEnv
from .vm_management import VmManagementEnv  # noqa: F401
from .job_placement import JobPlacementEnv  # noqa: F401

import gymnasium

gymnasium.register(
    id="VmManagement-v0",
    entry_point=VmManagementEnv,
)

gymnasium.register(
    id="JobPlacement-v0",
    entry_point=JobPlacementEnv,
)