from gymnasium.envs.registration import register

register(
    id='SmallDC-v0',
    entry_point='gym_cloudsimplus.envs:SmallDC',
)

register(
    id='LargeDC-v0',
    entry_point='gym_cloudsimplus.envs:LargeDC',
)