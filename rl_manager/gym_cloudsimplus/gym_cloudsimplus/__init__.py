from gymnasium.envs.registration import register

register(
    id='SingleDCAppEnv-v0',
    entry_point='gym_cloudsimplus.envs:SingleDCAppEnv',
)