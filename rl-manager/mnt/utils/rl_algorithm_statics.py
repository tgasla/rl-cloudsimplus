ALGORITHMS_WITH_ENT_COEF = [
    "PPO",
    "MaskablePPO",
    "RecurrentPPO",
    "A2C",
    "SAC",
    "CrossQ",
    "TQC",
]
ALGORITHMS_WITH_ACTION_NOISE = ["TD3", "DDPG", "DQN", "QR-DQN", "SAC", "CrossQ", "TQC"]
ALGORITHMS_WITH_N_STEPS = ["PPO", "MaskablePPO", "RecurrentPPO", "A2C", "TRPO"]

DEFAULT_LEARNING_RATES = {
    "PPO": 3e-4,
    "MaskablePPO": 3e-4,
    "RecurrentPPO": 3e-4,
    "A2C": 7e-4,
    "SAC": 3e-4,
    "TD3": 3e-4,
    "DQN": 1e-4,
    "QR-DQN": 1e-4,
    "TQC": 3e-4,
    "CrossQ": 3e-4,
}
