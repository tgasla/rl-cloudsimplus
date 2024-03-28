from datetime import datetime

def get_filename_id(env_id, algorithm_str, timesteps, retraining_env=None):
    filename_id = (
        f"p{env_id}"
        f"_{algorithm_str}"
        f"_{millify(timesteps)}"
        # f"{datetime_to_str()}"
    )
    if retraining_env is not None:
        filename_id = (
            f"{filename_id}"
            f"_r{retraining_env}"
        )
    return filename_id

def datetime_to_str():
    return datetime.now().strftime("%m%d%y_%H%M%S")

def millify(num):
    num = float(f"{num:.3f}")
    magnitude = 0
    suffix = ['', 'K', 'M', 'B', 'T']
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000
    return f"{num:.0f}" + suffix[magnitude]