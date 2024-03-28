from datetime import datetime

def get_filename_id(algorithm_str, timesteps):
    filename_id = (
        f"{algorithm_str}_"
        f"{millify(timesteps)}_"
        f"{datetime_to_str()}"
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