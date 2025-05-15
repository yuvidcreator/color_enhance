import os
from datetime import datetime


def get_time_stamp() -> str:
    now = datetime.now()
    # Format it as a string: YYYY-MM-DD HH:MM:SS
    timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return timestamp_str

def get_ouput_image_path(output_dir):
    # Set ouput directory path
    # output_dir = "output_data"
    time_stamp = get_time_stamp()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{time_stamp}", exist_ok=True)
    img_dest = f"{output_dir}/{time_stamp}"
    return img_dest, time_stamp