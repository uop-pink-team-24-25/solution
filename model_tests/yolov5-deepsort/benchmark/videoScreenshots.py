import subprocess
import os
from datetime import timedelta

input_video = "Data.mp4"          # Input video file
output_dir = "screenshots"          # Screenshot folder
interval_sec = 30                   # Interval between screenshots
output_prefix = "screenshot"        # Prefix for output files

os.makedirs(output_dir, exist_ok=True)

cmd = [
    'ffprobe', '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    input_video
]
duration = float(subprocess.check_output(cmd))  # Get duration in seconds

num_shots = int(duration // interval_sec) + 1  # Calculate number of screenshots

for i in range(num_shots): 
    timestamp = i * interval_sec
    time_str = str(timedelta(seconds=timestamp))
    
    cmd = [
        'ffmpeg', 
        '-ss', time_str,           # Timestamp
        '-i', input_video,         # Input video
        '-vframes', '1',           # Capture one frame
        '-q:v', '2',               # Quality level
        '-y',                      # Overwrite existing files
        f'{output_dir}/{output_prefix}_{time_str.replace(":","-")}.jpg' # Output file
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Run command
    
    print(f"Captured: {time_str}") 

print("Finished capturing screenshots!") 