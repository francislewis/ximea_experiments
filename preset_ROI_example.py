from ximea_experiments.experiment import experiment
from ximea import xiapi

# This will automatically grab the first connected camera's SN
cam = xiapi.Camera()
cam.open_device()
serial_number = cam.get_device_sn()
cam.close_device()
cam_sn = serial_number.decode('utf-8')  # Can specify a specific serial number here instead

# USER DEFINED SETTINGS
duration = 1  # Time in seconds to record for
framerate = 500  # In FPS
exposure = 1.8  # In ms
gain = 19  # In dB
parent_folder = 'test_set_roi'
record_times = [0, 1, 2, 3]  # Times in minutes at which to record data
roi = {'width': 60, 'height': 50, 'x_offset': 200, 'y_offset': 300}

user_settings = {'cam_sn': cam_sn, 'duration': duration, 'framerate': framerate, 'exposure': exposure, 'gain': gain,
                    'parent_folder': parent_folder, 'roi': roi}

run_experiment = experiment(settings=user_settings, record_times=record_times, supply_roi=True)
