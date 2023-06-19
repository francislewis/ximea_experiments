import cv2
import numpy as np
import time
import schedule
import datetime
from ximea import xiapi
import ctypes
import json
import imageio
from typing import List
from dataclasses import dataclass
from threading import Thread
from ctypes import sizeof
from functools import partial
from itertools import starmap
from os import makedirs
from os.path import join

class coordinateSelect():
    """
    A class which displays an image and saves 2 user selected coordinates.
    The first 2 clicks are saved in self.locations
    The window exits on the 3rd click, or if a key is pressed at any time

    Inputs: image_loc (str) - filename/location of image
            type (int) [0,1] - 0, provide str 'image_loc', 1 - provide image data in np.array format

    Usage: test_instance = coordinateSelect('test.png')

    To select a region of interest (ROI), points must be selected in the order top left, bottom right:
    (1,2 below)
    -----------------------------------------
    |                                       |
    |    1:(x1, y1)      w                  |
    |      ------------------------         |
    |      |                      |         |
    |      |         ROI          | h       |
    |      |                      |         |
    |      ------------------------         |
    |                          2:(x2, y2)   |
    |                                       |
    -----------------------------------------
    """

    def __init__(self, image_loc, type):
        # Set variables
        self.image_loc = image_loc
        self.type = type
        self.locations = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.roi_height = 0
        self.roi_width = 0
        self.dimensions = {}  # Used to store the offsets and width/height for use in ROI setting

        if self.type == 0:
            # Read image
            self.img = cv2.imread(self.image_loc, 1)
        elif self.type == 1:
            self.img = image_loc
        else:
            print("Type must be 0 for file or 1 for capture")

        # More variables now image is read
        self.original_width = float(np.shape(self.img)[1])
        self.original_height = float(np.shape(self.img)[0])

        # Create re-sizeable window
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        # Display image
        cv2.imshow('image', self.img)

        # Draw a scale line
        cv2.line(self.img, (int(0 + self.original_width / 5), int(self.original_height / 2)),
                 (int(100 + self.original_width / 5), int(self.original_height / 2)), (0, 255, 0), thickness=10)
        cv2.putText(self.img, '100 px line for scale',
                    (int(0 + self.original_width / 5), int((self.original_height / 2) + 100)), self.font, 1,
                    (0, 255, 0), 2)

        # Set mouse handler
        cv2.setMouseCallback('image', self.__click_event)

        # Exit early if key pressed
        cv2.waitKey(0)

        # Close  window
        cv2.destroyAllWindows()

        # Check order correct and print dimensions
        self.__check()


    def __click_event(self, event, x, y, flags, params):
        """
        Private method only to be used by coordinateSelect instances.
        Used to draw useful graphics to the screen during ROI selection
        """
        # Check for clicks
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:

            # Save first two clicks
            if len(self.locations) < 2:
                self.locations.append([x, y])
            else:
                cv2.destroyAllWindows()
                return 0

            # Display the coordinates on the image window
            cv2.putText(self.img, str(x) + ',' + str(y), (x, y), self.font, 1, (255, 0, 0), 2)
            # Draw rectangle
            cv2.rectangle(self.img, (self.locations[0][0], self.locations[0][1]), (x, y), (0, 0, 255), 2)
            # Display the final dimensions in the middle of each line
            if len(self.locations) > 1:
                cv2.putText(self.img, 'height: ' + str(y - self.locations[0][1]),
                            (int(x + self.original_width * 0.01), int(y - (y - self.locations[0][1]) / 2)), self.font,
                            1, (255, 0, 0), 2)
                cv2.putText(self.img, 'length: ' + str(x - self.locations[0][0]),
                            (int(x - (x - self.locations[0][1]) / 2), int(y + self.original_width * 0.02)), self.font,
                            1, (255, 0, 0), 2)
            cv2.imshow('image', self.img)

        if len(self.locations) == 1:
            if event == cv2.EVENT_MOUSEMOVE:
                # Create a copy so it updates on mouse move
                temp_img = self.img.copy()
                # Draw rectangle
                cv2.rectangle(temp_img, (self.locations[0][0], self.locations[0][1]), (x, y), (0, 0, 255), 2)
                # Print current coordinates
                cv2.putText(temp_img, str(x) + ',' + str(y), (x, y), self.font, 1, (255, 0, 0), 2)
                # Print the current dimensions in the middle of each line
                cv2.putText(temp_img, 'height: ' + str(y - self.locations[0][1]),
                            (int(x + self.original_width * 0.01), int(y - (y - self.locations[0][1]) / 2)), self.font,
                            1, (255, 0, 0), 2)
                cv2.putText(temp_img, 'length: ' + str(x - self.locations[0][0]),
                            (int(x - (x - self.locations[0][1]) / 2), int(y + self.original_width * 0.02)), self.font,
                            1, (255, 0, 0), 2)
                # Display these overlay graphics
                cv2.imshow("image", temp_img)

    def __check(self):
        """
        Private method only to be used by coordinateSelect instances
        Checks if the coordinates have been selected in the correct order as described in __init__
        Raises an error if they haven't been, along with some warnings for larger ROIs
        If it passes the check then the dimensions are saved and printed
        """
        # Check coords picked in correct order
        assert self.locations[0][0] < self.locations[1][0] and self.locations[0][1] < self.locations[1][1], \
            'Have to pick corners in top left corner first, bottom right corner second'

        # Save and show dimensions
        self.roi_width = self.locations[1][0] - self.locations[0][0]
        self.roi_height = self.locations[1][1] - self.locations[0][1]
        self.x_offset = self.locations[0][0]
        self.y_offset = self.locations[0][1]
        print('Width of ROI: ' + str(self.roi_width))
        print('Height of ROI: ' + str(self.roi_height))
        print('X Offset: ' + str(self.x_offset))
        print('Y Offset: ' + str(self.y_offset))

        self.dimensions['width'] = self.roi_width
        self.dimensions['height'] = self.roi_height
        self.dimensions['x_offset'] = self.x_offset
        self.dimensions['y_offset'] = self.y_offset

        if self.roi_width > 150 or self.roi_height > 150:
            print('ROI is large (length > 150px), analysis may be slow')


class experiment:
    def __init__(self, settings={}, record_times=[], supply_roi=False):
        if not supply_roi:
            connection = camConnect(settings)  # Create camera object with user defined settings
            connection.captureImg()  # Capture single image
            choose_coords = coordinateSelect(connection.data, 1)  # Choose ROI from camera image
            connection.addROI(choose_coords.dimensions)  # Add saved ROI to camera settings

            # Schedule the tasks
            scheduler(record_times, connection.captureVideo, args=record_times, iterate=True)

            while True:
                # Check schedule every 1 sec to avoid using too many CPU cycles
                schedule.run_pending()
                time.sleep(1)

                # TODO: Ensure once all measurements are complete, this loop exits

        if supply_roi:
            connection = camConnect(settings)  # Create camera object with user defined settings

            # Schedule the tasks
            scheduler(record_times, connection.captureVideo, args=record_times, iterate=True)

            while True:
                # Check schedule every 1 sec to avoid using too many CPU cycles
                schedule.run_pending()
                time.sleep(1)

def scheduler(times, function_to_call, args={}, iterate=False):
    """
    Inputs:
        times: array of ints, representing times in minutes from execution at which measurements should be made
        function_to_call: function or method must be pre-defined and be passed without parentheses
        args: if Iterate=False, dict of arguments that function_to_call takes
        iterate: if True, will iterate over the arguments held in args
    """
    # Scheduler set up takes too long to run function at time = 0
    if 0 in times:
        if iterate:
            function_to_call(args.pop(0))
        else:
            function_to_call(args)
    for t in times:
        time_hr = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).hour)
        time_min = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).minute)
        time_sec = str((datetime.datetime.now() + datetime.timedelta(minutes=t)).second)

        time_string = ''  # String to store the scheduled times

        for time_part in [time_hr, time_min, time_sec]:
            if len(time_part) == 1:
                # Add zero padding on left to ensure time is HH:MM:SS format when there are single digits
                time_part = time_part.zfill(2)
                time_string += (time_part + ':')
            else:
                time_string += (time_part + ':')

        # Remove the last ':' here instead of extra loop nesting
        time_string = time_string[:-1]

        # Add to schedule
        if iterate:
            schedule.every().day.at(time_string).do(lambda: function_to_call(args.pop(0)))
        else:
            schedule.every().day.at(time_string).do(function_to_call, **args)


class camConnect():
    def __init__(self, *setting_args):
        """
        Connect to Ximea camera and record image/video
        """
        self.settings = setting_args[0]

    def captureImg(self):
        """
        Capture image from first connected camera
        This is currently only used for the coordinate selection so no user settings are applied
        Could be nice to apply the user settings here too but currently seems somewhat unnecessary
        """
        # Create instance for camera
        self.cam = xiapi.Camera()

        # Start Communication
        print('Opening camera...')
        self.cam.open_device()
        print(self.settings)
        # Apply Settings
        self.cam.set_exposure(self.settings['exposure'] / 0.001)
        self.cam.set_gain(self.settings['gain'])

        # Store image data and metadata
        self.img = xiapi.Image()

        # Start data acquisition
        print('Starting data acquisition...')
        self.cam.start_acquisition()

        # Capture image
        self.cam.get_image(self.img)

        # Store data of final image
        self.data = self.img.get_image_data_numpy()

        # Stop data acquisition
        print('Stopping acquisition...')
        self.cam.stop_acquisition()

        # Stop communication
        self.cam.close_device()

    def captureVideo(self, min_stamp):
        self.settings['min_stamp'] = min_stamp
        run_recording(**self.settings)

    def addROI(self, roi={}):
        self.settings['roi'] = roi

"""
This code is modified from https://github.com/kaspervn/xirec
I've removed some features which aren't needed here and turned it into a single script instead of CLI

It could definitely be cleaned up futher and I think putting everything into classes could help in the long run
I also removed some functionality allowing multiple cameras, will probably have ot look at xirec if this is needed
"""

###################### Recorder Functions ######################

@dataclass
class RecordingBuffers:
    video_buffer: ctypes.Array
    meta_buffer: ctypes.Array


def probe_memory_requirements(cam: xiapi.Camera):
    img = xiapi.Image()
    cam.start_acquisition()
    cam.get_image(img)
    frame_data_size = img.get_bytes_per_pixel() * (img.width + img.padding_x) * img.height
    cam.stop_acquisition()

    return frame_data_size


def allocate_recording_buffers(frame_size, no_frames):
    video_buffer = ((ctypes.c_char * frame_size) * no_frames)()
    meta_buffer = (xiapi.Image * no_frames)()

    return RecordingBuffers(video_buffer, meta_buffer)


def record_camera_thread(cam: xiapi.Camera, buffer: RecordingBuffers, no_frames):
    cam.start_acquisition()
    img = xiapi.Image()
    for i in range(no_frames):
        cam.get_image(img)
        ctypes.memmove(ctypes.addressof(buffer.meta_buffer[i]), ctypes.addressof(img), ctypes.sizeof(xiapi.XI_IMG))
        ctypes.memmove(buffer.video_buffer[i], img.bp, ctypes.sizeof(buffer.video_buffer[i]))

    cam.stop_acquisition()


def record_cameras(cameras: List[xiapi.Camera], buffers: List[RecordingBuffers], no_frames: List[int]):
    threads = [
        Thread(target=record_camera_thread, name=f'recording thread {n}', args=(cameras[n], buffers[n], no_frames[n]))
        for n in range(len(cameras))]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def detect_skipped_frames(recording_buffer: RecordingBuffers):
    skip_count = 0
    for i in range(1, len(recording_buffer.meta_buffer)):
        a = recording_buffer.meta_buffer[i].nframe
        b = recording_buffer.meta_buffer[i - 1].nframe
        skip_count += a - b - 1
    return skip_count


###################### Utility Functions ######################

def round_neare(num):
    """
        round_to_nearest_even
    """
    rounded_num = round(num)
    return rounded_num if rounded_num % 2 == 0 else rounded_num + 1

def frame_metadata_as_dict(img):
    def ctypes_convert(obj):  # Very crippled implementation, that is good enough to convert XI_IMG structs.
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, ctypes.Array):
            return [ctypes_convert(e) for e in obj]
        if obj is None:
            return obj

        if isinstance(obj, ctypes.Structure):
            result = {}
            anonymous = getattr(obj, '_anonymous_', [])

            for key, *_ in getattr(obj, '_fields_', []):
                value = getattr(obj, key)
                if key.startswith('_'):
                    continue
                if key in anonymous:
                    result.update(ctypes_convert(value))
                else:
                    result[key] = ctypes_convert(value)
            return result

    result = ctypes_convert(img)
    for key in ['bp', 'size', 'bp_size']:
        del result[key]
    return result


def get_all_camera_parameters(cam: xiapi.Camera):
    def safe_cam_get(cam, param):
        try:
            val = cam.get_param(param)
            if isinstance(val, bytes):
                val = val.decode()
            return val
        except xiapi.Xi_error as e:
            return None

    return {param: val for param in xiapi.VAL_TYPE.keys() if (val := safe_cam_get(cam, param)) is not None}


def get_frame_counters(cam: xiapi.Camera):
    cam.set_counter_selector('XI_CNT_SEL_API_SKIPPED_FRAMES')
    api_skipped = cam.get_counter_value()
    cam.set_counter_selector('XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES')
    transport_skipped = cam.get_counter_value()
    return api_skipped, transport_skipped


def open_camera_by_sn(sn, framerate, exposure, gain, roi):
    cam = xiapi.Camera()
    cam.open_device_by_SN(str(sn))
    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
    set_roi(cam=cam, roi=roi)
    cam.set_exposure(exposure)
    cam.set_gain(gain)
    cam.set_framerate(framerate)
    return cam

def set_roi(cam, roi):
    """
    Since there is a minimum increment for both x/y offset and width/height, the input has to be checked
    """
    # print('Desired Roi')
    # print(roi)
    desired_width = roi['width']
    desired_height = roi['height']
    desired_x_offset = roi['x_offset']
    desired_y_offset = roi['y_offset']

    allowed_width_increment = cam.get_aeag_roi_width_increment()
    allowed_height_increment = cam.get_aeag_roi_height_increment()
    allowed_x_offset_increment = cam.get_offsetX_increment()
    allowed_y_offset_increment = cam.get_offsetY_increment()

    # Width has to be divisible by both height increment and x_offset increment strangely
    capture_width = (round(desired_width / allowed_width_increment) * allowed_width_increment if (desired_width % allowed_width_increment) != 0 else desired_width)
    capture_width = (round(desired_width / allowed_x_offset_increment) * allowed_x_offset_increment if (desired_width % allowed_x_offset_increment) != 0 else desired_width)
    capture_height = (round(desired_height / allowed_height_increment) * allowed_height_increment if (desired_height % allowed_height_increment) != 0 else desired_height)
    # Now values are allowed, can pass to camera
    cam.set_width(round_neare(capture_width))
    cam.set_height(round_neare(capture_height))

    # Have to first set height and width, only then can offsets be set
    capture_x_offset = (round(desired_x_offset / allowed_x_offset_increment) * allowed_x_offset_increment if (desired_x_offset % allowed_x_offset_increment) != 0 else desired_x_offset)
    capture_y_offset = (round(desired_y_offset / allowed_y_offset_increment) * allowed_y_offset_increment if (desired_y_offset % allowed_y_offset_increment) != 0 else desired_y_offset)
    cam.set_offsetX(capture_x_offset)
    cam.set_offsetY(capture_y_offset)
    # set_roi = {'width': capture_width, 'height':capture_height, 'x_offset': capture_x_offset, 'y_offset': capture_y_offset}
    # print("Set roi:")
    # print(set_roi)

###################### main xirec ######################

def parse_camera_arg(s):
    parts = s.split(':')
    if len(parts) > 2:
        raise ValueError()
    parts[0] = int(parts[0])
    return parts


def save_camera_parameters(parameters, data_dir: str, min_stamp=0):
    with open(join(data_dir, 'camera_parameters_' + str(min_stamp) + 'min.json'), 'w') as file:
        json.dump(parameters, file, indent=1)


def save_recording(buf: RecordingBuffers, data_dir: str, file_format='tiff', save_name="frames", min_stamp=0):
    frames_dir = join(data_dir, str(save_name))
    makedirs(frames_dir, exist_ok=True)

    with open(join(data_dir, 'frames_metadata_' + str(min_stamp) + 'min.json'), 'w') as file:
        json.dump([frame_metadata_as_dict(frame) for frame in buf.meta_buffer], file, indent=1)

    # make a string format with the right amount of leading 0's
    path_format = f'{{:0{len(str(len(buf.meta_buffer)))}}}'

    for n, (img, video_buf) in enumerate(zip(buf.meta_buffer, buf.video_buffer)):
        img.bp = ctypes.addressof(video_buf)
        img_np = img.get_image_data_numpy()

        imageio.imwrite(join(frames_dir, path_format.format(n) + f'.{file_format}'), img_np)


def run_recording(cam_sn, duration, framerate, exposure, gain, min_stamp, roi, parent_folder='test'):
    """
    cam_sn: str of XIMEA camera's serial number, get with: cam.get_device_sn() ximea.xiapi
    duration: number of seconds to record for
    framerate: framerate in fps to record at, this is constant
    exposure: exposure in ms
    gain: gain in dB
    min_stamp: the 'minute' value the recording is taken at, used in filenames for labelling
    roi: a dict containing info for the ROI
    parent_folder: the name of the parent folder containing each set of frames + metadata
                    this should probably be experiment specific, e.g. which size bead and the date
    """

    # Set folder name to save frames in
    save_name = str(exposure) + "ms_" + str(gain) + "dB_highres_16bit_" + str(framerate) + "fps_" + str(
        min_stamp) + "min_files"

    # Convert exposure into us from ms to pass to xiapi
    exposure = exposure / 0.001

    print('Opening camera')
    cameras = [open_camera_by_sn(cam_sn, framerate=framerate, exposure=exposure, gain=gain, roi=roi)]
    cameras_sn_str = [cam_sn]

    no_frames = [int(cam.get_framerate() * duration) + 1 for cam in cameras]
    for sn, framerate, fcount in zip(cameras_sn_str, map(lambda cam: cam.get_framerate(), cameras), no_frames):
        print(
            f'serial number: {sn} framerate: {framerate} | exposure: {exposure} | gain : {gain} | no frames: {fcount}')

    memory_requirement_per_frame = list(map(probe_memory_requirements, cameras))

    camera_buffers = list(starmap(allocate_recording_buffers, zip(memory_requirement_per_frame, no_frames)))
    print(f'Allocated {sum(sizeof(b.video_buffer) for b in camera_buffers) / 1024 ** 3:.2f} gigabyte for video')

    print('Storing all camera parameters')
    cameras_parameter_dump = [get_all_camera_parameters(cam) for cam in cameras]

    print('Recording')
    record_cameras(cameras, camera_buffers, no_frames)

    print('Saving')

    recording_dirs = [str(parent_folder)]
    list(map(partial(makedirs, exist_ok=True), recording_dirs))
    list(starmap(partial(save_recording, file_format='tiff', save_name=save_name, min_stamp=min_stamp),
                 zip(camera_buffers, recording_dirs)))
    list(starmap(partial(save_camera_parameters, min_stamp=min_stamp), zip(cameras_parameter_dump, recording_dirs)))

    print('Analysing skipped frames')
    skipped_frames = list(map(detect_skipped_frames, camera_buffers))
    with open(join(parent_folder, 'skipped_frames_' + str(min_stamp) + 'min.txt'), 'w') as file:
        if sum(skipped_frames) > 0:
            for count, camera_sn in zip(skipped_frames, cameras_sn_str):
                file.write(f'[{camera_sn}]: skipped frames: {count}')
        else:
            file.write('No skipped frames')

    list(map(xiapi.Camera.close_device, cameras))

    print('Done')


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()

    argparser.add_argument('-sn','--camera_sn', type=str, default=0,
                           help='The serial number of the camera to record from. If not provided the first connected camera is used')
    argparser.add_argument('-f', '--framerate', default=100, type=int, help='framerate to record at in fps, default is 100fps')
    argparser.add_argument('-d', '--duration', default=5, type=int, help='number of seconds to record, default = 5s')
    argparser.add_argument('-i', '--intervals', default=[0], help='intervals in minutes to take recordings at, supplied like 0,1,3,5,10 (comma seperated)')
    argparser.add_argument('-g', '--gain', default='0', type=int,  help='Gain in dB, default = 0')
    argparser.add_argument('-e', '--exposure', default='5', type=float, help='Exposure in ms, default = 5 ms')
    argparser.add_argument('-t', '--take_number', default='1', help='String to name folder where results will be stored')
    argparser.add_argument('-r', '--roi', default=None,
                           help="If supplied, the roi to record from. If not supplied (default) then can choose a roi interactively\n Has to be supplied, in format : width, height, x_offset, y_offset . Comma seperated and all in pixels")

    args = argparser.parse_args()

    args = argparser.parse_args()

    # This will automatically grab the first connected camera's SN
    if args.camera_sn == 0:
        cam = xiapi.Camera()
        cam.open_device()
        serial_number = cam.get_device_sn()
        cam.close_device()
        cam_sn = serial_number.decode('utf-8')  # Can specify a specific serial number here instead

    # USER DEFINED SETTINGS
    duration = int(args.duration)
    framerate = int(args.framerate)
    exposure = float(args.exposure)
    gain = int(args.gain)
    parent_folder = str(args.take_number)
    record_times = list(args.intervals.split(","))
    record_times = [int(i) for i in record_times]

    user_settings = {'cam_sn': cam_sn, 'duration': duration, 'framerate': framerate, 'exposure': exposure, 'gain': gain,
                     'parent_folder': parent_folder}

    if not args.roi == None:
        print("")
        print(args.roi)
        print("")
        # Check correct number of arguments:
        if not len(args.roi) == 4:
            print("Incorrect number of roi input arguments")
            raise ValueError
        roi_input = list(args.roi.split(','))
        roi = {'width': roi_input[0], 'height': roi_input[1], 'x_offset': roi_input[2], 'y_offset': roi_input[3]}
        user_settings['roi'] = roi
        run_experiment = experiment(settings=user_settings, record_times=record_times, supply_roi=True)

    else:
        run_experiment = experiment(settings=user_settings, record_times=record_times, supply_roi=False)
