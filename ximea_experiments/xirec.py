"""
This code is modified from https://github.com/kaspervn/xirec
I've removed some features which aren't needed here and turned it into a single script instead of CLI

It could definitely be cleaned up futher and I think putting everything into classes could help in the long run
I also removed some functionality allowing multiple cameras, will probably have ot look at xirec if this is needed
"""

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
from ximea import xiapi


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


# cam_sn = '13953350'
# duration = 1
# framerate = 500
# exposure = 1.8
# gain = 19
# roi = {'width': 60, 'height': 50, 'x_offset': 200, 'y_offset': 300}
#
# run_recording(cam_sn, duration, framerate=framerate, exposure=exposure, gain=gain, min_stamp=1, parent_folder='test2', roi=roi)
