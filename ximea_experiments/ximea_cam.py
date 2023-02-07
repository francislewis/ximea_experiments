from ximea import xiapi
from ximea_experiments.xirec import run_recording


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


# cam_sn = '13953350'
# duration = 1
# framerate = 500
# exposure = 1.8
# gain = 19
# parent_folder = 'test_folder'
# roi = {'width': 60, 'height': 50, 'x_offset': 200, 'y_offset': 300}
#
# settings_example = {'cam_sn': cam_sn, 'duration': duration, 'framerate': framerate, 'exposure': exposure, 'gain': gain,
#                     'parent_folder': parent_folder}
#
# connection = camConnect(settings_example)
# connection.addROI(roi)
# connection.captureVideo(min_stamp=1)
