# ximea_experiments
This is a tool to take recordings on XIEMA USB Microscopes at set intervals. 
For instance, you can record a 3-second-long video in 1,2,5 and 10 minutes time.
This is especially useful for scientific experiments requiring repeated data collection. 

### Usage

#### Command Line Interface

```
usage: ximea_experiments.py [-h] [-sn CAMERA_SN] [-f FRAMERATE] [-d DURATION] [-i INTERVALS] [-g GAIN] [-e EXPOSURE]
              [-t TAKE_NUMBER] [-r ROI]

options:
  -h, --help            show this help message and exit
  -sn CAMERA_SN, --camera_sn CAMERA_SN
                        The serial number of the camera to record from. If not provided the first connected camera is
                        used
  -f FRAMERATE, --framerate FRAMERATE
                        framerate to record at in fps, default is 100fps
  -d DURATION, --duration DURATION
                        number of seconds to record, default = 5s
  -i INTERVALS, --intervals INTERVALS
                        intervals in minutes to take recordings at, supplied like 0,1,3,5,10 (comma seperated)
  -g GAIN, --gain GAIN  Gain in dB, default = 0
  -e EXPOSURE, --exposure EXPOSURE
                        Exposure in ms, default = 5 ms
  -t TAKE_NUMBER, --take_number TAKE_NUMBER
                        String to name folder where results will be stored
  -r ROI, --roi ROI     If supplied, the roi to record from. If not supplied (default) then can choose a roi
                        interactively Has to be supplied, in format : width, height, x_offset, y_offset . Comma
                        seperated and all in pixels`
```
Here is an example of how you would take a 5-second recording at times: 0 minutes, 1 minute, 3 minutes and 5 minutes with 
framerate 500fps gain of 10dB and exposure of 1.8ms and save the results in a folder named 'take_1'

`ximea_experiments.py -f 500 -d 5 -i 0,1,3,5 -g 10 -e 1.8 -t take_1`

Equivalently:

`ximea_experiments.py --framerate 500 --duration 5 --intervals 0,1,3,5 --gain 10 --exposure 1.8 -take_number take_1`

If you want to supply a custom ROI, instead of using the interactive/graphical picker, use `-r`. The previous example 
would become:

`ximea_experiments.py -f 500 -d 5 -i 0,1,3,5 -g 10 -e 1.8 -t take_1 -r 100,150,10,20`

Equivalently:

`ximea_experiments.py --framerate 500 --duration 5 --intervals 0,1,3,5 --gain 10 --exposure 1.8 -take_number take_1 -roi 100,150,10,20`

for a ROI of width = 100px and height = 150px, x_offset = 10px, y_offset = 20px


#### Python 

The best way to start is to run the example scripts, these will likely be enough for most use cases.

Outside of these, some changes may have to be made, if I have the time, 
fixing the relative imports and packaging it as a usual Python library would be useful.
If you have a specific requirement, there's a good chance there's the start of it within the code, I originally aimed
for quite a general library but have had to get a MVP working quickly. Any questions just ask.

### Requirements
- [XIMEA API](https://www.ximea.com/support/wiki/apis/APIs)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [schedule](https://pypi.org/project/schedule/) 
- [imageio](https://pypi.org/project/imageio/)

### To-Do / Issues
- The processes doesn't currently exit on completion, so you have to manually kill it when it's done

### Troubleshooting
If you get a `ximea.xiapi.Xi_error` error, there are a few things to check:
- Make sure that exposure/framerate works together, if increasing the framerate then 
you may need to lower the exposure
  - Similarly, smaller ROIs allow for higher frame rates
- Sometimes you need to unplug the USB camera and plug it back in (not sure why though)
- Since this is built on the XIMEA API, the camera firmware must be set to XIMEA mode, not USB3 Vision
(this is achieved using the [xiCOP tool](https://www.ximea.com/support/wiki/allprod/XiCOP))
  - Could be nice to try and re-write this using the USB3 firmware to end reliance on the XIMEA API

- #### [coordinateSelect()](ximea_experiments/coord_picker.py)
  - Make sure you choose top left then bottom right
  - There may be issues when selecting large areas near edges, since the actual ROI is rounded (due to set increments on 
  XIMEA cameras) This rounding occurs in `set_roi` within [`xirec.py`](ximea_experiments/xirec.py).
  - Even with the correction in `xirec.set_roi()` it seems that certain choices still fail. If you get a 
  `ximea.xiapi.Xi_error` immediately after selecting the ROI then just restart the script and try again, 
  changing the ROI selection slightly. I think this is fixed now (19/06/23) but I haven't been able to check - the width and height need to be even.

### Acknowledgements
- [`xirec`](ximea_experiments/xirec.py) is mostly copied from [kaspervn/xirec](https://github.com/kaspervn/xirec), 
the frame buffer in RAM is particularly important! 
