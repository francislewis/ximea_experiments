# ximea_experiments
This is a tool to take recordings on XIEMA USB Microscopes at set intervals. 
For instance, you can record a 3-second-long video in 1,2,5 and 10 minutes time.
This is especially useful for scientific experiments requiring repeated data collection. 

### Usage
The best way to start is to run the example scripts, these will likely be enough for most use cases.

Outside of these, some changes may have to be made, if I have the time, 
fixing the relative imports and packaging it as a usual Python library would be useful, along with a CLI version.
If you have a specific requirement, there's a good chance there's the start of it within the code, I originally aimed
for quite a general library but have had to get a MVP working quickly. Any questions just ask.

### Requirements
- [XIMEA API](https://www.ximea.com/support/wiki/apis/APIs)
- OpenCV
- schedule 

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
  changing the ROI selection slightly

### Acknowledgements
- [`xirec`](ximea_experiments/xirec.py) is mostly copied from [kaspervn/xirec](https://github.com/kaspervn/xirec), 
the frame buffer in RAM is particularly important! 
