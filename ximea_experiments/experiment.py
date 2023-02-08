import schedule
import time
from ximea_experiments.schedules import scheduler
from ximea_experiments.coord_picker import coordinateSelect
from ximea_experiments.ximea_cam import camConnect

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
