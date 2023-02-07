import cv2
import numpy as np


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


######## EXAMPLE ########
# Create/run instance
# test_instance = coordinateSelect('results/test.png', 0)
# print(test_instance.dimensions)
