import numpy as np
import cv2 as cv
import os.path

class Video():
    """Class for a single video.

    Attributes:
        video: OpenCV VideoCapture type.
        next_frame_index: an integer video frame pointer that points to the next
                             frame taken out.
        video_path: a string of video path
        FPS,HEIGHT,WIDTH,FRAME_COUNT: constant float values of video properties.
        CAP_PROP_POS_FRAMES: opencv tag for frame position in the video, 
                            different for opencv 3.x.x and 2.x.x.
    """

    def __init__(self,video_path):
        """Init video file, its parameters and video frame pointer."""

        assert os.path.isfile(video_path)

        self.video = cv.VideoCapture(video_path)
        self.video_path = video_path
        self._get_video_param()
        self.next_frame_index = 0

    def get_one_frame(self):
        """Get next frame from video file while the pointer increases one.

        Returns:
            frame: a numpy array of a frame from the video in BGR.
        """

        ret, frame = self.video.read()
        self.next_frame_index += 1
        return frame

    def set_next_frame_index(self, pos):
        """Set the frame pointer to specified position.

        Args:
            pos: the position set to frame pointer
        """
        self.video.set(self.CAP_PROP_POS_FRAMES,pos)
        self.next_frame_index = pos
        
    def _get_video_param(self):
        """Internal funtion used to read the parameter from video file."""

        # get OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
        # get video frame rate and resolution
        if int(major_ver)  < 3:
            self.FPS = self.video.get(cv.cv.CV_CAP_PROP_FPS)
            self.HEIGHT = self.video.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT)
            self.WIDTH = self.video.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)
            self.FRAME_COUNT = self.video.get(cv.cv.CV_CAP_PROP_FRAME_COUNT)
            self.CAP_PROP_POS_FRAMES = cv.cv.CV_CAP_PROP_POS_FRAMES
        else :
            self.FPS = self.video.get(cv.CAP_PROP_FPS)
            self.HEIGHT = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)
            self.WIDTH = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
            self.FRAME_COUNT = self.video.get(cv.CAP_PROP_FRAME_COUNT)
            self.CAP_PROP_POS_FRAMES = cv.CAP_PROP_POS_FRAMES

        if self.FPS == None or np.isinf(self.FPS):
            self.FPS = input(
                        "Cannot get video FPS info! Please manually input!\n")
        if self.FRAME_COUNT == None or np.isinf(self.FRAME_COUNT):
            self.FRAME_COUNT = input(
                "Cannot get video FRAME_COUNT info! Please manually input!\n")

