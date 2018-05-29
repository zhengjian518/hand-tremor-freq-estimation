import numpy as np
import cv2 
import util
import os

class VideoInFrame():
    """Class for a single video in frame format.

    Attributes:
        video_path: a string of video saving path.
        frame_list: a list of all frame names saved in video path.
        next_frame_index: an integer video frame pointer that points to the next
                         frame taken out.
        FPS,HEIGHT,WIDTH,FRAME_COUNT: constant integer values of video 
                                    properties.    
    """

    def __init__(self,video_path,frame_format='jpg',fps=30.0):
        """Init a video file, its parameters and video frame pointer."""

        assert os.path.isdir(video_path)

        self.next_frame_index = 0
        self.video_path = video_path
        self._get_frame_param(video_path)
        self.FPS = fps
        self.frame_list = util.get_file_list(video_path,frame_format)
        convert = lambda text: float(text) if text.isdigit() else text
        alphanum = \
            lambda key: [ convert(x) for x in (key.split('.')[0]).split('_') ]
        self.frame_list.sort(key=alphanum)              # sort by B - G - R ?
        self.FRAME_COUNT = len(self.frame_list)
        

    def get_one_frame(self):
        """Get next frame from video file while the pointer increases one.

        Returns:
            frame: a numpy array of a frame from the video in BGR.
        """

        frame = cv2.imread(self.video_path+ \
                                        self.frame_list[self.next_frame_index])
        self.next_frame_index += 1
        return frame

    def set_next_frame_index(self, pos):
        """Set the frame pointer to specified position.

        Args:
            pos: the position set to frame pointer
        """
    	self.next_frame_index = pos

    def _get_frame_param(self, video_path):
        """Internal funtion used to read the parameter from video file."""

        img = cv2.imread(video_path+os.listdir(video_path)[0])
        self.HEIGHT,self.WIDTH = img.shape[:2]

