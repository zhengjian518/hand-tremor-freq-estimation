import numpy as np
import cv2 as cv
from video import Video
from video_in_frame import VideoInFrame
import os
from time import sleep
class IOVideo:
    """Class for video and frames input and ouput.

    Attributes:
        _blurring_on: a boolean indicating whether use blurring or not.
        sigmaX,sigmaY: float parameters for Gaussian blurring.
        ksize: float parameters for Gaussian blurring, if use, set both sigmaX 
                and sigmaY to zero.
        _resizing_on: a boolean indicating whether use resizing or not.
        scale: float parameter to scale the input image.

    """

    def __init__(self,blurring_on=False,sigmaX=None,sigmaY=None,ksize=None,
                resizing_on=False,scale=None,write_to_video_on=False,
                video_path=None,fps=None,height=None,width=None):
        """Init IOVideo."""

        self._blurring_on = blurring_on
        self._resizing_on = resizing_on
        self._write_to_video_on = write_to_video_on
        if self._blurring_on:
            self._sigmaX,self._sigmaY,self._ksize = sigmaX,sigmaY,ksize
        if self._resizing_on:
            self._scale = scale
        if self._write_to_video_on:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            self.out_video_writer = cv.VideoWriter(video_path, fourcc, fps, 
                                                    (int(width),int(height)) )

    def preprocess_frame(self,frame):
        """Preprocessing the frame, including blurring and resizing.
        
        Args:
            frame: a numpy array, the frame to be processed.
        """

        if self._blurring_on:
            frame = cv.GaussianBlur(frame,(self._sigmaX,self._sigmaY),
                                                                    self._ksize) 
            # frame = cv.medianBlur(frame,self._ksize) 
            # frame = cv2.blur(frame,self._ksize)
        if self._resizing_on and self._scale!=None:
            frame = cv.resize(frame,(0,0), fx=self._scale, fy=self._scale, 
                                interpolation = cv.INTER_CUBIC)
            # frame = cv.pyrDown(frame, (int(params['width']*scale), 
            #                                      int(params['height']*scale)))  
        return frame

    def get_video_frames(self,video,frame_num,grayscale_on=True):
        """Get frames from video file. If only one is required, the shape of the
             array will be (h,w) instead of (1,h,w).
        
        Args:
            video: a Video/VideoInFrame instance to be processed.
            frame_num: an integer number indicating how many frames to extract.
            grayscale_on: a boolean indicating whether get grayscale of the 
                        frame.
        Returns:
            frames: a numpy array, frames required, shape:(frame_num*w*h)
        """
        assert isinstance(video,Video) or isinstance(video,VideoInFrame)

        frames = [] 
        for i in range(0,frame_num):
            frame = self.preprocess_frame(video.get_one_frame())
            assert (frame is not None)
            if grayscale_on:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frames.append(frame)
        frames = np.array(frames)
        # return frames
        return frames.squeeze()

    def write_frame_to_image(self,path,frame):
        """Save one frame to image file.

        Args:
            frame: a numpy array, the frame to be saved.
            path: path to save image, including file name.
        """
        cv.imwrite(path,frame)

    def write_frame_to_video(self,frame):
        """Save one frame to video file.

        Args:
            frame: a numpy array, the frame to be saved.
        """
        self.out_video_writer.write(frame)

    def sample_video_frames(self,video,frame_num=None,save_path=None):
        """Sample frames from video and save to files.

        Args:
            video: a Video/VideoInFrame instance to be processed.
            frame_num: an integer indicating how many frames to sample.
            save_path: a string indicating the folder to save file.
        Returns:
            frames: a numpy array, frames required, shape:(frame_num*w*h)
        """
        assert isinstance(video,Video) or isinstance(video,VideoInFrame)
        assert frame_num != 0

        if save_path is None:
            save_path = video.video_path[:-(len(
                                    os.path.splitext(video.video_path)[1]))] + \
                                     '_sample_frames/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        frames = []
        framename_string = ''
        video.set_next_frame_index(0)
        if frame_num is None:
            print 'Sample {} frames from {}'.format(
                                            video.FRAME_COUNT,video.video_path)
            input('Too many framess, will not return frames!')
            len_name = len(str(video.FRAME_COUNT))
            for frame_No in range(1,int(video.FRAME_COUNT+1)):
                frame = self.preprocess_frame(video.get_one_frame())
                frame_name = '{:0{}d}.png'.format(frame_No,len_name)
                framename_string += frame_name + '\n'
                self.write_frame_to_image(save_path + frame_name,frame)
                sleep(0.01)
        else:
            print 'Sample {} frames from {}'.format(frame_num,video.video_path)
            len_name = len(str(frame_num))
            for frame_No in range(1,frame_num+1):
                frame_index = int(video.FRAME_COUNT/frame_num) * frame_No
                video.set_next_frame_index(frame_index-1)
                frame = self.preprocess_frame(video.get_one_frame())
                frames.append(frame)
                frame_name = '{:0{}d}.png'.format(frame_No,len_name)
                framename_string += frame_name + '\n'
                self.write_frame_to_image(save_path + frame_name,frame)
                sleep(0.01)

        with open(save_path+"images_list.txt", "w") as text_file:
            text_file.write(framename_string[:-1])

        print 'Done! Save frames to {}'.format(save_path)

        frames = np.array(frames)
        return frames
