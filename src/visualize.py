from video import Video
from io_video import IOVideo
from fftm import FFTM
import cv2
import numpy as np
import os
import sys
import multiprocessing
from logger import Logger
import matplotlib.colors as cl
import matplotlib.cm as cm
from scipy import stats

class Visualize():
    """A class used to generate a heatmap video for visualization."""

    @staticmethod
    def gen_visual_video(video_path, visual_video_savepath,window_size):
        """This function is used to generate a video isualization.

        Args:
            video_path: a string indicating the path to video.
            visual_video_savepath: a string indicating where to save result
                                    video.
            window_size: an integer presenting FFT window size.
        """

        def fftm_proc(window_size, fps, frame_sequence, frame_count,frame_queue,
                        freqmap_queue,freqampl_queue):
            c = 0
            fftm = FFTM(window_size,fps,frame_sequence=frame_sequence)

            while(c<frame_count):
                c+=1
                # print "c:{}".format(c)
                if c < frame_count-window_size/2:
                    fftm.add_frames(frame_queue.get())
                else:
                    fftm.remove_frames(1)
                
                [_,_,fft_sequence_ampl_max,freq_max_ampl] = \
                                        fftm.fft_frames_sequence(filter_on=True)
                freqampl_queue.put(fft_sequence_ampl_max)
                freqmap_queue.put(freq_max_ampl)

        level_name = sys.argv[1] if len(sys.argv) > 1 else 'debug'
        vv_logger = Logger('vv_logger',level_name)
        vv_logger.info("Save visual video to {}".format(visual_video_savepath))

        video = Video(video_path)
        video_bkg = Video(video_path)
        io_video = IOVideo(resizing_on=True,scale=368/video.HEIGHT,
                        write_to_video_on=True,video_path=visual_video_savepath,
                        fps=30,height=368,width=368*video.WIDTH/video.HEIGHT)

        NUM_PROC = 8
        FREQ_SHOW_MAX = 10
        QUEUE_LEN = 10
        WIDTH_PART = int(368*video.WIDTH/video.HEIGHT/NUM_PROC)

        frame_sequence=io_video.get_video_frames(video,window_size/2-1)
        fftm_procs,frame_queues,freqmap_queues,freqampl_queues = [],[],[],[]
        for i in range(NUM_PROC):
            frame_sequence_i = frame_sequence[:,:,
                                        int(i*WIDTH_PART):int((i+1)*WIDTH_PART)]
            frame_queues.append(multiprocessing.Queue(QUEUE_LEN))
            freqmap_queues.append(multiprocessing.Queue(QUEUE_LEN))
            freqampl_queues.append(multiprocessing.Queue(QUEUE_LEN))
            fftm_procs.append(multiprocessing.Process(target=fftm_proc, args=
                (window_size,video.FPS,frame_sequence_i,video_bkg.FRAME_COUNT,
                frame_queues[i],freqmap_queues[i],freqampl_queues[i])))
            fftm_procs[i].start()

        colorize = cm.ScalarMappable(norm=cl.Normalize(vmin=0, vmax=10),
                                                                    cmap=cm.jet)

        freq_map = np.zeros( (368,int(368*video.WIDTH/video.HEIGHT)) )
        freq_ampl_map = np.zeros( (368,int(368*video.WIDTH/video.HEIGHT)) )
        while(video_bkg.next_frame_index<video_bkg.FRAME_COUNT):
            frame_bkg = io_video.get_video_frames(video_bkg,1,
                                                    grayscale_on=False)
            vv_logger.debug( "Frame No.{} is being processed".format(
                                                video_bkg.next_frame_index) )
            if video_bkg.next_frame_index < video_bkg.FRAME_COUNT-window_size/2:
                frame = io_video.get_video_frames(video,1)
                for i in range(NUM_PROC):
                    frame_i = frame[:,int(i*WIDTH_PART):int((i+1)*WIDTH_PART)]
                    frame_queues[i].put(frame_i)

            for i in range(NUM_PROC):
                freq_map[:,int(i*WIDTH_PART):int((i+1)*WIDTH_PART)] = \
                                                        freqmap_queues[i].get()
                freq_ampl_map[:,int(i*WIDTH_PART):int((i+1)*WIDTH_PART)] = \
                                                        freqampl_queues[i].get()

            threshold = np.mean(freq_ampl_map) * 1.5
            mask = freq_ampl_map
            mask = stats.threshold(mask, threshmin=threshold, newval=0)
            mask = stats.threshold(mask, threshmax=threshold, newval=1)
            freq_map = np.multiply(mask, freq_map)
            freq_map_color = (colorize.to_rgba(freq_map) )[:,:,0:3] * 255
            freq_map_color = freq_map_color[:,:,[2,1,0]]

            frame_to_save = 0.5*freq_map_color + 0.5*frame_bkg
            # cv2.imshow('Heatmap',np.uint8(frame_to_save))
            # cv2.waitKey(20)
            io_video.write_frame_to_video( np.uint8(frame_to_save) )
