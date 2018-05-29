from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
from cpm import CPM
from fftm import FFTM
from sim import Sim
from configobj import ConfigObj
from error_analysis import get_cropped_frames,plot_to_file
import util
import cv2
import numpy as np
import csv
import os
import sys

import multiprocessing
import time
from scipy import stats
from collections import deque
import math
import matplotlib.pyplot as plt

from logger import Logger
from scipy import signal,io


class TPD():
    """Tremor Frequency Detector Class, similarity approach."""
    
    @staticmethod
    def pe_save_batch(video_path_list,window_size_list):
        """Batch saving process

        Args:
            video_path_list: a list of video path strings.
            window_size_list: a list of window size integers.
        """
        for i in range(len(video_path_list)):
            video_path = video_path_list[i]
            window_size = window_size_list[i]
            noverlap = window_size / 2
            # video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_name = video_path.split('/')[4]
            print 'Video {} in process'.format(video_name)
            result_save_path = '../results/' + video_name +'/'
            if not os.path.isdir(result_save_path):
                os.mkdir(result_save_path)

            prediction_arr_path = result_save_path + 'prediction_arr/'
            segment_img_path = result_save_path + 'segment_img/'
            conf_arr_path = result_save_path + 'conf_arr/'
            if not os.path.isdir(prediction_arr_path):
                os.mkdir(prediction_arr_path)
            if not os.path.isdir(segment_img_path):
                os.mkdir(segment_img_path)
            if not os.path.isdir(conf_arr_path):
                os.mkdir(conf_arr_path)

            TPD.pe_save_joint_box(video_path,window_size,noverlap,
                                    prediction_arr_path,segment_img_path,
                                    conf_arr_path)

    @staticmethod
    def pe_save_joint_box(video_path,window_size,noverlap,prediction_arr_path,
                            segment_img_path,conf_arr_path):
        """Save PE confidence matrix, prediction and prediction box to file, 
            to save eperiment time."""

        JOINTS_NUM = 14
        stride = window_size - noverlap
        level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'

        def pe():

            pe_logger = Logger('pe_logger',level_name)

            video_pe = Video(video_path)
            # video_pe.set_next_frame_index(716)  # for making joint video
            io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
                                fps=30,height=368,
                                width=368*video_pe.WIDTH/video_pe.HEIGHT) #368
            cpm = CPM()

            box_size = None

            while(video_pe.next_frame_index<video_pe.FRAME_COUNT):
                
                pe_logger.info( 'Frame {}/{} is being processed'.format(
                                                    video_pe.next_frame_index,
                                                    int(video_pe.FRAME_COUNT)) )
                frame_pe = io_video.get_video_frames(video_pe,1,
                                                            grayscale_on=False)
                [prediction,conf_maps] = cpm.locate_joints(frame_pe,
                                                        return_conf_map=True)
                if prediction.shape[2]==0:
                    print("Cannot find people in the frame. Skip the frame!")
                    continue
                np.savetxt(prediction_arr_path+'pred_{}.txt'.format(\
                                        video_pe.next_frame_index),prediction)

                # if box_size is None:
                #     box_size = math.sqrt( math.pow(\
                #                             prediction[0,0]-prediction[1,0],2)+\
                #                          math.pow(\
                #                             prediction[0,1]-prediction[1,1],2) )
                #     box_size = int( box_size/2)
                
                box_size = 22

                # for joint_i in [4]: #changed in 28th Feb
                for joint_i in range(JOINTS_NUM):
                    # TODO: may have bug - box is out of image
                    pred_y = int(prediction[joint_i,0])
                    pred_x = int(prediction[joint_i,1])
                    joint_frame = frame_pe[pred_y-box_size:pred_y+box_size,
                                            pred_x-box_size:pred_x+box_size]
                    cv2.imwrite(segment_img_path+'seg_{}_{}.png'.format(joint_i,
                                                    video_pe.next_frame_index),
                                                                    joint_frame)

                    joint_conf_map = conf_maps[pred_y-box_size:pred_y+box_size,
                                                pred_x-box_size:pred_x+box_size,
                                                joint_i]
                    joint_conf_map = joint_conf_map / np.sum(joint_conf_map) # Renormalize
                    np.savetxt(conf_arr_path+'conf_{}_{}.txt'.format(joint_i,
                                    video_pe.next_frame_index),joint_conf_map)
                tfd_logger = Logger('tfd_logger',level_name)

        # PE,FFT Process initialization
        # lock1  = multiprocessing.Lock()
        # pe_process = multiprocessing.Process(target=pe, args=())
        # pe_process.start()
        pe()

    @staticmethod
    def tremor_period_detec_sim_fft_ball(segment_img_path,window_size,noverlap,
                                            fps=30):
        """Compute the frequency of synthetic videos using sim method.
        
        Args:
            segment_img_path: a string of the path to the image folder.
            window_size,noverlap,fps: computation parameters.
        """

        io_video = IOVideo()
        video_lwri = VideoInFrame(segment_img_path,'png',fps)
        sim = Sim(window_size,video_lwri.FPS)

        stride = int(window_size - noverlap)
        len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
        freq_series = np.fft.fftfreq( window_size, 
                                        d=1/float(video_lwri.FPS) )[0:len_half]

        freq_results = []
        psd_sum = None
        stride_count = stride-1
        while(video_lwri.next_frame_index<video_lwri.FRAME_COUNT):
            # print("No_start:{},frame_num:{},No_pe:{},box_size:{}".format(
            #            i*stride,window_size,i*stride+window_size/2,box_size))

            # No_start:0-based, No_pe: 0-based
            # cv2.GaussianBlur(io_video.get_video_frames(video_lwri,1),(0,0),5)
            sim.add_frames(io_video.get_video_frames(video_lwri,1)) 
            if sim.frame_num == window_size:
                stride_count += 1
                if stride_count == stride:
                    stride_count = 0
                    [freq,psd] = sim.fft()
                    freq_results.append(str(freq))
                    if psd_sum is None:
                        psd_sum = psd
                    else:
                        psd_sum += psd
        print freq_results
        # return [freq_results,psd/psd.max()]
        return freq_results

    @staticmethod
    def tremor_period_detec_sim_fft_(segment_img_path,window_size,noverlap,
                                                            joint_No,fps=30):
        """Compute the frequency of videos.

        Args:
            segment_img_path: a string of the path to the image folder.
            window_size,noverlap,fps: computation parameters.
            joint_No: No of the joint to be tested.
        """

        video_code = segment_img_path.split('/')[4]
        video_name = segment_img_path.split('/')[5]

        save_path = '../results/compare_sim_method/{}/'.format(video_code)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        video_lwri = VideoInFrame(segment_img_path,'png',fps)
        sim = Sim(window_size,video_lwri.FPS)

        stride = int(window_size - noverlap)
        len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
        freq_series = np.fft.fftfreq( window_size, 
                                        d=1/float(video_lwri.FPS) )[0:len_half]
        pred = np.loadtxt(
            '../data/video/pe_data_/{}/{}/prediction_arr/pred_{}.txt'.format(
                                                    video_code,video_name,1))
        box_size = int(math.sqrt( math.pow(pred[0,0]-pred[1,0],2) + \
                                    math.pow(pred[0,1]-pred[1,1],2) ) / 2 )

        freq_results = []
        psd_sum = None
        for i in range(0,int(video_lwri.FRAME_COUNT/stride)-1):
            # print("No_start:{},frame_num:{},No_pe:{},box_size:{}".format(i*stride,window_size,i*stride+window_size/2,box_size))

            # No_start:0-based, No_pe: 0-based
            sim.add_frames(get_cropped_frames(video_code,video_name,joint_No,
                                                No_start=i*stride,
                                                frame_num=window_size,
                                                No_pe=i*stride+window_size/2,
                                                box_size=box_size))
            [freq,psd] = sim.fft()

            plot_to_file(x=freq_series,y=psd,xlabel='Frequency/Hz',
                ylabel='PSD',
                title='f={},window_size={},overlap={},fps={}'.format(6,
                                        window_size,noverlap,video_lwri.FPS),
                save_path=save_path+'{}_psd_{}.eps'.format(video_name,i))

            freq_results.append(freq)
            # print(freq)
            if psd_sum is None:
                psd_sum = psd
            else:
                psd_sum += psd

        # Avg PSD
        # plt.figure()
        # plt.plot(freq_series,weighted_fft_seq_sum/weighted_fft_seq_sum.max())
        # plt.xlabel('Frequency/Hz')
        # plt.ylabel('Power Spectral Density')
        # plt.title('Accumulated PSD')
        # plt.show()
        plot_to_file(x=freq_series,y=psd_sum,xlabel='Frequency/Hz',
                ylabel='avg PSD',title='Accumulated PSD',
                save_path=save_path+'{}_avgpsd.eps'.format(video_name))

        print freq_results
        io.savemat(save_path+video_name+'sim.mat',
                                        mdict={'freq':np.array(freq_results)})        
        return freq_results

