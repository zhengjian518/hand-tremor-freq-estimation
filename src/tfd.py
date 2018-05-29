import matplotlib.pyplot as plt

from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
from cpm import CPM
from fftm import FFTM
from configobj import ConfigObj
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
from cycler import cycler
from logger import Logger

class TFD():
    """Tremor Frequency Detector Class."""

    @staticmethod
    def tfd_batch(video_path_list,window_size_list,joint_list):
        """TFD Pipeline batch processing, automatically generate saving path and
            file name,
            outcomes will be placed at '../results/video_name/' directory. 

        Args:
            video_path_list: a list of video path strings.
            window_size_list: a list of window size integers.
            joint_list: a list of integer indicating the joints to test.
        """
        for i in range(len(video_path_list)):
            video_path = video_path_list[i]
            window_size = window_size_list[i]
            noverlap = window_size / 2
            # video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_code = video_path.split('/')[3]# 3(T008), 4(All)
            video_name = video_path.split('/')[4]# 4      , 5
            print 'Video {}_{} in process'.format(video_code,video_name)
            result_save_path = '../results/' + video_code +'/'+ video_name +'/'
            if not os.path.isdir(result_save_path):
                os.makedirs(result_save_path)

            fft_video_path = result_save_path + video_name + '_freq_heatmap.avi'
            final_freq_csv_path = result_save_path + video_name + '_tfd_freq.csv'

            TFD.tremor_freq_detec(video_path,window_size,noverlap,fft_video_path,final_freq_csv_path,joint_list)

    # TFD Pipeline v2.1: sliding pe cropping, estimate pe on the middle frame of a window 
    #                   v1.0 single-frame confidence approach + accumulated-PSD
    @staticmethod
    def tremor_freq_detec(video_path,window_size,noverlap, 
                            fft_video_path,final_freq_csv_path,
                            JOINT_LIST = [7]):
        # Constant
        FREQ_SHOW_MAX = 10
        JOINTS_NUM = 14
        stride = window_size - noverlap # 121- 60(121/2) = 61

        level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'

        def pe(conf_map_queue,joint_pred_queue,lock):
            """Pose estimation.
            
            Args: 
                conf_map_queue: a queue saving confidence map, every stride 
                                steps.
                joint_pred_queue: a queue saving saving joint predictions for 
                                    every frame.
                lock: lock for safety
            """
            pe_logger = Logger('pe_logger',level_name)

            video_pe = Video(video_path)
            io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
                        fps=25,height=368,
                        width=368*video_pe.WIDTH/video_pe.HEIGHT)
            cpm = CPM()

            for i in range(0,int(video_pe.FRAME_COUNT/stride)-1):
                # Step 1: Set pe pointer to the middle of window
                frame_pe_index = i * stride + window_size/2
                video_pe.set_next_frame_index(frame_pe_index)

                pe_logger.info( 'Frame {}/{} is being processed'.format( \
                                                    video_pe.next_frame_index,
                                                    int(video_pe.FRAME_COUNT)) )

                # Step 2: Predict pose and save to queue
                frame_pe = io_video.get_video_frames(video_pe,1,
                                                            grayscale_on=False)
                [prediction,conf_map] = cpm.locate_joints(frame_pe,
                                                        return_conf_map=True)
                lock.acquire()
                joint_pred_queue.put(prediction)
                conf_map_queue.put(conf_map)
                lock.release()

                # Step 3: Label pose on image and save to file 
                pe_save_path = os.path.dirname(final_freq_csv_path)+'/pe/'
                if not os.path.isdir(pe_save_path):
                    os.mkdir(pe_save_path)

                for joint_i in JOINT_LIST:
                    pred_y = int(prediction[joint_i,0])
                    pred_x = int(prediction[joint_i,1])
                    frame_pe = cv2.circle(frame_pe, (pred_x, pred_y), 3, 
                                                                (0,0,255), -1)  # red spot in image
                cv2.imwrite(pe_save_path+'{}.png'.format(i), frame_pe)
                np.savetxt(pe_save_path+'pred_{}.txt'.format(i),prediction)

        def fft(joint_pred_queue,conf_map_queue,fft_sequence_queue,
                joint_conf_map_queue,lock,):
                            
            fft_logger = Logger('fft_logger',level_name)

            # Init videos
            video_background = Video(video_path) # Visualization
            video_fft = Video(video_path)
            io_video = IOVideo(resizing_on=True,scale=368/video_fft.HEIGHT,
                    write_to_video_on=True,video_path=fft_video_path,fps=25,
                    height=368,width=368*video_fft.WIDTH/video_fft.HEIGHT)
            
            # Init FFTM for each joint
            fftm = []
            for joint_i in range(JOINTS_NUM):
                fftm.append(FFTM(window_size,video_fft.FPS))
            
            box_size = None
            stride_count = stride-1  # 61-1
            
            for i in range(0,int(video_fft.FRAME_COUNT/stride)-1):
                fft_logger.info( 'Frame ({}~{})/{} is being processed'.format(
                                i*stride, i*stride+window_size-1,
                                int(video_fft.FRAME_COUNT)) )

                joint_fft_squences = []
                joint_conf_maps = []
                freq_map = np.zeros( (368,
                                    int(368*video_fft.WIDTH/video_fft.HEIGHT)) )

                # Step 1: Get pe prediction and init box size with first 
                #           estimation.
                joint_preds =  joint_pred_queue.get()
                conf_maps = conf_map_queue.get()
                if box_size is None:
                    box_size = math.sqrt( 
                                math.pow(joint_preds[0,0]-joint_preds[1,0],2)+ \
                                math.pow(joint_preds[0,1]-joint_preds[1,1],2) )
                    box_size = int(box_size/2)

                # Step 2: Crop joint segments from image and send to fftm 
                #           and get PSD,
                for joint_i in JOINT_LIST:# range(JOINTS_NUM):
                    # TODO: may have bug - box is out of image
                    pred_y = int(joint_preds[joint_i,0])
                    pred_x = int(joint_preds[joint_i,1])
                    fftm[joint_i].add_frames(
                                        get_cropped_frames(video_fft,io_video, 
                                        No_start=i*stride,frame_num=window_size, 
                                        pred_x=pred_x,pred_y=pred_y,
                                        box_size=box_size))
                                                
                    fft_sequence_ampl,_,_,freq_max_ampl= \
                            fftm[joint_i].fft_frames_sequence(filter_on=False,
                            threshold_on=True)
                    joint_fft_squences.append(fft_sequence_ampl)

                    joint_conf_map = conf_maps[pred_y-box_size:pred_y+box_size,
                                                pred_x-box_size:pred_x+box_size,
                                                joint_i]
                    joint_conf_map = joint_conf_map / np.sum(joint_conf_map) 
                    joint_conf_maps.append(joint_conf_map)
              
                    # Visualization
                    freq_map[pred_y-box_size:pred_y+box_size,\
                                pred_x-box_size:pred_x+box_size] = freq_max_ampl

                # Step 3: Save conf map and fft results to queue
                lock.acquire()
                joint_conf_map_queue.put(joint_conf_maps)
                fft_sequence_queue.put(joint_fft_squences)
                lock.release()

                # Step 4: Save to Video for Visualization
                if i==0:
                    start_No = i*stride
                    frame_to_take_num = window_size*3/4 + 1
                elif i==int(video_fft.FRAME_COUNT/stride)-2:
                    start_No = i*stride+window_size/4
                    frame_to_take_num = window_size*3/4 + 1
                else:
                    start_No = i*stride+window_size/4
                    frame_to_take_num = window_size/2 + 1
                video_background.set_next_frame_index(start_No)
                # print("start from:{}, frame_num:{}".format(start_No,
                #                                            frame_to_take_num))
                freq_map = 0.5*util.colorize(np.divide(freq_map,FREQ_SHOW_MAX))
                for frame_No in range(frame_to_take_num):
                    frame_background = io_video.get_video_frames(
                                        video_background,1,grayscale_on=False)
                    frame_to_save = freq_map + 0.5*frame_background
                    io_video.write_frame_to_video( np.uint8(frame_to_save) )

            del io_video


        tfd_logger = Logger('tfd_logger',level_name)

        # Init PE,FFT process
        lock1  = multiprocessing.Lock()
        lock2  = multiprocessing.Lock()
        conf_map_queue = multiprocessing.Queue(1)
        joint_pred_queue = multiprocessing.Queue(1)
        fft_sequence_queue = multiprocessing.Queue(1)
        joint_conf_map_queue = multiprocessing.Queue(window_size/2+1)
        pe_process = multiprocessing.Process(target=pe, args=(conf_map_queue,
                                                joint_pred_queue,lock1,))
        fft_process = multiprocessing.Process(target=fft,args=(joint_pred_queue, 
                                                conf_map_queue, 
                                                fft_sequence_queue, 
                                                joint_conf_map_queue,lock2,))
        pe_process.start()
        fft_process.start()

        # Init frequency series
        len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
        freq_series = np.fft.fftfreq( window_size, d=1/float(30) )[0:len_half]
        tfd_logger.debug('Frequency Series: {}'.format(freq_series))

        # Init CSV file to save results
        joints_string = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',  
                        'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank']
        joints_string = [joints_string[x] for x in JOINT_LIST]
        psd_save_path = os.path.dirname(final_freq_csv_path) + \
                                                            '/accumulated_psd/'
        if not os.path.isdir(psd_save_path):
            os.mkdir(psd_save_path)
        csvfile = open(final_freq_csv_path, 'wb')
        csvwriter = csv.writer(csvfile)
        csv_head = []
        for joint in joints_string:
            csv_head.extend([joint,'is_peak'])
        csvwriter.writerow(csv_head)
        csvwriter_avgpsd = csv.writer(open(psd_save_path+'psd_avg.csv', 'wb'))
        csvwriter_ispeak = csv.writer(open( \
                                        os.path.dirname(final_freq_csv_path) + \
                                        '/is_peak_overall.csv', 'wb'))
        csvwriter_ispeak.writerow(joints_string)

        fft_count = 0
        psd_avg = []
        while( pe_process.is_alive() or fft_process.is_alive() or \
                                             not fft_sequence_queue.empty() ):
            if( (not joint_conf_map_queue.empty()) and \
                                            (not fft_sequence_queue.empty()) ):
                freq_results, psds = [],[]

                csvwriter_psd = csv.writer(open(\
                                psd_save_path+"{}.csv".format(fft_count), 'wb'))

                # Step 1: Get fft psd and conf map from queue
                joint_conf_maps = joint_conf_map_queue.get()
                fft_sequences = fft_sequence_queue.get()
                
                psds.append(freq_series)
                # Step 2: Compute weighted psd, detect peak frequency and save 
                #         psd to file
                for joint_i in range(len(fft_sequences)):
                    weighted_fft_sequence = fft_sequences[joint_i] * \
                                                     joint_conf_maps[joint_i] 
                    # window_size/2 *width*height * width*height
                    weighted_fft_sequence = np.sum(weighted_fft_sequence,
                                                                    axis=(1,2))
                    weighted_fft_sequence = weighted_fft_sequence / \
                                                     weighted_fft_sequence.max()
                    psds.append(tuple(weighted_fft_sequence))
                    if(len(psd_avg)<len(fft_sequences)):
                        psd_avg.append(weighted_fft_sequence)
                    else:
                        psd_avg[joint_i] += weighted_fft_sequence

                    
                    plot_to_file(x=freq_series,y=weighted_fft_sequence,
                            xlabel='Frequency (Hz)',
                            ylabel='Power Spectral Density',
                            title='Accumulated PSD for {}, time: {:.1f} (s)'.format(
                                                        joints_string[joint_i],
                                                        float((fft_count+1)*\
                                                        window_size)/2.0/30.0),
                            save_path=psd_save_path+'apsd_{}_{}.eps'.format(
                                            joints_string[joint_i],fft_count))

                    freq_i = freq_series[np.argmax(weighted_fft_sequence)]

                    freq_results.append(str(freq_i))
                    if weighted_fft_sequence.max()> \
                                                np.mean(weighted_fft_sequence)+\
                                                3*np.std(weighted_fft_sequence):
                        is_peak = 1 
                    else:
                        is_peak = 0
                    freq_results.append(is_peak)

                    tfd_logger.debug('accumulated_psd: {}'.format( \
                                                        weighted_fft_sequence))
                    tfd_logger.info('{} Freq: {} Hz'.format( \
                                                joints_string[joint_i],freq_i))

                # Step 3: Save results to .csv file
                csvwriter.writerow(freq_results)
                csvwriter_psd.writerow(['freq']+joints_string)
                for row in zip(*psds):
                    csvwriter_psd.writerow(row)
                del csvwriter_psd
                fft_count+=1


        for joint_i in range(len(joints_string)):
            plot_to_file(x=freq_series,y=psd_avg[joint_i]/fft_count,
                        xlabel='Frequency (Hz)',ylabel='Power Spectral Density',
                        title='Average PSD for {}'.format( \
                                                        joints_string[joint_i]),
                        save_path=psd_save_path+'avgpsd_{}.eps'.format( \
                                                        joints_string[joint_i]))        
        is_peak = []
        for psd in psd_avg:
            if psd.max()>np.mean(psd)+3*np.std(psd):
                is_peak.append(1)
            else:
                is_peak.append(0)
        csvwriter_ispeak.writerow(is_peak)
        del csvwriter_ispeak

        psd_avg = [ tuple(psd/fft_count) for psd in psd_avg ]
        psd_avg.append(tuple(freq_series))
        csvwriter_avgpsd.writerow(joints_string+['freq'])
        for row in zip(*psd_avg):
            csvwriter_avgpsd.writerow(row)
        csvfile.close()
        del csvwriter_avgpsd

def get_cropped_frames(video,io_video,No_start,frame_num,
                        pred_x,pred_y,box_size):
    """Get cropped joint frames from video.

    Args:
        video: a Video or VideoInFrame object.
        io_video: an IO_Video object.
        No_start: an integer indicating which frame to start from, 0-based.
        frame_num: an integer indicating how many frames to extract.
        pred_x,pred_y: integers indicating the prediction of joints.
        box_size: an integer indicating half size of the box.
    Return:
        frames: a numpy array including cropped frames in gray scale, 
                shape:frame_num*height*width. 
    """

    video.set_next_frame_index(No_start)

    frames = []
    while(len(frames)<frame_num):
        frame = io_video.get_video_frames(video,1,grayscale_on=True)
        joint_frame = frame[pred_y-box_size:pred_y+box_size, \
                            pred_x-box_size:pred_x+box_size]
        joint_frame = cv2.GaussianBlur(joint_frame,(0,0),5)

        # save_path = '../results/T035_Links/Rust/pe_joint/{}/'.format(No_start)
        # if not os.path.isdir(save_path):
        #     os.makedirs(save_path)
        # cv2.imwrite(save_path+'{}.png'.format(len(frames)),joint_frame)
        
        frames.append(joint_frame)
    return np.array(frames)

def plot_to_file(x,y,xlabel,ylabel,title,save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['b']));
    ax.plot( x, y )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(save_path, format='eps')
    plt.close(fig)
