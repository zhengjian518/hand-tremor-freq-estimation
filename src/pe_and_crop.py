# -*- coding: utf-8 -*-
from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
from cpm import CPM
import cv2
import numpy as np
import csv
import os
import sys
import time
from scipy import stats
from collections import deque
import math
from cycler import cycler
from logger import Logger
import util

class PE_AND_CROP(): 
    """
    This class is for pose estimation and joint frame crop on Silvia'PC
    """
    @staticmethod
    def pe_save_batch(video_path_list):
        """Batch saving process

        Args:
            video_path_list: a list of video path strings.
        """

        level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'
        pe_logger = Logger('pe_logger',level_name)

        def pe_save_joint_box(video_path,prediction_arr_path,
                                segment_img_path,conf_arr_path,JOINTS_NUM):
            """Save PE confidence matrix, prediction and prediction box to file, 
                to save eperiment time.
            """
            # JOINTS_NUM = 14
            part_str = ["head", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri",\
                         "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank"]

            # init video writter
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            box_size = 28
            name = part_str[JOINTS_NUM]
            
            def pe():
                video_pe = Video(video_path)
                io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
                                    fps=30,height=368,
                                    width=368*video_pe.WIDTH/video_pe.HEIGHT) #368
                cpm = CPM()
                box_size = 28
                out_video = cv2.VideoWriter(segment_img_path + '{}.avi'.format(name),\
                                                            fourcc, 30.0, (box_size*2,box_size*2),isColor = True)
                print '{}_{} start'.format(video_code,video_name)
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

                    for joint_i in [JOINTS_NUM]:
                        # TODO: may have bug - box is out of image
                        pred_y = int(prediction[joint_i,0])
                        pred_x = int(prediction[joint_i,1])

                        joint_segment_img_path = segment_img_path + '{}/'.format(part_str[joint_i])
                        if not os.path.isdir(joint_segment_img_path):
                            os.mkdir(joint_segment_img_path)

                        joint_conf_arr_path = conf_arr_path + '{}/'.format(part_str[joint_i])
                        if not os.path.isdir(joint_conf_arr_path):
                            os.mkdir(joint_conf_arr_path)

                        joint_frame = frame_pe[pred_y-box_size:pred_y+box_size,
                                                pred_x-box_size:pred_x+box_size]
                        
                        cv2.imwrite(joint_segment_img_path+'seg_{}_{}.png'.format(joint_i,
                                                        video_pe.next_frame_index),
                                                                        joint_frame)
                        out_video.write(joint_frame)
                        joint_conf_map = conf_maps[pred_y-box_size:pred_y+box_size,
                                                    pred_x-box_size:pred_x+box_size,
                                                    joint_i]
                        # joint_conf_map = joint_conf_map / np.sum(joint_conf_map) # Renormalize
                        binay_file = file(joint_conf_arr_path+'conf_{}_{}.bin'.format(joint_i,
                                                video_pe.next_frame_index),"wb")
                        np.save(binay_file,joint_conf_map)

                out_video.release()
                del io_video, video_pe, cpm
            pe()

        for i in range(len(video_path_list)):
            video_path = video_path_list[i]
            video_code = video_path.split('/')[5]
            video_name = video_path.split('/')[6]
            
            print video_code,video_name
            if 'Rechts' in video_code:
                JOINTS_NUM = 4
            elif 'Links' in video_code:
                JOINTS_NUM = 7
            else:
                print 'Error in video_path_list!'
                break

            print 'Video {}_{} in process'.format(video_code,video_name)
            video_code_save_path = '/local/guest/pose_data/results/' +video_code +'_crop/'
            if not os.path.isdir(video_code_save_path):
                os.mkdir(video_code_save_path)
            video_name_save_path = video_code_save_path+ video_name +'/'
            if not os.path.isdir(video_name_save_path):
                os.mkdir(video_name_save_path)

            prediction_arr_path = video_name_save_path + 'prediction_arr/'
            segment_img_path = video_name_save_path + 'segment_img/'
            conf_arr_path = video_name_save_path + 'conf_arr/'
            if not os.path.isdir(prediction_arr_path):
                os.mkdir(prediction_arr_path)
            if not os.path.isdir(segment_img_path):
                os.mkdir(segment_img_path)
            if not os.path.isdir(conf_arr_path):
                os.mkdir(conf_arr_path)

            pe_save_joint_box(video_path,prediction_arr_path,segment_img_path,
                                    conf_arr_path,JOINTS_NUM)

if __name__ == '__main__':
    pe_instance = PE_AND_CROP()

    full_video_list = ['/media/tremor-data/TremorData_split/Tremor_data/T003_Links/Extra_taak_–_Tremor/kinect.avi',
	                  '/media/tremor-data/TremorData_split/Tremor_data/T005_Rechts/Extra_taak_–_links_top/kinect.avi',]
    # patient_code_folder = '/media/tremor-data/TremorData_split/Tremor_data/T001_Links/'
    # video_task_list = util.get_full_path_under_folder(patient_code_folder)
    # full_video_list=[]
    # for task_path in video_task_list:
    #     task_path = task_path + 'kinect.avi'
    #     full_video_list.append(task_path)
    pe_instance.pe_save_batch(full_video_list)
    del pe_instance
