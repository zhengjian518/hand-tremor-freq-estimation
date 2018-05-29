import matplotlib
matplotlib.use('TkAgg')

from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
from cpm import CPM
from fftm import FFTM
from tfd import TFD
from configobj import ConfigObj
import util
import cv2
import numpy as np
import csv
import os
import scipy.io
import math
class Evaluate:
    """Static Class for the evaluation of pose estimation and frequency detector,
    All results path will be defines here.
    TremorFrequencyDetector/results/evaluate/: {video_name}/{video_name}_samples/im{frame_number}.png + images_list.txt

    """
    
    @staticmethod
    def pe_evaluate_phase1_1_v2_pe(video_path_list,frame_sample_num):
        """Sample frames from the remote disk and save to local.

        Args:
            video_path_list: a list of strings indicating the video path.
            frame_sample_num: an integer indicating the number of frames to 
                                sample.
        """
        for video_path in video_path_list:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print 'Video {} is being sampled'.format(video_name)
            # Dir for each video evaluate results
            evaluate_result_path = '../results/evaluate/' + video_name + '/'
            if not os.path.isdir(evaluate_result_path):
                os.mkdir(evaluate_result_path)
            # Dir for sampled frames
            sample_frames_path = evaluate_result_path + video_name + '_samples/'

            # Sample frames for each video
            video_to_evaluate = Video(video_path)
            io_video = IOVideo(resizing_on=True,
                                scale=368/video_to_evaluate.HEIGHT)
            io_video.sample_video_frames(video_to_evaluate,frame_sample_num,
                                                            sample_frames_path)
    
    @staticmethod
    def pe_evaluate_phase1_2_v2_pe(video_path_list):
        """Predict pose on sampled frames.

        Args:
            video_path_list: a list of strings indicating the video path.
        """

        def _pe_video(sample_frames_path,csv_savepath,cpm):
            csvfile = open(csv_savepath, 'wb')
            csvwriter = csv.writer(csvfile)
            # Header
            joints_list = []
            for joint in cpm._model['joints_string']:
                joints_list.append(joint+'_x')
                joints_list.append(joint+'_y')
            csvwriter.writerow(joints_list)

            frame_name_list = util.get_file_list(sample_frames_path,'png')
            # Prediction
            for i in range(len(frame_name_list)):
                print '   Frame No.{} in pe '.format(i)
                frame = cv2.imread(sample_frames_path+frame_name_list[i])
                prediction = cpm.locate_joints(frame)
                if prediction is not None:
                    util.write_mat_to_csv(prediction[:,:,0].squeeze(),csvwriter)
                else:
                    util.write_mat_to_csv(np.zeros(14,2),csvwriter)

        cpm = CPM()
        for video_path in video_path_list:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print 'Video {} is being PE evaluated'.format(video_name)
            sample_frames_path = '../results/evaluate/' + video_name + '/' + \
                                                        video_name + '_samples/'
            # Dir for pe results
            pe_csv_path = '../results/evaluate/' + video_name + '/' + \
                                                        video_name + '_pe.csv'
            _pe_video(sample_frames_path,pe_csv_path,cpm)

    @staticmethod
    def pe_evaluate_phase2_gt(video_gt_mat_path_list):
        """Evaluate_pe Step 2: generate ground truth csv.
        
        Args:
            video_gt_mat_path_list:  a list of strings, manual annotation mat 
                                    file path.
        """
        for video_gt_mat_path in video_gt_mat_path_list:
            video_name = os.path.basename( \
                                    os.path.normpath(video_gt_mat_path))[:-7]
            gt_csv_path = '../results/evaluate/' + video_name + '/' + \
                                                        video_name + '_gt.csv'
            Evaluate.mat_files_to_csv(video_gt_mat_path,gt_csv_path)

    @staticmethod
    def pe_evaluate_phase3_pckh(video_path_list,norm_factor):
        """Evaluate_pe Step 3: compare ground truth csv and test results.
        
        Args:
            video_path_list: a list of video path strings.
            norm_factor: an integer inficating the normalization factor for pck 
                            computation.

        Return:
            pckh_list: a list of pckh value for total joints, head, arm and leg 
                        parts.
        """
        csvfile = open('../results/evaluate/pckh_{}.csv'.format(norm_factor),
                                                                         'wb')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video name','PCKh_frac','PCKh','PCKh_head',
                            'PCKh_arm','PCKh_leg','PCKh_frac_dc','PCKh_dc',
                            'PCKh_head_dc','PCKh_arm_dc','PCKh_leg_dc'])

        tag_list = ['total','tcorrect','head','hcorrect','arm','acorrect','leg',
                    'lcorrect']
        count_cpm,count_dc = {},{}
        for tag in tag_list:
            count_cpm[tag],count_dc[tag] = 0,0

        for video_path in video_path_list:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            pe_csv_path = '../results/evaluate/' + video_name + '/' + \
                                                        video_name + '_pe.csv'
            dc_pe_csv_path = '../results/evaluate/' + video_name + '/' + \
                                                 video_name + '_pe_deepcut.csv'
            gt_csv_path = '../results/evaluate/' + video_name + '/' + \
                                                        video_name + '_gt.csv'

            count_cpm_1video = Evaluate.pckh(gt_csv_path,pe_csv_path,
                                                                    norm_factor)
            count_dc_1video = Evaluate.pckh(gt_csv_path,dc_pe_csv_path,
                                                                    norm_factor)
            
            video_row_to_save = [video_name]
            for count in [count_cpm_1video,count_dc_1video]:
                video_row_to_save.append( str(int(count['tcorrect']))+'/'+\
                                                    str(int(count['total'])) )
                for x in range(len(tag_list)/2):
                    if count[tag_list[2*x]]==0:
                        video_row_to_save.append('0.0')
                    else:
                        video_row_to_save.append( str( 
                                                float(count[tag_list[2*x+1]])/\
                                                float(count[tag_list[2*x]]) ) )
            csvwriter.writerow(video_row_to_save)

            for tag in tag_list:
                count_cpm[tag] += count_cpm_1video[tag]
                count_dc[tag] += count_dc_1video[tag]

        for cc in range(2):
            count_list = [count_cpm,count_dc]
            name_list = ['CPM','DC']
            count = count_list[cc]
            result = 'Norm fractor:{}'.format(norm_factor)
            for x in range(len(tag_list)/2):
                result += ', PCKh_{}_{}: {}'.format(name_list[cc],tag_list[2*x],
                     float(count[tag_list[2*x+1]])/float(count[tag_list[2*x]]))
            print result

        row_to_save = ['Total']
        for x in range(len(tag_list)/2):
            row_to_save.append( str(int(count_cpm[tag_list[2*x+1]]))+'/'+\
                                            str(int(count_cpm[tag_list[2*x]])) )
            row_to_save.append( float(count_cpm[tag_list[2*x+1]])/\
                                            float(count_cpm[tag_list[2*x]]) )
        csvwriter.writerow(row_to_save)
        csvfile.close()

        pckh_list = []
        for count in [count_cpm,count_dc]:
            for x in range(len(tag_list)/2):
                pckh_list.append( float(count[tag_list[2*x+1]])/\
                                                float(count[tag_list[2*x]]) )
        return pckh_list

    @staticmethod
    def mat_files_to_csv(mat_dir_path,csv_save_path):
        """Translate multiple .mat files to single .csv file.
        
        Args:
            mat_dir_path: a string indicating the path saving .mat files.
            csv_save_path: a string indicating the path to save .csv file.
        """
        mat_list = util.get_file_list(mat_dir_path, 'mat')

        csvfile = open(csv_save_path, 'wb')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['head_x','head_y','neck_x','neck_y','Rsho_x',
                            'Rsho_y','Relb_x','Relb_y','Rwri_x','Rwri_y',
                            'Lsho_x','Lsho_y','Lelb_x','Lelb_y','Lwri_x',
                            'Lwri_y','Rhip_x','Rhip_y','Rkne_x','Rkne_y',
                            'Rank_x','Rank_y','Lhip_x','Lhip_y','Lkne_x',
                            'Lkne_y','Lank_x','Lank_y'])
        for mat_file in mat_list:
            mat = (scipy.io.loadmat(mat_dir_path + mat_file))['labels']
            mat.shape
            order_joint = [13,12,8,7,6,9,10,11,2,1,0,3,4,5]
            order_xy = [1,0,2]
            mat = mat[:,order_joint]
            mat = mat[order_xy,:]
            util.write_mat_to_csv(mat[0:2,:].squeeze(),csvwriter,False)

    @staticmethod
    def pckh(csvfile_name_gt,csvfile_name_pe,norm_factor):
        """Calculate Percentage Correct Keypoint(half-head size as boxsize) 
            between ground truth and test results.

        Args:
            csvfile_name_gt: a string indicating the ground-truth csv file name.
            csvfile_name_pe: a string indicating the pose estimation csv file 
                                name.
            norm_factor: an integer indicating the normalization factor.
        Return:
            count: a dictionary showing the number of correct predictions and 
                    total number of joints
        """
        PCKh = 0
        header = ['head_x','head_y','neck_x','neck_y','Rsho_x','Rsho_y',
                'Relb_x','Relb_y','Rwri_x','Rwri_y','Lsho_x','Lsho_y',
                'Lelb_x','Lelb_y','Lwri_x','Lwri_y','Rhip_x','Rhip_y',
                'Rkne_x','Rkne_y','Rank_x','Rank_y','Lhip_x','Lhip_y',
                'Lkne_x','Lkne_y','Lank_x','Lank_y']
        with open(csvfile_name_gt,'rb') as csvfile_gt, \
                open(csvfile_name_pe,'rb') as csvfile_pe:
            # Check the length of two csv files
            row_count = sum(1 for row in csvfile_gt) -1
            assert row_count == sum(1 for row in csvfile_pe)-1

            csvfile_gt.seek(0)
            csvfile_pe.seek(0)
            reader_gt = csv.DictReader(csvfile_gt, delimiter=',')
            reader_pe = csv.DictReader(csvfile_pe, delimiter=',')
            
            tag_list = ['total','tcorrect','head','hcorrect','arm','acorrect',
                        'leg','lcorrect']
            count = {}
            for tag in tag_list:
                count[tag] = 0

            correct_count_total,correct_count_arm = 0,0
            correct_count_leg,correct_count_head = 0,0
            for row_gt in reader_gt:
                
                row_gt = dict((k,float(v)) for k,v in row_gt.iteritems())
                row_pe = next(reader_pe)
                row_pe = dict((k,float(v)) for k,v in row_pe.iteritems())

                # Compute head size
                head_size = (math.sqrt( math.pow( (row_gt[header[0]] - \
                                                    row_gt[header[2]]) , 2) \
                                        + math.pow( (row_gt[header[1]] - 
                                                    row_gt[header[3]]) , 2) ))
                
                # For each joint, check if the distance of pe and gt is in the range of half head_size
                for joint_i in range( 0,len(header)/2 ):
                    dist = math.sqrt(math.pow(row_gt[header[joint_i*2]]-\
                                                row_pe[header[joint_i*2]],2)+\
                                    math.pow(row_gt[header[joint_i*2+1]]-\
                                                row_pe[header[joint_i*2+1]],2))
                    if (dist <= head_size * norm_factor):
                        count['tcorrect'] += 1
                        count['hcorrect'] +=1 if joint_i in [0,1] else 0
                        count['acorrect'] +=1 if joint_i in [2,3,4,5,6,7] else 0
                        count['lcorrect'] +=1 if joint_i in [8,9,10,11,12,13] \
                                                                        else 0

            count['total'],count['head'] = 14*row_count,2*row_count
            count['arm'],count['leg'] = 6*row_count,6*row_count
        return count

    @staticmethod
    def mse(sensor_csv_path,tfd_csv_path):
        """Calculate Mean Square Error between sensor frequency data and 
            estimated frequency data.

        Args:
            sensor_csv_path: a string indicating sensor data path, only part of
                             joints will be tested.
            tfd_csv_path: a string indicating estimation data path.

        Return:
            mse_joints: a dictionary, joint name string and cooresponding 
                        frequency float number, for example,{'Lwri':0.5644094}.
        """

        with open(sensor_csv_path,'rb') as csvfile_sensor, \
                open(tfd_csv_path,'rb') as csvfile_tfd:
            # Check the length of two csv files
            sensor_csv_len = sum(1 for row in csvfile_sensor) -1
            assert sensor_csv_len == sum(1 for row in csvfile_tfd)-1

            csvfile_sensor.seek(0)
            csvfile_tfd.seek(0)

            header = csv.reader(csvfile_sensor).next()
            csvfile_sensor.seek(0)

            reader_sensor = csv.DictReader(csvfile_sensor, delimiter=',')
            reader_tfd = csv.DictReader(csvfile_tfd, delimiter=',')

            mse_joints = {}
            for joint in header:
                mse_joints[joint] = 0.0                
            for row_sensor in reader_sensor:
                row_tfd = next(reader_tfd)
                for joint in header:
                    mse_joints[joint] += math.pow( float(row_sensor[joint])-\
                                                    float(row_tfd[joint]), 2)
            for joint in header:
                mse_joints[joint] /= sensor_csv_len
                # print 'MSE for \'{}\': {}'.format(joint,mse_joints[joint])
        return mse_joints