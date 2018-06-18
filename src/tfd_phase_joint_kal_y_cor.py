# this script is used to embed in the kalman tracking for Top_neus tasks
# take in the raw video with full size and process the croped frames at kalman estimated point.
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
import re
from scipy import stats
from collections import deque
import math
from cycler import cycler
from logger import Logger

from filterbank import *
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KFilter():
	def __init__(self,start_pos):
		self.f = KalmanFilter(dim_x=2, dim_z=1)
		self.f.x = start_pos # initial state
		self.f.F = np.array([[1, 0], [0, 1]], np.float32) # state transition matrix A 
		self.f.H = np.array([[1, 0]], np.float32) # Measurement function
		self.f.P *= 0.001 # covariance matrix
		self.f.R = np.array([[1]],np.float32) * 3 # measurement noise, state uncertainty
		# f.Q = Q_discrete_white_noise(dim=2, dt=0.033, var=0.003) # process noise
		self.f.Q = np.array([[1,0],[0,1]],np.float32) * 0.003

	def Estimate(self,coordY):
		''' This function estimates the position of the object'''
		measured = np.array([[np.float32(coordY)]])
		# self.kf.correct(measured)
		# predicted = self.kf.predict()
		# return predicted
		self.f.predict()
		self.f.update(measured)
		return self.f.x

class TFD_PHASE_JOINT_KAL_Y_COR():
	"""
	Tremor Frequency Detector Class. Assume cropped frames, videos and belief maps are already saved.
	Only for cropped frames and video 
	"""

	@staticmethod
	def tfd_batch_phase_clipped(video_path_list,window_size_list,joint_list):
		"""TFD Pipeline batch processing, automatically generate saving path and
			file name,
			outcomes will be placed at '../results/joint_tfd/' directory. 

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
			video_code = video_path.split('/')[5]# 3(T008), 4(All)
			video_name = video_path.split('/')[6]# 4      , 5
			print 'Video {}_{} in process'.format(video_code,video_name)
			result_save_path = '../result/{}_joint_tfd_{}/'.format(video_name,window_size) + video_code +'_tfd/'
			if not os.path.isdir(result_save_path):
				os.makedirs(result_save_path)

			final_freq_csv_path = result_save_path + video_name + '_tfd_freq.csv'

			TFD_PHASE_JOINT_KAL_Y_COR.tremor_freq_detec_phase(video_path,window_size,noverlap,\
										final_freq_csv_path,joint_list[i],use_conf_map=False)

	@staticmethod
	def tremor_freq_detec_phase(video_path,window_size,noverlap, 
							final_freq_csv_path,
							JOINT_LIST,use_conf_map):
		# Constant

		stride = window_size - noverlap # 121- 60(121/2) = 61

		# load conf_maps and predcition_arr from joint_data folder
		video_code = video_path.split('/')[5]
		video_name = video_path.split('/')[6]

		joints_string = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',  
						'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank']
		joints_string = [joints_string[x] for x in [JOINT_LIST[0]]]
		joint_name = joints_string[0]

		# read in the joint data(conf_maps and position estimation) on Silvia's PC
		conf_arr_list = []
		conf_arr_path = '/local/guest/pose_data/results/' + video_code + '_crop' +'/'+video_name+'/'+'conf_arr/' + joint_name + '/'
		conf_arr_fullpath_list = util.get_file_fullpath_list(conf_arr_path,file_format='bin')
		# conf_arr_fullpath_list = util.get_file_fullpath_list(conf_arr_path,file_format='txt')
		conf_arr_fullpath_list = sorted(conf_arr_fullpath_list,key=lambda x: (int(re.sub('\D','',x)),x))

		for conf_arr_index in range(0,len(conf_arr_fullpath_list)):
			conf_arr = np.load(conf_arr_fullpath_list[conf_arr_index])
			conf_arr = np.array(conf_arr)
			if use_conf_map:
				conf_arr_list.append(conf_arr) # frames of elements in the list , each elem is a array
			else:
				conf_arr = np.ones(conf_arr.shape,dtype=float)/sum(sum(conf_arr))
				conf_arr_list.append(conf_arr)
		cpm_joint_path = '/local/guest/pose_data/results/' + video_code + '_crop' +'/'+video_name+'/'+'prediction_arr/'

		pos_arr_list = util.get_jonit_pos_sequence(cpm_joint_path,JOINT_LIST[0],type="cpm")

		# Init a kalman filter object
		start_pos = np.array([[pos_arr_list[0][0]],[0.]])
		kfObj = KFilter(start_pos)

		predictedCoords = np.zeros((1, 1), np.float32) # only Y coordinate

		pos_for_crop = list()
		# Smooth the Y pos for all frames

		for frame_i in range(0,len(pos_arr_list)):
			pred_y = pos_arr_list[frame_i][0]
			pred_x = pos_arr_list[frame_i][1]
			predictedCoords = kfObj.Estimate(pred_y)
			pos_for_crop.append([int(predictedCoords[0][0]),pred_x])

		print 'Trajectory smoothing is done.'

		def fft(video_path,conf_arr_list):

			box_size = 28
			video_fft = Video(video_path)
			io_video_instance = IOVideo(resizing_on=True,scale=368/video_fft.HEIGHT,write_to_video_on=False,\
							video_path = cpm_joint_path+ 'joint_video.avi',fps=5.0,height=box_size*2,width=box_size*2) #368
			# can set videoWriter use IOVideo class
			resized_frames = io_video_instance.get_video_frames(video_fft,int(video_fft.FRAME_COUNT),grayscale_on=True)
			
			# For some tasks the CPM fails and cannot PE for every frame 
			if not (len(pos_arr_list) == video_fft.FRAME_COUNT):
				print 'In {}_{}, the length of pos_arr_list is not consist with the frame_count, skip. '.format(video_code,video_name)

			frames = []
			for frame_i in range(0,len(pos_arr_list)):
				[top,bottom] = [int(pos_for_crop[frame_i][0])-box_size,int(pos_for_crop[frame_i][0])+box_size]
				[left,right] = [int(pos_for_crop[frame_i][1])-box_size,int(pos_for_crop[frame_i][1])+box_size]
				cropped_frame = resized_frames[frame_i][top:bottom, left:right]
				frames.append(cropped_frame)
			frames = np.array(frames)

			# Init FFTM for each joint
			fftm = []
			for joint_i in range(0,13): # 13 imgs
				fftm.append(FFTM(window_size,video_fft.FPS))

			# results
			joint_fft_squences = []
			joint_conf_maps = []
			
			# start number : (0,61,121,182,242,303, ...)
			for i in range(0,int(video_fft.FRAME_COUNT/stride)-1):
			# for i in range(0,int(2)):
				if not ((i+2)%2 == 0): # odd count 
					f_start_number = stride + window_size*(i/2)
				else:					# even count
					f_start_number = window_size*(i/2)

				f_end_number= f_start_number+window_size-1

				fft_logger.info( 'Frame ({}~{})/{} is processing'.format(
								f_start_number, f_end_number,
								int(video_fft.FRAME_COUNT)) )

				conf_maps = np.array(conf_arr_list[f_start_number:f_end_number+1]) # len(0:120) = 120 not 121

				# Crop joint segments from image and send to fftm and get PSD

				cropped_jonit_frames = frames[f_start_number:f_end_number+1,:,:]

				phase_image_top = get_phase_images_real(cropped_jonit_frames,\
													frame_num=len(cropped_jonit_frames),filter_on = False)

				phase_image_top.insert(0,cropped_jonit_frames) # 13 elements

				joint_fft_squences_inner, freq_max_list,joint_conf_maps_inner= [],[],[]

				for j in range(0,len(phase_image_top)): # 13
					cropped = phase_image_top[j]
					fftm[j].add_frames(cropped)

					fft_sequence_ampl,_,_,freq_max_ampl= \
							fftm[j].fft_frames_sequence(filter_on=True,threshold_on=True)

					joint_fft_squences_inner.append(fft_sequence_ampl)
					freq_max_list.append(freq_max_ampl)

					if j < 5:
						conf_maps_cropped = conf_maps
						conf_maps_cropped = conf_maps_cropped / np.sum(conf_maps_cropped) 
						joint_conf_maps_inner.append(conf_maps_cropped)

					elif j < 9:
						conf_maps_cropped = []
						for k in range(0,conf_maps.shape[0]):
							resize_conf_map = cv2.resize(conf_maps[k],(0,0), fx=0.5, fy=0.5,\
														interpolation = cv2.INTER_CUBIC)
							resize_conf_map = resize_conf_map / np.sum(resize_conf_map)
							conf_maps_cropped.append(resize_conf_map)

						joint_conf_maps_inner.append(np.array(conf_maps_cropped))

					else:
						assert j <= 12
						conf_maps_cropped = []
						for k in range(0,conf_maps.shape[0]):
							resize_conf_map = cv2.resize(conf_maps[k],(0,0), fx=0.25, fy=0.25,\
														interpolation = cv2.INTER_CUBIC)
							resize_conf_map = resize_conf_map / np.sum(resize_conf_map)
							conf_maps_cropped.append(resize_conf_map)

						joint_conf_maps_inner.append(np.array(conf_maps_cropped))


				joint_fft_squences.append(joint_fft_squences_inner) # element number = fft_count, each element contains 13 results
				joint_conf_maps.append(joint_conf_maps_inner)

			del io_video_instance
			return joint_fft_squences, joint_conf_maps

		# Init frequency series
		len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
		freq_series = np.fft.fftfreq( window_size, d=1/float(30) )[0:len_half]
		tfd_logger.debug('Frequency Series: {}'.format(freq_series))

		# Init CSV file to save results
		joints_string = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb',  
						'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank']
		joints_string = [joints_string[x] for x in [JOINT_LIST[0]]]
		psd_save_path = os.path.dirname(final_freq_csv_path) + \
															'/accumulated_psd/'
		if not os.path.isdir(psd_save_path):
			os.mkdir(psd_save_path)
		csvfile = open(final_freq_csv_path, 'wb')
		csvwriter = csv.writer(csvfile)

		csvwriter_avgpsd = csv.writer(open(os.path.dirname(final_freq_csv_path)+'/psd_avg.csv', 'wb'))
		csvwriter_ispeak = csv.writer(open(os.path.dirname(final_freq_csv_path) + \
										'/is_peak_overall_{}.csv'.format(joints_string[0]), 'wb'))
		csvwriter_auto_slct = csv.writer(open(os.path.dirname(final_freq_csv_path)+'/auto_slct.csv', 'wb'))
		csvwriter_auto_slct_head = ['J&C','select','freq','fall']
		csvwriter_auto_slct.writerow(csvwriter_auto_slct_head)
		phase_list = ['no_phase','scl_0_ori_0','scl_0_ori_pi/4','scl_0_ori_pi/2','scl_0_ori_3pi/4',\
						'scl_1_ori_0','scl_1_ori_pi/4','scl_1_ori_pi/2','scl_1_ori_3pi/4',\
						'scl_2_ori_0','scl_2_ori_pi/4','scl_2_ori_pi/2','scl_2_ori_3pi/4',]

		# Init TXT file to save freq for calculating absolute error
		txt_folder_path = os.path.dirname(final_freq_csv_path)+ '/freq_psd_txt'
		if not os.path.isdir(txt_folder_path):
			os.mkdir(txt_folder_path)
		freq_result_txt_path = txt_folder_path + '/freq_result.txt'
		psd_txt_path = txt_folder_path + '/psd.txt'
		freq_rgb_txt_path = txt_folder_path + '/freq_rgb.txt'
		psd_rgb_txt_path = txt_folder_path + '/psd_rgb.txt'

		psd_avg = [] # for each phase
		peak_count = np.zeros((13,1),dtype = int)
		for pahse_num in range(0,13):
			psd_avg.append([]) 
		freq_results, psds = [],[]
		# for overall, select psd of a phase with biggest fall in one stride, add to psd_overall, normalized.
		psd_overall = []
		freq_txt = [] # for writing to freq.txt

		fft_sequences,joint_conf_maps = fft(video_path,conf_arr_list)

		freq_rgb, psd_rgb = [], []
		for fft_count in range(len(fft_sequences)): # count level
			freq_results_inner,psds_inner = [],[]

			for phase_index in range(len(fft_sequences[fft_count])): # phase level

				weighted_fft_sequence = fft_sequences[fft_count][phase_index] * \
											 np.average(joint_conf_maps[fft_count][phase_index],axis = 0)
				# window_size/2 *width*height * width*height
				
				weighted_fft_sequence = np.sum(weighted_fft_sequence,axis=(1,2))
				weighted_fft_sequence = weighted_fft_sequence / weighted_fft_sequence.max()

				psds_inner.append(tuple(weighted_fft_sequence))  # phase level
				if psd_avg[phase_index] == []:
					# TODO: if not a peak then not added,and divide count -1
					psd_avg[phase_index] = weighted_fft_sequence
				else:
					psd_avg[phase_index] += weighted_fft_sequence

				plot_to_file(x=freq_series,y=weighted_fft_sequence,
						xlabel='Frequency (Hz)',
						ylabel='Power Spectral Density',
						title='Accumulated PSD of {}, count: {}, phase_pos: {}, time: {:.1f} (s)'.format(
				 									joints_string[0],fft_count,phase_index,
				 									float((fft_count+1)*\
				 									window_size)/2.0/30.0),
				 		save_path=psd_save_path+'apsd_{}_{}_p{}.eps'.format(
				 						joints_string[0],fft_count,phase_index))

				freq_i = freq_series[np.argmax(weighted_fft_sequence)]
				if phase_index ==0:
					row_freq_rgb_txt = [fft_count+1,float(freq_i)]
					freq_rgb.append(row_freq_rgb_txt)
					if psd_rgb == []:
						psd_rgb = weighted_fft_sequence
					else:
						psd_rgb = psd_rgb + weighted_fft_sequence

				freq_results_inner.append(str(freq_i))

				if weighted_fft_sequence.max()> \
											np.mean(weighted_fft_sequence)+\
											3*np.std(weighted_fft_sequence):
					is_peak = 1 

					peak_count[phase_index] += 1
					# for calculate the average psd of specific phase, if not peak, psd not added to the sum
					if psd_avg[phase_index] == []:
						psd_avg[phase_index] = weighted_fft_sequence
					else:
						psd_avg[phase_index] += weighted_fft_sequence

				else:
					is_peak = 0
				freq_results_inner.extend(str(is_peak))

				tfd_logger.debug('P_{},accumulated_psd: {}'.format(phase_index,weighted_fft_sequence))
				tfd_logger.info('{}, count: {}, P_{} Freq: {} Hz'.format(\
												joints_string[0],fft_count,phase_index,freq_i))
			freq_results.append(freq_results_inner)
			psds.append(psds_inner)
			
			assert len(psds) == len(freq_rgb)
			psd_rgb = psd_rgb/psd_rgb.max()

		# writer file freq_rgb.txt, psd_rgb.txt
		peak_rgb = int(psd_rgb.max()> np.mean(psd_rgb)+ 3*np.std(psd_rgb))
		peak_freq_rgb = freq_series[np.argmax(psd_rgb)]
		freq_rgb.append([peak_rgb,peak_freq_rgb])
		np.savetxt(freq_rgb_txt_path,freq_rgb)

		psd_rgb_txt = map(list,zip(*[freq_series,psd_rgb])) # transpose
		np.savetxt(psd_rgb_txt_path,psd_rgb_txt)

		print 'the peak_count for each phase is \n{}'.format(peak_count)

		# Save results to .csv file
		print 'writting results into files : XXX_tfd_freq.csv, X.csv, auto-slct.csv'

		for fft_count in range(len(fft_sequences)): # count level
			# writer file XXX_tfd_freq.csv
			csvwriter_head = []
			for phase_index in range(len(freq_results[fft_count])/2):
				csvwriter_head.extend(['{}_P{}_c{}'.format(joints_string[0],\
														phase_index,fft_count,),'is_peak'])
			csvwriter.writerow(csvwriter_head)
			csvwriter.writerow(freq_results[fft_count])

			# writer file X.csv
			csvwriter_psd = csv.writer(open(psd_save_path+"{}.csv".format(fft_count), 'wb'))
			
			csvwriter_psd_head = []
			for phase_index in range(len(freq_results[fft_count])/2):
				csvwriter_psd_head.extend(['{}_P{}'.format(joints_string[0],phase_index)])
			csvwriter_psd_head.insert(0,'freq')
			csvwriter_psd.writerow(csvwriter_psd_head)
			csvwriter_psd_row = psds[fft_count]
			csvwriter_psd_row.insert(0,freq_series)

			for row in zip(*csvwriter_psd_row):
				csvwriter_psd.writerow(row)
			del csvwriter_psd
			# pop out the freq_series
			pop = psds[fft_count].pop(0)

			# write file auto-slct.csv
			psds_mean = np.mean(psds[fft_count],1)
			psds_max = np.max(psds[fft_count],1)
			psds_fall = psds_max - psds_mean
			# psds_fall = psds_fall[1:]
			
			slct_index = np.int(np.argmax(psds_fall))
			
			csvwriter_auto_slct_row = ['{}_c{}'.format(joints_string[0],fft_count),\
					phase_list[slct_index],freq_results[fft_count][slct_index*2],psds_fall[slct_index]]
			
			csvwriter_auto_slct.writerow(csvwriter_auto_slct_row)

			row_freq_txt = [fft_count+1,float(freq_results[fft_count][slct_index*2]),slct_index]
			freq_txt.append(row_freq_txt)

		del csvwriter, csvwriter_auto_slct

		# write file freq_result.txt and psd.txt
		print 'writting results into files : freq_result.txt and psd.txt'

		psds_overall_mean = np.mean(psds,2)
		psds_overall_max = np.max(psds,2)
		psds_overall_fall = psds_overall_max - psds_overall_mean

		for fft_count in range(0,len(psds_overall_fall)):
			slct_index_in_count = np.int(np.argmax(psds_overall_fall[fft_count]))
			if psd_overall == []:
				psd_overall = np.array(psds[fft_count][slct_index_in_count])
			else:
				psd_overall = psd_overall + np.array(psds[fft_count][slct_index_in_count])

		psd_overall = psd_overall / psd_overall.max()
		is_peak_overall = int(psd_overall.max()> psd_overall.mean() + psd_overall.std())
		freq_overall = freq_series[np.argmax(psd_overall)]
		freq_txt.append([is_peak_overall,freq_overall,freq_overall])
		psd_txt = map(list,zip(*[freq_series,psd_overall])) # transpose
		np.savetxt(psd_txt_path,psd_txt)
		np.savetxt(freq_result_txt_path,freq_txt)

		# plot psd_avg in all scale and phase levels
		for phase_index in range(len(psd_avg)):
			if peak_count[phase_index] > 0:
				psd_avg[phase_index] = psd_avg[phase_index]/psd_avg[phase_index].max()  # normalization
				plot_to_file(x=freq_series,y=np.array(psd_avg[phase_index]),
							xlabel='Frequency (Hz)',ylabel='Power Spectral Density',
							title=' Average PSD of {}_P_{}'.format( \
															joints_string[0],phase_index),
							save_path=psd_save_path+'avgpsd_{}_P_{}.eps'.format( \
															joints_string[0],phase_index))
			else:
				pass

		# plot psd_avg in diff scale levels

		# if not peak in all strides, use 0 array as complement.
		complement_arr = np.zeros(freq_series.shape,dtype= float)

		for scale in range(3):
			phase_index = list(np.array((1,2,3,4))+scale*4)
			phase_index.insert(0,0) # use no_phase line as comparison
			psd_avg_scale_level = []
			for index in phase_index:
				if peak_count[index]>0:
					psd_avg_scale_level.append(psd_avg[index]) # this sum now, take average in plot by div total fft_count
				else:
					psd_avg_scale_level.append(complement_arr)

			legend = ['no_phase','scl{}_ori_0'.format(scale),'scl{}_ori_pi/4'.format(scale),\
						'scl{}_ori_pi/2'.format(scale),'scl{}_ori_3pi/4'.format(scale)]
			n_plot_to_file(run_time = peak_count[phase_index],x=freq_series,y_list=psd_avg_scale_level,\
							xlabel='Frequency (Hz)', ylabel='Power Spectral Density',\
							legends = legend,title=' Average PSD of {}_scl{}'.format( \
														joints_string[0],scale),\
							plot_path=psd_save_path+'avgpsd_{}_scl{}.eps'.format( \
														joints_string[0],scale))

		# write file is_peak_overall.csv
		print 'writting results into file : is_peak_overall_{}.csv'.format(joints_string[0])
		csvwriter_ispeak_head =['Phase', 'is_peak', 'freq', 'fall']
		csvwriter_ispeak.writerow(csvwriter_ispeak_head)

	   	for phase_index in range(len(psd_avg)):
	   		if peak_count[phase_index]>0:
		   		if psd_avg[phase_index].max()>np.mean(psd_avg[phase_index])+3*np.std(psd_avg[phase_index]):
					is_peak = 1
				else:
					is_peak = 0

				slct_index = np.int(np.argmax(psd_avg[phase_index]))

				psd_avg_mean = np.mean(psd_avg[phase_index])
				psd_avg_max = np.max(psd_avg[phase_index])
				psd_avg_fall = psd_avg_max - psd_avg_mean

		   		csvwriter_ispeak_row = [phase_list[phase_index],is_peak,freq_series[slct_index],psd_avg_fall]
				csvwriter_ispeak.writerow(csvwriter_ispeak_row)
			else:
				pass
		del csvwriter_ispeak

		# write file psd_avg.csv
		print 'writting results into file : psd_avg.csv'
		psd_avg_print = []
		for psd in psd_avg:
			if not (psd == []):
				psd_avg_print.append(tuple(psd))
			else:
				pass
		psd_avg = psd_avg_print

		# psd_avg = [ tuple(psd/len(fft_sequences)) for psd in psd_avg]
		csvwriter_avgpsd_head = []
		for phase_index in range(len(psd_avg)):
			if peak_count[phase_index]>0:
				csvwriter_avgpsd_head.extend(['{}_P{}'.format(joints_string[0],phase_index)])
			else:
				pass
		csvwriter_avgpsd_head.insert(0,'freq')
		csvwriter_avgpsd.writerow(csvwriter_avgpsd_head)
		psd_avg.insert(0,tuple(freq_series))
		for row in zip(*psd_avg):
			csvwriter_avgpsd.writerow(row)
		
		del csvwriter_avgpsd
		csvfile.close()
		

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
		# joint_frame = cv2.GaussianBlur(joint_frame,(0,0),5)
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

def n_plot_to_file(run_time,x,y_list,xlabel,ylabel,legends,title,plot_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(cycler('color', ['b', 'r']))
    lenth = len(y_list)

    for i in range(0,lenth):
        values = y_list[i]
        # y = np.array(values)/run_time
        y = np.array(values)
        # y = y/y.max()
        ax.plot(x,y,label = 'i')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legends)
    fig.savefig(plot_path, format='eps')
    plt.close(fig)


def get_phase_images_real(frames,frame_num,filter_on = False):
	"""creat phase images on joint
		phase pyramid shape is 3 scales * 4 orientation

	Args:
		frames: original joint frames .
		io_video: an IO_Video object.
		frame_num: an integer indicating how many frames to extract.
		filter_on: an boolean indicating whether implement filter on ori_frames.
	Return:
		phase_image_top: a list with array-type elements, depend on the shape of phase pyramid, default is 12 elments 
	"""
	phase_image_top = []

	frames.astype(np.float32)  # easy for FFT
	assert frames.shape[1] == frames.shape[2]
	frame_size = frames.shape[1]

	# get pyramid shape
	steer = Steerable()
	frame_exmp = frames[1,:,:]
	coeff_exmp = steer.buildSCFpyr(frame_exmp)
	num_scale = len(coeff_exmp)     # 5
	num_orient  = len(coeff_exmp[1])  # 4
	# NOTE: the first and last element of coeff is a array instead of a list (low-pass part and high-pass part)

	size = [frame_size,frame_size,frame_size/2,frame_size/4,frame_size/4]

	for i in range(1,num_scale-1):
		for j in range (0,num_orient):
			phase_image_bottom = []
			for k in range (0,frame_num):
				frame = frames[k,:,:]
				if filter_on == True:
					frame = cv2.GaussianBlur(frame,(5,5),0)
					# frame = cv2.bilateralFilter(frame,-1,5,5)
				coeff = steer.buildSCFpyr(frame)
				phase_emp = coeff[i][j]
				hight = phase_emp.shape[0]
				width = phase_emp.shape[1]
				angles = np.angle(phase_emp)
				amp = 1*np.ones((hight,width),dtype = int)
				phase_cos = amp * np.exp(angles * 1j).real
				phase_image = np.uint8(cv2.convertScaleAbs((phase_cos+1.0),alpha=255.0/2.0))
				phase_image_bottom.append(phase_image)
			phase_image_bottom = np.array(phase_image_bottom)
			phase_image_top.append(phase_image_bottom) # creat a list with array_type elements: [[],[],[],..,[]], total 12

	return phase_image_top


if __name__ == "__main__":

	tfd_phase_joint_kal_y_cor = TFD_PHASE_JOINT_KAL_Y_COR()
	level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'
	fft_logger = Logger('fft_logger',level_name)
	tfd_logger = Logger('tfd_logger',level_name)
	
	task_to_process = 'Spiraal_links'
	folders = util.get_full_path_under_folder('/media/tremor-data/TremorData_split/Tremor_data/')
	folders = sorted(folders,key=lambda x: (int(re.sub('\D','',x)),x))
	video_path_list ,window_size_list, joint_list = [],[],[]
	for i in range(0,len(folders)):
		video_path = folders[i]+ '{}/'.format(task_to_process) + 'kinect.avi'

		if "Rechts" in folders[i]:
			if os.path.isfile(video_path):
				video_path_list.append(video_path)
				joint_list.append([4])
				window_size_list.append(61)
			else:
				pass
		else:
			if os.path.isfile(video_path):
				video_path_list.append(video_path)
				joint_list.append([7])
				window_size_list.append(61)
			else:
				pass

	tfd_phase_joint_kal_y_cor.tfd_batch_phase_clipped(video_path_list,window_size_list,joint_list)
