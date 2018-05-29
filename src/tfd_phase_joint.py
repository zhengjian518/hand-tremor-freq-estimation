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

from filterbank import *

class TFD_PHASE_JOINT():
	"""
	Tremor Frequency Detector Class. Assume cropped frames and videos are already saved.
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
			video_code = video_path.split('/')[3]# 3(T008), 4(All)
			video_name = video_path.split('/')[4]# 4      , 5
			print 'Video {}_{} in process'.format(video_code,video_name)
			result_save_path = '../results/' +'joint_tfd/' video_code +'/'+ video_name +'/'
			if not os.path.isdir(result_save_path):
				os.makedirs(result_save_path)

			fft_video_path = result_save_path + video_name + '_freq_heatmap.avi'
			final_freq_csv_path = result_save_path + video_name + '_tfd_freq.csv'

			TFD_PHASE_JOINT.tremor_freq_detec_phase(video_path,window_size,noverlap,\
										fft_video_path,final_freq_csv_path,joint_list)

	@staticmethod
	def tremor_freq_detec_phase(video_path,window_size,noverlap, 
							fft_video_path,final_freq_csv_path,
							JOINT_LIST = [7]):
		# Constant
		FREQ_SHOW_MAX = 10
		JOINTS_NUM = 14
		stride = window_size - noverlap # 121- 60(121/2) = 61

		level_name = sys.argv[1] if len(sys.argv) > 1 else 'info'


		# load conf_maps and predcition_arr from joint_data folder

		video_code = video_path.split('/')[3]
		video_name = video_path.split('/')[4]

		conf_arr_list = []

		conf_arr_path = '../results/joint_data/'+video_code+'/'+video_name+'/'+'conf_arr/'
		conf_arr_fullpath_list = util.get_file_fullpath_list(conf_arr_path,file_format='txt')
		# TODO: to check whether sorted in order
		# may use 'sorted(path_list,key=lambda x: (int(re.sub('\D','',x)),x))'
		for conf_arr_index in range(0,len(conf_arr_fullpath_list)):
			conf_arr = np.genfromtxt(conf_arr_fullpath_list[conf_arr_index],dtype=None)
			conf_arr = [list(i) for i in conf_arr]
			conf_arr_list.append(np.array(conf_arr))

		# prediction_arr_path = '../results/joint_data/'+video_code+'/'+video_name+'/'+'prediction_arr/'
		# prediction_arr_fullpath_list = util.get_file_fullpath_list(prediction_arr_path,file_format='txt')

		# for prediction_arr_index in range(0,len(prediction_arr_fullpath_list)):
		# 	prediction_arr = np.genfromtxt(prediction_arr_fullpath_list[prediction_arr_index],dtype=None)
		# 	prediction_arr = [list(i) for i in prediction_arr]
		# 	prediction_arr_list.append(np.array(prediction_arr))


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

				# # Step 3: Label pose on image and save to file 
				# pe_save_path = os.path.dirname(final_freq_csv_path)+'/pe/'
				# if not os.path.isdir(pe_save_path):
				# 	os.mkdir(pe_save_path)

				# for joint_i in JOINT_LIST:
				# 	pred_y = int(prediction[joint_i,0])
				# 	pred_x = int(prediction[joint_i,1])
				# 	frame_pe = cv2.circle(frame_pe, (pred_x, pred_y), 3, 
				# 												(0,0,255), -1)  # red spot in image
				# cv2.imwrite(pe_save_path+'{}.png'.format(i), frame_pe)
				# np.savetxt(pe_save_path+'pred_{}.txt'.format(i),prediction)

		def fft(joint_pred_queue,conf_map_queue,fft_sequence_queue,
				joint_conf_map_queue,lock,):
							
			fft_logger = Logger('fft_logger',level_name)

			# Init videos
			# video_background = Video(video_path) # Visualization
			video_fft = Video(video_path)
			io_video_instance = IOVideo()

			frames = io_video_instance.get_video_frames(video_fft,\
			video_fft.FRAME_COUNT,grayscale_on=True)

			# io_video = IOVideo(resizing_on=False,scale=368/video_fft.HEIGHT,
			# 		write_to_video_on=True,video_path=fft_video_path,fps=25,
			# 		height=368,width=368*video_fft.WIDTH/video_fft.HEIGHT)
			
			# Init FFTM for each joint
			fftm = []
			for joint_i in range(JOINTS_NUM):
				fftm.append(FFTM(window_size,video_fft.FPS))
			
			# box_size = None
			stride_count = stride-1  # 61-1
			
			for i in range(0,int(video_fft.FRAME_COUNT/stride)-1):
				fft_logger.info( 'Frame ({}~{})/{} is being processed'.format(
								i*stride, i*stride+window_size-1,
								int(video_fft.FRAME_COUNT)) )

				joint_fft_squences = []
				joint_conf_maps = []
				# joint_freq_max = []
				# freq_map = np.zeros( (368,
				# 					int(368*video_fft.WIDTH/video_fft.HEIGHT)) )

				# Step 1: Get pe prediction and init box size with first 
				#           estimation.
				# joint_preds =  joint_pred_queue.get()
				# conf_maps = conf_map_queue.get()

				conf_maps = []
				for conf_num in range(0,stride):
					conf_maps.append(conf_arr_list[conf_num+ fft_count * stride]) # 61 elements


				# if box_size is None:
				#     box_size = math.sqrt( 
				#                 math.pow(joint_preds[0,0]-joint_preds[1,0],2)+ \
				#                 math.pow(joint_preds[0,1]-joint_preds[1,1],2) )
				#     box_size = int(box_size/2)

				# TODO: how to find a proper box size can be devided by 4
				# if box_size is None:
				# 	box_size = 44
				# 	box_size = int(box_size/2)

				# Step 2: Crop joint segments from image and send to fftm 
				#           and get PSD,
				for joint_i in JOINT_LIST:# range(JOINTS_NUM):
					# TODO: may have bug - box is out of image
					# pred_y = int(joint_preds[joint_i,0])
					# pred_x = int(joint_preds[joint_i,1])
					# cropped_jonit_frames = get_cropped_frames(video_fft,io_video, 
					# 						No_start=i*stride,frame_num=window_size, 
					# 						pred_x=pred_x,pred_y=pred_y,
					# 						box_size=box_size) # grayscale on

					cropped_jonit_frames = frames[fft_count*stride:fft_count*stride+window_size,:,:]

					phase_image_top = get_phase_images_real(cropped_jonit_frames,\
										frame_num=window_size,filter_on = False)

					phase_image_top.insert(0,cropped_jonit_frames) # 13 elements

					joint_fft_squences_inner, freq_max_list,joint_conf_maps_inner= [],[],[]

					for j in range(0,len(phase_image_top)): # 13
						fftm[joint_i] = FFTM(window_size,video_fft.FPS)
						cropped = phase_image_top[j]
						fftm[joint_i].add_frames(cropped)

						fft_sequence_ampl,_,_,freq_max_ampl= \
								fftm[joint_i].fft_frames_sequence(filter_on=True,
								threshold_on=True)
						joint_fft_squences_inner.append(fft_sequence_ampl)
						freq_max_list.append(freq_max_ampl)

						if j < 5:
							conf_maps_cropped = conf_maps
							conf_maps_cropped = conf_maps_cropped / np.sum(conf_maps_cropped) 
							joint_conf_maps_inner.append(conf_maps_cropped)

						elif j < 9:
							conf_maps_cropped = cv2.resize(conf_maps,(0,0), fx=0.5, fy=0.5,\
															interpolation = cv2.INTER_CUBIC)
							conf_maps_cropped = conf_maps_cropped / np.sum(conf_maps_cropped) 
							joint_conf_maps_inner.append(conf_maps_cropped)

						else:
							assert j <= 12
							conf_maps_cropped = cv2.resize(conf_maps,(0,0), fx=0.25, fy=0.25,\
															interpolation = cv2.INTER_CUBIC)
							conf_maps_cropped = conf_maps_cropped / np.sum(conf_maps_cropped) 
							joint_conf_maps_inner.append(conf_maps_cropped)


					joint_fft_squences.append(joint_fft_squences_inner)
					joint_conf_maps.append(joint_conf_maps_inner)

					# Visualization
					# freq_map[pred_y-box_size:pred_y+box_size,\
					# 			pred_x-box_size:pred_x+box_size] = freq_max_list[0]

				# # Step 3: Save conf map and fft results to queue
				# lock.acquire()
				# joint_conf_map_queue.put(joint_conf_maps)
				# fft_sequence_queue.put(joint_fft_squences)
				# lock.release()

				# # Step 4: Save to Video for Visualization
				# if i==0:
				# 	start_No = i*stride
				# 	frame_to_take_num = window_size*3/4 + 1
				# elif i==int(video_fft.FRAME_COUNT/stride)-2:
				# 	start_No = i*stride+window_size/4
				# 	frame_to_take_num = window_size*3/4 + 1
				# else:
				# 	start_No = i*stride+window_size/4
				# 	frame_to_take_num = window_size/2 + 1
				# video_background.set_next_frame_index(start_No)
				# print("start from:{}, frame_num:{}".format(start_No,
				#                                            frame_to_take_num))
				# freq_map = 0.5*util.colorize(np.divide(freq_map,FREQ_SHOW_MAX))
				# for frame_No in range(frame_to_take_num):
				# 	frame_background = io_video.get_video_frames(
				# 						video_background,1,grayscale_on=False)
				# 	frame_to_save = freq_map + 0.5*frame_background
				# 	io_video.write_frame_to_video( np.uint8(frame_to_save) )

			del io_video



		tfd_logger = Logger('tfd_logger',level_name)

		# Init PE,FFT process
		# lock1  = multiprocessing.Lock()
		# lock2  = multiprocessing.Lock()
		# conf_map_queue = multiprocessing.Queue(1)
		# joint_pred_queue = multiprocessing.Queue(1)
		# fft_sequence_queue = multiprocessing.Queue(1)
		# joint_conf_map_queue = multiprocessing.Queue(window_size/2+1)
		# pe_process = multiprocessing.Process(target=pe, args=(conf_map_queue,
		# 										joint_pred_queue,lock1,))
		# fft_process = multiprocessing.Process(target=fft,args=(joint_pred_queue, 
		# 										conf_map_queue, 
		# 										fft_sequence_queue, 
		# 										joint_conf_map_queue,lock2,))
		# pe_process.start()
		# fft_process.start()

		# Init frequency series
		len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
		freq_series = np.fft.fftfreq( window_size, d=1/float(30) )[0:len_half]
		# freq_series = np.fft.fftfreq( window_size, d=1/float(25) )[0:len_half]
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

		csvwriter_avgpsd = csv.writer(open(psd_save_path+'psd_avg.csv', 'wb'))
		csvwriter_ispeak = csv.writer(open(os.path.dirname(final_freq_csv_path) + \
										'/is_peak_overall.csv', 'wb'))
		csvwriter_auto_slct = csv.writer(open(psd_save_path+'auto_slct.csv', 'wb'))
		csvwriter_auto_slct_head = ['J&C','select','freq','fall']
		csvwriter_auto_slct.writerow(csvwriter_auto_slct_head)
		phase_list = ['no_phase','scl_0_ori_0','scl_0_ori_pi/4','scl_0_ori_pi/2','scl_0_ori_3pi/4',\
						'scl_1_ori_0','scl_1_ori_pi/4','scl_1_ori_pi/2','scl_1_ori_3pi/4',\
						'scl_2_ori_0','scl_2_ori_pi/4','scl_2_ori_pi/2','scl_2_ori_3pi/4',]


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
				
				# Step 2: Compute weighted psd, detect peak frequency and save 
				#         psd to file

				for joint_i in range(len(fft_sequences)): # joint level
					psd_avg_inner,freq_results_inner,psds_inner = [],[],[]

					for phase_index in range(len(fft_sequences[joint_i])): # phase level

						weighted_fft_sequence = fft_sequences[joint_i][phase_index] * \
													 joint_conf_maps[joint_i][phase_index] 
						# window_size/2 *width*height * width*height
						weighted_fft_sequence = np.sum(weighted_fft_sequence,
																		axis=(1,2))
						weighted_fft_sequence = weighted_fft_sequence / \
														 weighted_fft_sequence.max()
						psds_inner.append(tuple(weighted_fft_sequence))

						if(len(psd_avg)<len(fft_sequences)):
							psd_avg_inner.append(weighted_fft_sequence)
						else:
							psd_avg[joint_i][phase_index] += weighted_fft_sequence

						
						plot_to_file(x=freq_series,y=weighted_fft_sequence,
								xlabel='Frequency (Hz)',
								ylabel='Power Spectral Density',
								title='Accumulated PSD for {},phase_pos: {}, time: {:.1f} (s)'.format(
															joints_string[joint_i],phase_index,
															float((fft_count+1)*\
															window_size)/2.0/25.0),
								save_path=psd_save_path+'apsd_{}_{}_p{}.eps'.format(
												joints_string[joint_i],fft_count,phase_index))

						freq_i = freq_series[np.argmax(weighted_fft_sequence)]


						freq_results_inner.append(str(freq_i))

						if weighted_fft_sequence.max()> \
													np.mean(weighted_fft_sequence)+\
													3*np.std(weighted_fft_sequence):
							is_peak = 1 
						else:
							is_peak = 0
						freq_results_inner.extend(str(is_peak))

						tfd_logger.debug('P_{},accumulated_psd: {}'.format(phase_index,weighted_fft_sequence))
						tfd_logger.info('{} P_{} Freq: {} Hz'.format(\
														joints_string[joint_i],phase_index,freq_i))

					psd_avg.append(psd_avg_inner)
					freq_results.append(freq_results_inner)
					psds.append(psds_inner)

				# Step 3: Save results to .csv file
				for joint_i in range(len(joints_string)): # joint level
					# writer file XXX_tfd_freq.csv
					csvwriter_head = []
					for phase_index in range(len(freq_results[joint_i])/2):
						csvwriter_head.extend(['{}_P{}_c{}'.format(joints_string[joint_i],\
																phase_index,fft_count,),'is_peak'])
					csvwriter.writerow(csvwriter_head)
					csvwriter.writerow(freq_results[joint_i])

					# writer file X.csv
					csvwriter_psd_head = []
					for phase_index in range(len(freq_results[joint_i])/2):
						csvwriter_psd_head.extend(['{}_P{}'.format(joints_string[joint_i],phase_index)])
					csvwriter_psd_head.insert(0,'freq')
					csvwriter_psd.writerow(csvwriter_psd_head)
					csvwriter_psd_row = psds[joint_i]
					csvwriter_psd_row.insert(0,freq_series)
					for row in zip(*csvwriter_psd_row):
						csvwriter_psd.writerow(row)

					# write file auto-slct.csv
					psds_mean = np.mean(psds[joint_i],1)
					psds_max = np.max(psds[joint_i],1)
					psds_fall = psds_max - psds_mean
					psds_fall = psds_fall[1:]
					print psds_fall
					slct_index = np.int(np.argmax(psds_fall))
					print slct_index
					csvwriter_auto_slct_row = [joints_string[joint_i]+'_c{}'.format(fft_count),\
							phase_list[slct_index],freq_results[joint_i][slct_index*2],psds_fall[slct_index]]
					print csvwriter_auto_slct_row
					csvwriter_auto_slct.writerow(csvwriter_auto_slct_row)
		
				del csvwriter_psd
				fft_count+=1

		# plot psd_avg in all scale adn phase levels
		for joint_i in range(len(joints_string)):
			for phase_index in range(len(psd_avg[joint_i])):
				plot_to_file(x=freq_series,y=psd_avg[joint_i][phase_index]/fft_count,
							xlabel='Frequency (Hz)',ylabel='Power Spectral Density',
							title=' Average PSD for {}_P_{}'.format( \
															joints_string[joint_i],phase_index),
							save_path=psd_save_path+'avgpsd_{}_P_{}.eps'.format( \
															joints_string[joint_i],phase_index))

		# plot psd_avg in diff scale levels
		for joint_i in range(len(joints_string)):
			for scale in range(3):
				phase_index = list(np.array((1,2,3,4))+scale*4)
				phase_index.insert(0,0)
				psd_avg_scale_level = []
				for index in phase_index:
					psd_avg_scale_level.append(psd_avg[joint_i][index]) # this sum now, take average in plot by div fft_count
				legend = ['no_phase','scl{}_ori_0'.format(scale),'scl{}_ori_pi/4'.format(scale),\
							'scl{}_ori_pi/2'.format(scale),'scl{}_ori_3pi/4'.format(scale)]
				n_plot_to_file(run_time = fft_count,x=freq_series,y_list=psd_avg_scale_level,\
								xlabel='Frequency (Hz)', ylabel='Power Spectral Density',\
								legends = legend,title=' Average PSD for {}_scl{}'.format( \
															joints_string[joint_i],scale),\
								plot_path=psd_save_path+'avgpsd_{}_scl{}.eps'.format( \
															joints_string[joint_i],scale))

	   	for joint_i in range(len(joints_string)):
	   		csvwriter_ispeak_head =[]
	   		is_peak = []
	   		for psd_phase in psd_avg[joint_i]:
	   			csvwriter_ispeak_head.extend(['{}_P{}'.format(joints_string[joint_i],phase_index)])
				if psd_phase.max()>np.mean(psd_phase)+3*np.std(psd_phase):
					is_peak.append(1)
				else:
					is_peak.append(0)
			assert len(csvwriter_ispeak_head)==len(is_peak)
			csvwriter_ispeak.writerow(csvwriter_ispeak_head)
			csvwriter_ispeak.writerow(is_peak)

		del csvwriter_ispeak

		for joint_i in range(len(joints_string)):
			psd_avg[joint_i] = [ tuple(psd/fft_count) for psd in psd_avg[joint_i] ]
			csvwriter_avgpsd_head = []
			for phase_index in range(len(psd_avg[joint_i])):
				csvwriter_avgpsd_head.extend(['{}_P{}'.format(joints_string[joint_i],phase_index)])
			csvwriter_avgpsd_head.insert(0,'freq')
			csvwriter_avgpsd.writerow(csvwriter_avgpsd_head)
			psd_avg[joint_i].insert(0,tuple(freq_series))
			for row in zip(*psd_avg[joint_i]):
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

def n_plot_to_file(run_time,x,y_list,xlabel,ylabel,legends,title,plot_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(cycler('color', ['b', 'r']))
    lenth = len(y_list)
    for i in range(0,lenth):
        values = y_list[i]
        y = np.array(values)/run_time
        ax.plot(x,y,label = 'i')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legends)
    fig.savefig(plot_path, format='eps')
    plt.close(fig)


def get_phase_images_real(frames,frame_num,filter_on = False):
	"""creat phase images on joint
		phase pyramid shape is 3 level * 4 orientation

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
	# NOTE: the first and last element of coeff is a array instead of a list (low-pass and high-pass)

	size = [frame_size,frame_size,frame_size/2,frame_size/4,frame_size/4]
	# need to change for video in different scales, [1,1,1/2,1/4,1/4] * original_size

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
				# write_path =save_path + '{}.png'.format(k)
				# cv2.imwrite(write_path,phase_image)
				# out_video.write(phase_image)
				phase_image_bottom.append(phase_image)
			phase_image_bottom = np.array(phase_image_bottom)
			phase_image_top.append(phase_image_bottom) # creat a list with list_type elements: [[],[],[],..,[]], total 12

			# out_video.release()
	return phase_image_top