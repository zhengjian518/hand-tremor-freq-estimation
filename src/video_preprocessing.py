import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy import signal
from cycler import cycler
import os

from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
from filterbank import *


class Video_Preprocessing():
	"""Class for video_preprocessing.
		the first edition only for test the synthetic ball video , which is analyzied in a frame-based method
		then the next edition will directily process video and creat new videos or be embedded into the frame 
		process in real body process
	Attributes:
		video_path: a string of video path
		filter_on: indicate whether use butter filter
		norm_mode: indicate the normolization way (or will turn to process_way
					-- noise remove -- norm = 0 for white noise spectrum) 
	TODO:
		add more Attributes like :Make_video_or_frames
	"""
	@staticmethod
	def video_preprocessing(video_path, filter_on = False, norm_mode = 'opencv_default',debug_type = 'dis_and_noise'):
		# make instances
		video_to_preprocess = Video(video_path)
		io_video_instance = IOVideo()
		
		# set video parameters 
		frame_number = int(video_to_preprocess.FRAME_COUNT)
		# frame_number = 10
		video_width = int(video_to_preprocess.WIDTH)
		video_height = int(video_to_preprocess.HEIGHT)

		# set VideoWriter
		fourcc = cv2.VideoWriter_fourcc(*'XVID')

		# TODO
		if debug_type =='dis_and_noise':
			out_video = cv2.VideoWriter('../results/JETcolor_ball/'+norm_mode+'filter_off_colorball.avi',fourcc, video_to_preprocess.FPS, 
												(video_width,video_height),isColor = False)
			path = video_path.split('/')[0]+'/'+video_path.split('/')[1]+'/'+video_path.split('/')[2]+\
					'/'+ video_path.split('/')[3]+'/'+ video_path.split('/')[4]+'/' #  path which contains original video

			frames_save_path = path + 'precessed_frames/'
			if not os.path.isdir(frames_save_path):
				os.mkdir(frames_save_path)

			out_video = cv2.VideoWriter(frames_save_path+ 'preprocessed_color_ball.avi',fourcc, video_to_preprocess.FPS, 
												(video_width,video_height),isColor = False)

		elif debug_type == 'norm_mode_debug':
			frames_save_path = video_path
			path = video_path.split('/')[0]+'/'+video_path.split('/')[1]+'/'+video_path.split('/')[2]+ '/'

		SAMPLE_FREQ = video_to_preprocess.FPS

		# # read in grayscale images

		frames = io_video_instance.get_video_frames(video_to_preprocess,\
			frame_number,grayscale_on=True)
		frames.astype(np.float32)  # easy for FFT

		# creat an array to store normalized frames 
		frames_in_gray = np.ndarray((frame_number,video_height,video_width),dtype=np.uint8)

		frames_in_gray_norm = np.subtract(frames,np.mean(frames, axis=0))

		# set normlization mode
		# FFT
		if norm_mode == 'opencv_default':
			frames_in_gray_norm = np.fft.fft(frames_in_gray_norm,axis=0,norm = 'ortho')
			frames_in_gray_norm = np.fft.ifft(frames_in_gray_norm, axis=0)

		elif norm_mode == 'noise_remove':
			frames_in_gray_norm = np.fft.fft(frames_in_gray_norm,axis=0) # complex
			power_pixel_fre = np.power(np.abs(frames_in_gray_norm),2)
			power_sum = np.sum(power_pixel_fre,axis = 0)
			threshold = 0.01*power_sum/frame_number

			mask = power_pixel_fre>threshold
			frames_in_gray_norm = frames_in_gray_norm*mask

			ampl = np.absolute(frames_in_gray_norm)
			frames_in_gray_norm = np.true_divide(frames_in_gray_norm,ampl)
			frames_in_gray_norm = np.fft.ifft(frames_in_gray_norm, axis=0)

		elif norm_mode == 'normalization':
			frames_in_gray_norm = np.fft.fft(frames_in_gray_norm,axis=0) # complex
			ampl = np.absolute(frames_in_gray_norm)
			frames_in_gray_norm = np.true_divide(frames_in_gray_norm,ampl)
			frames_in_gray_norm = np.fft.ifft(frames_in_gray_norm, axis=0)


		# phase images

		# make pixel value 0-255
		frames_in_gray_norm = frames_in_gray_norm.real
		min_pixel = np.min(frames_in_gray_norm,axis = 0)
		frames_in_gray_norm -= min_pixel
		frames_in_gray_norm *= 255.0/frames_in_gray_norm.max(axis = 0)
		frames_in_gray = frames_in_gray_norm.astype(np.uint8)

		# make imgs and video
		for i in range(frame_number):
			single_img_GRAY = frames_in_gray[i,:,:]
			cv2.imwrite(frames_save_path+'{}.png'.format(i),single_img_GRAY)
			out_video.write(np.uint8(single_img_GRAY))

	@staticmethod
	def get_phase_images(video_path,ball_path,dType,mag):

		video_to_preprocess = Video(video_path)
		io_video_instance = IOVideo()

		# set video parameters 
		frame_number = int(video_to_preprocess.FRAME_COUNT)
		# frame_number = 10
		video_width = int(video_to_preprocess.WIDTH)
		video_height = int(video_to_preprocess.HEIGHT)
		fps = video_to_preprocess.FPS
		
		frames = io_video_instance.get_video_frames(video_to_preprocess,\
			frame_number,grayscale_on=True)
		frames.astype(np.float32)  # easy for FFT

		# set video writer 
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# out_video = cv2.VideoWriter(phaseImg_path+ 'phase_ball.avi',fourcc,\
		# 				 video_to_preprocess.FPS,(video_width,video_height),isColor = False)
		if dType == 'ball_phase_test_disturb':
			phasesImg_path = '../results/error_analysis/phase_ball_fps30.0_disturbmag/'
		elif dType == 'ball_phase_test_noise':
			phasesImg_path = '../results/error_analysis/phase_ball_fps30.0_noise/'
		else:
			print 'dType error!!!'

		if not os.path.isdir(phasesImg_path):
				os.mkdir(phasesImg_path)

		steer = Steerable()

		# get pyramid shape
		frame_exmp = frames[1,:,:]
		coeff_exmp = steer.buildSCFpyr(frame_exmp)
		num_scale = len(coeff_exmp)     # 5
		num_orient  = len(coeff_exmp[1])  # 4
		# NOTE: the first and last element of coeff is a array instead of a list

		mag = mag
		if dType == 'ball_phase_test_disturb':
			ori_scl_path = phasesImg_path + 'disturb_{}/'.format(mag)
		elif dType == 'ball_phase_test_noise':
			ori_scl_path = phasesImg_path + 'noise_{}/'.format(mag*0.1)
		else:
			print 'dType error!!!'

		if not os.path.isdir(ori_scl_path):
			os.mkdir(ori_scl_path)

		size = [32,32,16,8,8]
		for i in range(1,num_scale-1):
			for j in range (0,num_orient):
				save_path = ori_scl_path + 'scale_{}'.format(i)+'_ori_{}/'.format(j)

				if not os.path.isdir(save_path):
					os.mkdir(save_path)

				# print 'making the {}_th'.format(i) + ' scale {}_th orient scale phaseimages'.format(j)

				out_video = cv2.VideoWriter(save_path+ 'phase_ball.avi',fourcc,\
						 video_to_preprocess.FPS,(size[i],size[i]),isColor = False)

				for k in range (0,frame_number):
					frame = frames[k,:,:]
					frame = cv2.GaussianBlur(frame,(5,5),0)
					# frame = cv2.bilateralFilter(frame,-1,5,5)
					coeff = steer.buildSCFpyr(frame)
					phase_emp = coeff[i][j]
					hight = phase_emp.shape[0]
					width = phase_emp.shape[1]
					angles = np.angle(phase_emp)
					amp = 1*np.ones((hight,width),dtype = int)
					phase_image = amp * np.exp(angles * 1j).real

					phase_image = np.uint8(cv2.convertScaleAbs((phase_image+1.0),alpha=255.0/2.0))
					write_path =save_path + '{}.png'.format(k)
					cv2.imwrite(write_path,phase_image)
					out_video.write(phase_image)

				out_video.release()
		return ori_scl_path

	@staticmethod
	def get_phase_images_real(video_path,results_save_path,filter_on = False):
		video_to_preprocess = Video(video_path)
		io_video_instance = IOVideo()

		# set video parameters 
		frame_number = int(video_to_preprocess.FRAME_COUNT)
		video_width = int(video_to_preprocess.WIDTH)
		video_height = int(video_to_preprocess.HEIGHT)
		fps = video_to_preprocess.FPS
		
		frames = io_video_instance.get_video_frames(video_to_preprocess,\
			frame_number,grayscale_on=True)
		frames.astype(np.float32)  # easy for FFT

		# get pyramid shape
		steer = Steerable()
		frame_exmp = frames[1,:,:]
		coeff_exmp = steer.buildSCFpyr(frame_exmp)
		num_scale = len(coeff_exmp)     # 5
		num_orient  = len(coeff_exmp[1])  # 4
		# NOTE: the first and last element of coeff is a array instead of a list (low-pass and high-pass)

		if not os.path.isdir(results_save_path):
				os.mkdir(results_save_path)

		# set video writer 
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# out_video = cv2.VideoWriter(phaseImg_path+ 'phase_ball.avi',fourcc,\
		# 				 video_to_preprocess.FPS,(video_width,video_height),isColor = False)

		size = [44,44,22,11,11]
		# need to change for video in different scales, [1,1,1/2,1/4,1/4] * original_size

		for i in range(1,num_scale-1):
			for j in range (0,num_orient):
				save_path = results_save_path + 'scale_{}'.format(i)+'_ori_{}/'.format(j)

				if not os.path.isdir(save_path):
					os.mkdir(save_path)

				print 'making the {}_th'.format(i) + ' scale {}_th orient scale phase_images'.format(j)

				out_video = cv2.VideoWriter(save_path+ 'PVideo_in_scl_{}_ori_{}.avi'.format(i,j),fourcc,\
						 video_to_preprocess.FPS,(size[i],size[i]),isColor = False)

				for k in range (0,frame_number):
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
					write_path =save_path + '{}.png'.format(k)
					cv2.imwrite(write_path,phase_image)
					out_video.write(phase_image)

				out_video.release()
		# return ori_scl_path