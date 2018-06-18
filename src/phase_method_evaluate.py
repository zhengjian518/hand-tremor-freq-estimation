from tfd_phase import TFD_PHASE
from tpd import TPD
from evaluate import Evaluate
from error_analysis import Error_anlysis
from random import randint,seed
from visualize import Visualize
from video_preprocessing import Video_Preprocessing
import numpy as np
import error_analysis
import os
import cv2
from scipy import io

import matplotlib.pyplot as plt
from filterbank import *

def get_joint_box():

	video_path_list = ['../data/video/T000/clipped/test_02.avi']
	window_size_list = [121]
	TPD.pe_save_batch(video_path_list,window_size_list)

def make_joint_video():
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
	out_video = cv2.VideoWriter('../results/clipped/segment_img/joint_video.avi', fourcc, 30, 
												(44,44),isColor = True)
	for i in range(1,971):
		segments_path = '../results/clipped/segment_img/seg_4_{}.png'.format(i)
		img = cv2.imread(segments_path,1)
		print i,
		out_video.write(np.uint8(img))
	out_video.release()

def creat_phase_images_real():

	segment_path = '../results/clipped/segment_img/'
	video_path = segment_path + 'joint_video.avi'
	results_save_path = segment_path
	Video_Preprocessing.get_phase_images_real(video_path,results_save_path,filter_on = False)

def fequency_evaluate_real():

	# phase image + pixel-wise method
	segment_path = '../results/clipped/segment_img/'
	video_path = segment_path + 'joint_video.avi'

	# MSE_list = list()
	num_scale = 5
	num_orient  = 4

	print 'Start frequency analysizing: '

	window_size = 120
	for scl in range (1,num_scale-1):
		for ori in range(0,num_orient):
			# MSE = []
			print 'FA: scl_{}'.format(scl)+'_ori_{}'.format(ori)
			# for i in range(0,21):
			path_scl_ori = segment_path
			path = path_scl_ori + 'scale_{}_ori_{}/'.format(scl,ori)
			# path = path_list[i]
			freq_results = Error_anlysis.tfd_controled_test(
												path,window_size,
												window_size/2,30,
												is_ball_video= True )

	print 'FA: ori_images'
	path = segment_path + 'ori_images/'
	freq_results = Error_anlysis.tfd_controled_test(
												path,window_size,
												window_size/2,30,
												is_ball_video= True )

def evaluate_TFD_PHASE():

	video_path_list = ['../data/video/T000/clipped/test_02.avi']
	TFD_PHASE.tfd_batch_phase(video_path_list,[121]*len(video_path_list),joint_list=[4])

def main():

	# get_joint_box()

	# make_joint_video()

	# creat_phase_images_real()

	# fequency_evaluate_real()

	evaluate_TFD_PHASE()


if __name__ == "__main__":
	main()