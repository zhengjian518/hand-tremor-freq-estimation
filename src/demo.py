from tfd import TFD
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

def evaluate_TFD():
	"""1. TFD Architecture - Compute the frequency of tremors from video."""

	# Tremor Detection
	# video_path_list = ['../data/video/T000/clipped/test_02.mp4']
	# TFD.tfd_batch(video_path_list,[121]*len(video_path_list),joint_list=[4])
	video_path_list = ['../data/video/T000/clipped/test_02.avi']
	TFD.tfd_batch(video_path_list,[121]*len(video_path_list),joint_list=[4])
	
	# Web Visualization
	# video_path = '../data/video/T000/clipped/test_02.avi'
	# visual_video_savepath = '../results/visual/visual_' + os.path.basename(
	#                                                     video_path)[:-4]+'.avi' 
	# Visualize.gen_visual_video(video_path,visual_video_savepath,121)

def evaluate_PE():
	"""2. PE module Evaluation, need manual annotations."""
	# Run on server 
	video_path_list = util.get_file_fullpath_list('/media/mount/TremorData',
															  'kinect.avi')
	frame_sample_num = 15
	Evaluate.pe_evaluate_phase1_1_v2_pe(video_path_list,frame_sample_num)

	# Run on local machine, only video name is enough
	video_path_list =  os.listdir('../results/evaluate/') # ['T001'] # 
	video_path_list.sort()
	video_gt_mat_path_list = ['../results/evaluate/'+video_name+'/'+\
					  video_name+'_gt_mat/' for video_name in video_path_list]
	print video_path_list
	Evaluate.pe_evaluate_phase1_2_v2_pe(video_path_list)
	Evaluate.pe_evaluate_phase2_gt(video_gt_mat_path_list)

	csvfile = open('../results/evaluate/pckh_all.csv', 'wb')
	csvwriter = csv.writer(csvfile)
	for i in np.arange(0.0, 0.55, 0.05):
		pckh_results = Evaluate.pe_evaluate_phase3_pckh(video_path_list,i)
		csvwriter.writerow([str(x) for x in pckh_results])
	csvfile.close()

def evaluate_FE_syn():
	"""FE module, synthetic video"""

	# 3.1 MSE vs Frequency fps=30.0, no distub and noise
	# Error_anlysis.run(debug_type='make_ball',fps=30.0,freq_noise=0,
	#                   disturb_on=False,disturb_mag=0,window_size=120)
	# Error_anlysis.run(debug_type='ball_test',fps=30.0,freq_noise=0,
	#                   disturb_on=False,window_size=120)
	
	# 3.2 MSE vs disturb mag (6Hz)
	# MSE = None
	# run_time = 10
	# for i in range(0,run_time):
	#     print("No. {} running".format(i))
	#     seed(i)
	#     Error_anlysis.run(debug_type='make_ball_disturb',freq_tremor=6,
	#                                                           disturb_on=True)
	#     if MSE is None:
	#         MSE=Error_anlysis.run(debug_type='ball_test_disturb')
	#     else:
	#         MSE+=Error_anlysis.run(debug_type='ball_test_disturb')
	# disturb_range = np.arange(0,16)
	# error_analysis.plot_to_file(x=disturb_range,y=MSE/run_time,
	#               ylabel='Mean Square Error',
	#             xlabel='Disturb Magnitude (Pixel)',
	#             title='MSE vs Disturb Magnitude, fps={}, window_size={}'.format(30,120),
	#             save_path='../results/error_analysis/MSEvsDisturb.eps')

	# 3.3 sim vs noise sigma (6Hz)
	# MSE,MSE_sim = None,None
	# run_time = 1
	# for i in range(0,run_time):
	#     print("No. {} running".format(i))
	#     np.random.seed(i)
	#     Error_anlysis.run(debug_type='make_ball_noise',freq_tremor=6)
	#     if MSE is None:
	#         [MSE,MSE_sim]=Error_anlysis.run(debug_type='ball_test_noise_sim',
	#                                                             freq_tremor=6)
	#     else:
	#         [MSE_,MSE_sim_]=Error_anlysis.run(debug_type='ball_test_noise_sim',
	#                                                             freq_tremor=6)
	#         MSE+=MSE_
	#         MSE_sim+=MSE_sim_            
	# sigma_range = np.arange(0,21) * 0.1
	# io.savemat('../results/error_analysis/sim_mse/MSEvsNoise.mat', \
	#     mdict={'sigma_range':sigma_range,'MSE':np.array(MSE),
	#             'MSE_sim':np.array(MSE_sim)})
	# m = io.loadmat('../results/error_analysis/sim_mse/MSEvsNoise.mat')
	# sigma_range = (m['sigma_range'])[0]
	# MSE = (m['MSE'])[0]
	# MSE_sim = (m['MSE_sim'])[0]
	# error_analysis.multi_plot_to_file(x1=sigma_range,y1=MSE/run_time,
	#             x2=sigma_range,y2=MSE_sim/run_time,
	#         ylabel='Mean Squared Error',
	#         xlabel='Guassian Noise '+r'$\sigma$',
	#         legends=['Pixel-wise','Similarity'],
	#         title='MSE - Noise',
	#         save_path='../results/error_analysis/sim_mse/MSEvsNoise.eps')

	# 3.4 MSE vs noise sigma (6Hz)
	# MSE = None
	# run_time = 10
	# for i in range(0,run_time):
	#     print("No. {} running".format(i))
	#     np.random.seed(i)
	#     Error_anlysis.run(debug_type='make_ball_noise',freq_tremor=6)
	#     if MSE is None:
	#         MSE=Error_anlysis.run(debug_type='ball_test_noise',freq_tremor=6)
	#     else:
	#         MSE+=Error_anlysis.run(debug_type='ball_test_noise',freq_tremor=6)
	# sigma_range = np.arange(0,21) * 0.1
	# error_analysis.plot_to_file(x=sigma_range,y=MSE/run_time,
	#         ylabel='Mean Square Error',
	#         xlabel='Guassian Noise '+r'$\sigma$',
	#         title='MSE vs Guassian Noise '+r'$\sigma$'+\
	#                                 ', fps={}, window_size={}'.format(30,120),
	#         save_path='../results/error_analysis/MSEvsNoise.eps')    

	# new built
	# 3.5 MSE vs disturb mag (6Hz)(preprocessed)

	# MSE = None
	# run_time = 10
	# for i in range(0,run_time):
	#     print("No. {} running".format(i))
	#     seed(i)
	#     Error_anlysis.run(debug_type='make_color_ball_disturb',freq_tremor=6,
	#                                                           disturb_on=True)
	#     if MSE is None:
	#         MSE=Error_anlysis.run(debug_type='color_ball_test_disturb')
	#     else:
	#         MSE+=Error_anlysis.run(debug_type='color_ball_test_disturb')
	# disturb_range = np.arange(0,16)
	# error_analysis.plot_to_file(x=disturb_range,y=MSE/run_time,
	#               ylabel='Mean Square Error',
	#             xlabel='Disturb Magnitude (Pixel)',
	#             title='MSE vs Disturb Magnitude (preprocessed), fps={}, window_size={}'.format(30,120),
	#             save_path='../results/error_analysis/MSEvsDisturb(2.27).eps')

	# new built
	# 3.6 MSE vs noise sigma (6Hz)(preprocessed)

	# MSE = None
	# run_time = 3
	# for i in range(0,run_time):
	#     print("No. {} running".format(i))
	#     np.random.seed(i)
	#     Error_anlysis.run(debug_type='make_color_ball_noise',freq_tremor=6)
	#     if MSE is None:
	#         MSE=Error_anlysis.run(debug_type='color_ball_test_noise',freq_tremor=6)
	#     else:
	#         MSE+=Error_anlysis.run(debug_type='color_ball_test_noise',freq_tremor=6)
	# sigma_range = np.arange(0,21) * 0.1
	# error_analysis.plot_to_file(x=sigma_range,y=MSE/run_time,
	#         ylabel='Mean Square Error',
	#         xlabel='Guassian Noise '+r'$\sigma$',
	#         title='(preprocessed)MSE vs Guassian Noise '+r'$\sigma$'+\
	#                                 ', fps={}, window_size={}'.format(30,120),
	#         save_path='../results/error_analysis/preprocessed_MSEvsNoise.eps')    

	# new built
	# 3.7 MSE vs disturb (6Hz)(Phase images)
	MSE_list = None
	run_time = 1
	for i in range(0,run_time):
	    print("No. {} running".format(i))
	    np.random.seed(i)
	    Error_anlysis.run(debug_type='make_ball_disturb',freq_tremor=6,disturb_on=True)
	    if MSE_list is None:
	        MSE_list = np.array(Error_anlysis.run(debug_type='ball_phase_test_disturb'))
	    else:
	        MSE_list += np.array(Error_anlysis.run(debug_type='ball_phase_test_disturb'))

	    print 'No. {}__MSE_list'.format(i)
	    print MSE_list

	# # run_time = 5 # 12 *16

	# print MSE_list
	MSE_list = list(MSE_list)
	disturb_range = np.arange(0,16)
	legends=['0','1/4 pi','1/2 pi','3/4 pi']
	for plt in range(0,3):
	    MSE_ori_list = []
	    for ori in range(0,4):
	        leftpop = MSE_list.pop(0)
	        MSE_ori_list.append(leftpop)
			
	    error_analysis.n_plot_to_file(run_time, disturb_range,MSE_ori_list,xlabel = 'Disturb Magnitude (Pixel)',\
	                ylabel ='Mean Square Error',\
	                legends = legends,\
	                title = 'MSE vs Disturb Magnitude (phase scale_{}), fps={}, window_size={}'.format(plt,30,120),\
	                plot_path = '../results/error_analysis/phase_ball_fps30.0_disturbmag/'+ 'MSE_scale{}_{}runtimes.eps'.format(plt,run_time))


	# new built
	# 3.8 MSE vs noise (6Hz)(Phase images)

	# MSE_list = None
	# run_time = 1
	# for i in range(0,run_time):
	# 	print("No. {} running".format(i))
	# 	np.random.seed(i)
	# 	Error_anlysis.run(debug_type='make_ball_noise',freq_tremor=6)
	# 	if MSE_list is None:
	# 		MSE_list = np.array(Error_anlysis.run(debug_type='ball_phase_test_noise',freq_tremor=6))
	# 	else:
	# 		MSE_list += np.array(Error_anlysis.run(debug_type='ball_phase_test_noise',freq_tremor=6))

	# 	# print 'No. {}__MSE_list'.format(i)
	# 	print MSE_list

	# # print MSE_list

	# MSE_list = list(MSE_list)
	# sigma_range = np.arange(0,21) *0.1
	# legends=['0','1/4 pi','1/2 pi','3/4 pi']
	# for plt in range(0,3):
	# 	MSE_ori_list = []
	# 	for ori in range(0,4):
	# 		leftpop = MSE_list.pop(0)
	# 		MSE_ori_list.append(leftpop)

	# 	error_analysis.n_plot_to_file(run_time,sigma_range,MSE_ori_list,xlabel = 'Guassian Noise '+r'$\sigma$',\
	# 				ylabel ='Mean Square Error',\
	# 				legends = legends,\
	# 				title = 'MSE vs Guassian Noise '+r'$\sigma$ '+' (phase scale_{}), '.format(plt)+\
	# 						'fps={}, window_size={}'.format(30,120),\
	# 				plot_path = '../results/error_analysis/phase_ball_fps30.0_noise/'+ 'MSE_scale{}._{}runtimes.eps'.format(plt,run_time))


def evaluate_FE_real():
	"""FE module, real video.
		NOTICE: this experiment assumes that cropped images or predictions have 
				been cached (you can use the code in tpd.py to do so).
	"""

	# 4.1 Pixel-wise method
	window_size_list = [121]
	video_code_list = ['T008']
	joint_No_list = [4]
	video_name_list = ['Rust']

	for window_size in window_size_list:
		for i in range(len(video_code_list)):
			video_code = video_code_list[i]
			joint_No = joint_No_list[i]
			for video_name in video_name_list:
				print("Window: {}, Video '{}-{}-j{}' in process".format(
														window_size,video_code,
														video_name,joint_No))
				save_path = '../results/compare_cropping_method/window_size_{}/{}/'.format(
														window_size,video_code)
				if not os.path.isdir(save_path):
					os.makedirs(save_path)

				video_sliding_PE = '../data/video/pe_data_/{}/{}/seg_{}/'.format(
												video_code,video_name,joint_No)
				freq = Error_anlysis.tfd_controled_test_sliding_pe(
											segment_img_path=video_sliding_PE,
											window_size=window_size,
											noverlap=window_size/2,
											joint_No=joint_No,fps=30)
				freq = [float(x) for x in freq]
				io.savemat(save_path+video_name+'_sliding_PE.mat',
											mdict={'freq':np.array(freq)})

	# 4.2 Similarity method
	video_path_list = ['Rust']
	for video_path_ in video_path_list:
		video_path = '../data/video/pe_data_/T008/{}/seg_4/'.format(video_path_)
		print('Video-{} in process'.format(video_path_))
		TPD.tremor_period_detec_sim_fft_(video_path,121,61,[4])

def test_make_color_ball():
	"""This is used to test the colorful ball
		NOTICE: 
	"""    
	freq_tremor = 6
	freq_noise = 1
	save_path = '../results/JETcolor_ball/'
	fps=30.0
	disturb_on=True
	disturb_mag=2
	noise_on= True

	Error_anlysis.make_color_ball_video(freq_tremor,freq_noise,save_path,fps=30.0,
					disturb_on=False,disturb_mag=2,noise_on=False,
					noise_sigma=1,ball_radius=8)

def video_preprocess_check():
	"""This is used to check the video preprocess
		NOTICE: 
	"""    
	video_path = '../results/JETcolor_ball/color_ball.avi'
	Video_Preprocessing.video_preprocessing(video_path,filter_on = True, norm_mode = 'white_noise_remove',\
											debug_type = 'norm_mode_debug')

def phase_image_check(filter_type,parameter):

	# image_path = '../phase_image/ori_img.jpg'
	# im = cv2.imread(image_path)
	# grayIm = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),(30,30),cv2.INTER_AREA)
	# steer = Steerable()
	# coeff = steer.buildSCFpyr(grayIm)
	# phase_emp = coeff[3][1]
	# angles = np.angle(phase_emp)
	# print angles
	# angles = angles + np.pi
	# print angles
	# # amp = 255*np.ones((69,143),dtype = int)
	# # amp = np.abs(phase_emp)
	# # phase_image = np.uint8(amp * np.exp(angles * 1j).real)
	# phase_image = np.uint8(cv2.convertScaleAbs(angles,alpha=255.0/(2*np.pi)))
	# # phase_image = np.uint8(cv2.convertScaleAbs(angles))
	# print phase_image
	# save_path = '../phase_image/pha_img.jpg'
	# cv2.imwrite(save_path,phase_image)
	# print 'done!!!'
	# grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	# cv2.imwrite('../phase_image/grayIm.jpg',grayIm)
	# size = (im.shape[0],im.shape[1])
	# im = 255*np.random.normal(0, 1.0, size)
	# img = im + grayIm
	# cv2.imwrite('../phase_image/noise.jpg',img)
	# im = cv2.GaussianBlur(img,(0,0),2)
	# print im
	# cv2.imwrite('../phase_image/try.jpg',im)
	# # print phase_image.shape

	image_path = '../phase_image/1.png'
	# image_path = '../phase_image/1.png'
	im = cv2.imread(image_path)
	# size = (im.shape[0],im.shape[1])
	# noise= 255*np.random.normal(0, 1.0, size)
	# im = im + noise
	# print im
	# print np.uint8(im)
	# cv2.imwrite('../phase_image/ball-noise.jpg',im)
	if filter_type == 'GaussianBlur':
		frame_filter = cv2.GaussianBlur(im,(parameter[0],parameter[1]),parameter[2])
	elif filter_type == 'bilateralFilter':
		frame_filter = cv2.bilateralFilter(im, parameter[0],parameter[1],parameter[2])

	cv2.imwrite('../phase_image/1_{}_parameter({},{},{}).jpg'.format(filter_type,parameter[0],parameter[1],parameter[2]),frame_filter)


def main():
	# evaluate_TFD()

	# evaluate_PE()

	# evaluate_FE_syn()

	# evaluate_FE_real()

	# test_make_color_ball()

	# video_preprocess_check()

if __name__ == "__main__":
	main()