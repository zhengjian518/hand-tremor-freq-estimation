# This script is for exp2.2, reserve x coordinates, smooth the y cordinates, and save to a txt file
from video import Video
from video_in_frame import VideoInFrame
from io_video import IOVideo
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
		self.f.x = start_pos
		self.f.F = np.array([[1, 0], [0, 1]], np.float32) # A 
		self.f.H = np.array([[1, 0]], np.float32) 
		self.f.P *= 0.001 # covariance matrix
		self.f.R = np.array([[1]],np.float32) * 3 # measurement noise
		# f.Q = Q_discrete_white_noise(dim=2, dt=0.033, var=0.003) # process noise
		self.f.Q = np.array([[1,0],[0,1]],np.float32) * 0.003

	def Estimate(self,coordY):
		''' This function estimates the position of the object'''
		measured = np.array([[np.float32(coordY)]])
		self.f.predict()
		self.f.update(measured)
		return self.f.x

class EXP22_PRE():

	@staticmethod
	def tfd_batch_phase_clipped(cpm_joint_path_list,joint_list):
		for i in range(len(cpm_joint_path_list)):
			cpm_joint_path = cpm_joint_path_list[i]
			video_code = cpm_joint_path.split('/')[4]# 3(T008), 4(All)
			video_name = cpm_joint_path.split('/')[5]# 4      , 5
			print 'Video {}_{} in process'.format(video_code,video_name)
			result_save_path = '/local/guest/smooth_trajectory/{}/'.format(video_code)
			if not os.path.isdir(result_save_path):
				os.makedirs(result_save_path)

			EXP22_PRE.smooth_traj(cpm_joint_path,result_save_path,joint_list[i])

	@staticmethod
	def smooth_traj(cpm_joint_path,result_save_path,JOINT_LIST):

		video_name = cpm_joint_path.split('/')[5]
		# load  predcition_arr from joint_data folder
		pos_arr_list = util.get_jonit_pos_sequence(cpm_joint_path,JOINT_LIST[0],type="cpm")

		# Init a kalman filter object
		start_pos = np.array([[pos_arr_list[0][0]],[0.]])
		kfObj = KFilter(start_pos)

		predictedCoords = np.zeros((1, 1), np.float32) # only Y coordinate

		smooth_pos = list()
		# Smooth the Y pos for all frames
		for frame_i in range(0,len(pos_arr_list)):
			pred_y = pos_arr_list[frame_i][0]
			pred_x = pos_arr_list[frame_i][1]
			predictedCoords = kfObj.Estimate(pred_y)
			smooth_pos.append([int(predictedCoords[0][0]),pred_x])
		smooth_traj_txt_path = result_save_path + '{}.txt'.format(video_name)
		np.savetxt(smooth_traj_txt_path,smooth_pos)
		print 'Trajectory smoothing is done.'

if __name__ == "__main__":

	exp22_pre = EXP22_PRE()
	
	folders = util.get_full_path_under_folder('/local/guest/tfd_result_arti_conf_map/')
	folders = sorted(folders,key=lambda x: (int(re.sub('\D','',x)),x))

	for i in range(0,len(folders)):
		folders_code = util.get_full_path_under_folder(folders[i])
		folders_code = sorted(folders_code,key=lambda x: (int(re.sub('\D','',x)),x))

		cpm_joint_path_list, joint_list = [],[]
		for j in range(0, len(folders_code)):
			cpm_joint_path = folders_code[j] +'prediction_arr/'
			if "Rechts" in folders[i]:
				if os.path.isdir(cpm_joint_path):
					cpm_joint_path_list.append(cpm_joint_path)
					joint_list.append([4])
				else:
					pass
			elif "Links" in folders[i]:
				if os.path.isdir(cpm_joint_path):
					cpm_joint_path_list.append(cpm_joint_path)
					joint_list.append([7])
				else:
					pass
		exp22_pre.tfd_batch_phase_clipped(cpm_joint_path_list,joint_list)
	
