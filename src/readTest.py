import util
import csv
import os
import re
import numpy as np

def read_data_in_txt(txt_path,row_num):
	"""
	Read in the string data(predction position) in txt file, cast into 'int' type, 
	need to speccify the total number of rows in the txt file.
	"""
	rawText = open(txt_path)

	data = []
	row = row_num

	for x in range(0,row):
		data.append([])

	line_num = 0

	for line in rawText.readlines():
		for i in line.split():
			data[line_num].append(int(float(i)))
		line_num = line_num + 1

	rawText.close()
	return data

def get_jonit_pos_sequence(txt_top_path,joint_num):
	"""This function returns the joint positions along time series, for comparing the performance of traackers

	Args:
		txt_top_path: folder path that contains all txt files;
		joint_number: joint num, 4 for right wrist
	Returns:
		joint_pos : a list that contains all joint postions in the video frames
	"""

	# path_list= util.get_file_list(txt_top_path, 'txt')
	path_list= util.get_file_fullpath_list(txt_top_path, file_format='txt')

	path_list = sorted(path_list,key=lambda x: (int(re.sub('\D','',x)),x))

	joint_pos = []
	for x in range(0,len(path_list)):
		joint_pos.append([])

	for i in range(0,len(path_list)):
		joint_pos[i]= util.read_data_in_txt(path_list[i],14)[joint_num]

	return joint_pos


joint_pos = get_jonit_pos_sequence('../results/Top_neus_links/prediction_arr/',4)

print joint_pos
