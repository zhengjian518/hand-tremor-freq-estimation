# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import types
import math
from random import randint,seed
from scipy import io
import util
import re
import operator
import csv

Category_dict = dict([('Rest', ['Rust', 'Rust_Rechts','Extra_taak_-_Rust','Extra_taak_–_Rust_Rechts']),
					 ('Rest_in_supination', ['Rust_supinatie','Rust_Supinatie','Extra_taak_-_Rust_Supinatie']),
					 ('Top_nose_left', ['Top_neus_links','Extra_taak_-_Top_neus_links']),
					 ('Top_nose_right', ['Top_neus_rechts','Extra_taak_-_Top_neus_rechts']),
					 ('Top_top', ['Top-top']),
					 ('Hands_in_pronation', ['Handen_in_pronatie']),
					 ('Thumbs_up', ['Duimen_omhoog']),
					 ('Playing_piano', ['Pianospelen','Extra_taak_-_Pianospelen']),
					 ('Finger_tapping', ['Fingertap', 'Extra_taak_-_Fingertap_rechts', 'Extra_taak_-_Fingertap_links', 'Fingertap_2']),
					 ('Counting', ['100-7', 'Extra_taak_–_Van_10_naar_1_in_turks', 'Extra_taak_-_100-7_herhaald','Extra_taak_-_Rekenen__158-13']),
					 ('Months_backward', ['Maanden_terug', 'Maanden_terug_rust','Maanden_terug_supinatie']),
					 ('2_Hz_lower', ['2_hz_lager']),
					 ('2_Hz_higher', ['2_hz_hoger']),
					 ('Following', ['Volgen']),
					 ('Weight', ['Gewicht', 'Extra_taak_–_Gewicht_en_pianospelen']),
					 ('Spiral_right', ['Spiraal_rechts']),
					 ('Spiral_left', ['Spiraal_links']),
					 ('Writing_left', ['Schrijven_rechts','Extra_taak_-_Schrijven_rechts']),
					 ('Writing_right', ['Schrijven_rechts','Extra_taak_-_Schrijven_rechts']),
					 ('Extra_pose', ['Extra_taak_-_links_tappen', 'Extra_taak_-_proef_van_luria', 'Extra_taak_-_tappen_met_geluid',
									'Extra_taak_-_Lepelbeweging','Drinken', 'Extra_taak_-_Handen_op_leuning',
									'Extra_taak_–_links_top', 'Extra_taak_-_Linkerhand_supinatie', 'Extra_taak_–_Kniebuigingen', 
									'Tikken','Extra_taak_–_Drie_houdingen', 'Extra_taak_–_Eten', 'Extra_taak_-_Staan', 
									'Extra_taak_-_drinken_rechts', 'Extra_taak_-_Beker_drinken', 'Extra_taak_-_muis_bewegen',
						   			'Extra_taak_–_Drinken','Extra_taak_-__Drinken', 'Extra_taak_–_Tremor','Extra_taak_-_rechts_iets_pakken',
						   			'Extra_taak_-_Drinken', 'Extra_taak_–_Tikken','Extra_taak_–_Beker','Extra_taak_-_drinken_uit_beker',
						   			'rechts_iets_pakken', 'Extra_taak_-_Staan_handenpronatie_hoofddraaien']),
					 ('Extra_writing', ['Extra_taak_–_Rechte_lijn', 'Extra_taak_–_Halve_maan_tekenen2','Extra_taak_-_Speciale_pen_links',
									'Extra_taak_-_Schrijven_met_vingerpen', 'Extra_taak_–_Halve_maan_tekenen'])])

def category_err_analysis_all_patients(category,acc_result_path,tfd_result_path,exp2_1_result_path,\
										exp2_2_result_path,window_size,category_plots_folder):
	"""
	start from all patients
	"""
	# step 1

	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	code_num = len(acc_full_path)
	acc_result = []
	phase_result = []
	gray_result = []
	exp2_1_result = []
	exp2_2_result = []
	code_list = []
	total_task_count = 0
	# only append 4results when error < 2 Hz 
	two_hz_phase_abs_err, two_hz_gray_abs_err, two_hz_exp2_1_abs_err, two_hz_exp2_2_abs_err = [],[],[],[]
	phase_accurate_count, gray_accurate_count, exp2_1_accurate_count, exp2_2_accurate_count = 0,0,0,0
	mean_tremor_freq_list = [] # plot6

	freq_phase_list, freq_gray_list,freq_exp2_2_list,freq_exp2_1_list= [],[],[],[]

	for code_idx in range(0,code_num):
		patient_code = acc_full_path[code_idx].split('/')[9]
		sub_code_list,sub_acc_result,sub_phase_result,sub_gray_result,sub_exp2_1_result,sub_exp2_2_result = [],[],[],[],[],[]
		
		for sub_task in Category_dict[category]:
			# exp_1
			tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(sub_task,window_size) +patient_code + '_tfd/'+ 'freq_psd_txt/'
			phasefreq_txt_path = tfd_folder + 'freq_result.txt'
			grayfreq_txt_path = tfd_folder + 'freq_rgb.txt' # note the path here is rgb not gray !
			# exp_2
			exp_2_1_path = exp2_1_result_path + '{}_crop/{}/freq.txt'.format(patient_code,sub_task)
			exp_2_2_path = exp2_2_result_path + '{}_crop/{}/freq.txt'.format(patient_code,sub_task)

			if not os.path.isdir(acc_full_path[code_idx] + '{}/'.format(sub_task)) \
				or not os.path.isfile(phasefreq_txt_path)\
				or not os.path.isfile(grayfreq_txt_path):
				continue # no such sub_task in this patient folder
			total_task_count += 1
			accfreq_txt_path = acc_full_path[code_idx] + '{}/freq.txt'.format(sub_task)
			all_acclines = np.loadtxt(accfreq_txt_path)
			
			if not int(all_acclines[-1,:][0]) == 0: # periodic detected

				mean_tremor_freq_list.append(all_acclines[-1,:][1]) # for plot 6

				
				code = patient_code.split('_')[0]
				sub_code_list.append(code)
				sub_acc_result.append(list(all_acclines[-1,:])) # 2 columns
				# tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(sub_task,window_size) +patient_code + '_tfd/'+ 'freq_psd_txt/'
				# phasefreq_txt_path = tfd_folder + 'freq_result.txt'
				all_phaselines = np.loadtxt(phasefreq_txt_path)
				sub_phase_result.append(list(all_phaselines[-1,:])) # 3 columns

				# grayfreq_txt_path = tfd_folder + 'freq_gray.txt'
				all_graylines = np.loadtxt(grayfreq_txt_path)
				sub_gray_result.append(list(all_graylines[-1,:])) # 2 columns

				# 
				all_exp2_1_lines = np.loadtxt(exp_2_1_path)
				sub_exp2_1_result.append(list(all_exp2_1_lines[-1,:]))

				# 
				all_exp2_2_lines = np.loadtxt(exp_2_2_path)
				sub_exp2_2_result.append(list(all_exp2_2_lines[-1,:]))

				freq_phase_list.append(all_phaselines[-1,:][1])
				freq_gray_list.append(all_graylines[-1,:][1])
				freq_exp2_2_list.append(all_exp2_1_lines[-1,:][1])
				freq_exp2_1_list.append(all_exp2_2_lines[-1,:][1])

				if abs(all_phaselines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
					phase_accurate_count += 1
				if abs(all_graylines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
					gray_accurate_count += 1
				if abs(all_exp2_1_lines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
					exp2_1_accurate_count += 1
				if abs(all_exp2_2_lines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
					exp2_2_accurate_count += 1

				if abs(all_phaselines[-1,:][1] - all_acclines[-1,:][1]) <= 2:
					two_hz_phase_abs_err.append(abs(all_phaselines[-1,:][1] - all_acclines[-1,:][1]))
				if abs(all_graylines[-1,:][1] - all_acclines[-1,:][1]) <= 2:
					two_hz_gray_abs_err.append(abs(all_graylines[-1,:][1] - all_acclines[-1,:][1]))
				if abs(all_exp2_1_lines[-1,:][1] - all_acclines[-1,:][1]) <= 2:
					two_hz_exp2_1_abs_err.append(abs(all_exp2_1_lines[-1,:][1] - all_acclines[-1,:][1]))
				if abs(all_exp2_2_lines[-1,:][1] - all_acclines[-1,:][1]) <= 2:
					two_hz_exp2_2_abs_err.append(abs(all_exp2_2_lines[-1,:][1] - all_acclines[-1,:][1]))

		code_list = code_list + sub_code_list
		acc_result = acc_result + sub_acc_result
		phase_result = phase_result + sub_phase_result
		gray_result = gray_result + sub_gray_result
		exp2_1_result = exp2_1_result + sub_exp2_1_result
		exp2_2_result = exp2_2_result +sub_exp2_2_result

	codes_length = len(code_list) # how many patients have this task, and detected periodic

	# step 2
	# calculate abs error
	# abs_err_phase = list(map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1]))
	# abs_err_gray = list( map(operator.sub, np.array(gray_result)[:,1], np.array(acc_result)[:,1]))

	abs_err_phase = list( map(abs,map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1])))
	abs_err_gray  = list( map(abs,map(operator.sub, np.array(gray_result)[:,1], np.array(acc_result)[:,1])))
	abs_err_exp2_1 = list( map(abs,map(operator.sub, np.array(exp2_1_result)[:,1], np.array(acc_result)[:,1])))
	abs_err_exp2_2 = list( map(abs,map(operator.sub, np.array(exp2_2_result)[:,1], np.array(acc_result)[:,1])))

	mean_err_phase = sum(list(map(abs,abs_err_phase)))/len(abs_err_phase)
	mean_err_gray   = sum(list(map(abs,abs_err_gray)))/len(abs_err_gray)
	mean_err_exp2_1 = sum(list(map(abs,abs_err_exp2_1)))/len(abs_err_exp2_1)
	mean_err_exp2_2 = sum(list(map(abs,abs_err_exp2_2)))/len(abs_err_exp2_2)

	std_phase = np.std(abs_err_phase)
	std_gray  = np.std(abs_err_gray)
	std_exp2_1 = np.std(abs_err_exp2_1)
	std_exp2_2 = np.std(abs_err_exp2_2)

	std_two_hz_phase = np.std(two_hz_phase_abs_err)
	std_two_hz_gray = np.std(two_hz_gray_abs_err)
	std_two_hz_exp2_1 = np.std(two_hz_exp2_1_abs_err)
	std_two_hz_exp2_2 = np.std(two_hz_exp2_2_abs_err)

	# To give 0 error element a small value for visualization
	abs_err_list = [abs_err_phase,abs_err_gray,abs_err_exp2_2,abs_err_exp2_1]
	for idx, abs_err in enumerate(abs_err_list):
		abs_err_list[idx] = [0.1 if x < 0.1 else x for x in abs_err]
	abs_err_phase,abs_err_gray,abs_err_exp2_2,abs_err_exp2_1, = abs_err_list[0],abs_err_list[1],abs_err_list[2],abs_err_list[3]

	# plot for 4 methods
	fig1, ax1 = plt.subplots()

	index = np.arange(len(code_list))
	bar_width = 0.2

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax1.bar(index-bar_width, abs_err_phase, bar_width,
	                alpha=opacity, color='b',
	                error_kw=error_config,
	                label='Euler_Phase')

	rects2 = ax1.bar(index, abs_err_gray, bar_width,
	                alpha=opacity, color='g',
	                error_kw=error_config,
	                label='Euler_gray')

	rects3 = ax1.bar(index + bar_width, abs_err_exp2_2, bar_width,
	                alpha=opacity, color='r',
	                error_kw=error_config,
	                label='Lag_with_smooth')

	rects4 = ax1.bar(index + 2*bar_width, abs_err_exp2_1, bar_width,
	                alpha=opacity, color='c',
	                error_kw=error_config,
	                label='Lag_no_smooth')

	ax1.set_xlabel('Patient Codes')
	ax1.set_ylabel('Absolute Error (Hz)')
	ax1.set_title('Absolute errors on {} category for all patients'.format(category))
	ax1.set_xticks(index + bar_width/2)
	plt.xticks(rotation=90)
	ax1.set_xticklabels(code_list)
	ax1.legend()

	fig1.tight_layout()
	# fig.savefig('/local/guest/joint_postion/tremor-freq-detection/error_analysis/plots/'+ '{}.pdf'.format(category))
	fig1.savefig(category_plots_folder+ '{}.eps'.format(category))

	# plot for Euler_phase and Lag_with_smooth

	two_methods_category_plots_folder = category_plots_folder + 'comparision_Euler_phase_and_Lag_with_smooth/'
	if not os.path.isdir(two_methods_category_plots_folder):
		os.makedirs(two_methods_category_plots_folder)
	fig2, ax2 = plt.subplots()

	index = np.arange(len(code_list))
	bar_width = 0.45

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax2.bar(index, abs_err_phase, bar_width,
	                alpha=opacity, color='#1f77b4', # dark blue
	                error_kw= error_config,
	                label='Euler_Phase')

	rects2 = ax2.bar(index + bar_width, abs_err_exp2_2, bar_width,
	                alpha=opacity, color='#d62728', # normal red
	                error_kw= error_config,
	                label='Lag_with_smooth')

	ax2.set_xlabel('Patient Codes')
	ax2.set_ylabel('Absolute Error (Hz)')
	ax2.set_title('Absolute errors on {} category for all patients'.format(category))
	ax2.set_xticks(index + bar_width/2)
	plt.xticks(rotation=90)
	ax2.set_xticklabels(code_list)
	ax2.legend()

	fig2.tight_layout()
	# fig.savefig('/local/guest/joint_postion/tremor-freq-detection/error_analysis/plots/'+ '{}.pdf'.format(category))
	fig2.savefig(two_methods_category_plots_folder+ '{}.eps'.format(category))

	mean_tremor_freq = np.mean(mean_tremor_freq_list)
	std_tremor = np.std(mean_tremor_freq_list)
	mean_freq_phase = np.mean(freq_phase_list)
	mean_freq_gray = np.mean(freq_gray_list)
	mean_freq_exp2_2 = np.mean(freq_exp2_2_list)
	mean_freq_exp2_1 = np.mean(freq_exp2_1_list)

	return codes_length, phase_accurate_count, gray_accurate_count, exp2_1_accurate_count, exp2_2_accurate_count,\
			 mean_err_phase, mean_err_gray, mean_err_exp2_2, mean_err_exp2_1, std_phase, std_gray, std_exp2_2, std_exp2_1,\
			 std_two_hz_phase, std_two_hz_gray, std_two_hz_exp2_1, std_two_hz_exp2_2,std_tremor, total_task_count,\
			 mean_tremor_freq, mean_freq_phase, mean_freq_gray, mean_freq_exp2_2, mean_freq_exp2_1
	
def write_baseline_to_csv(acc_result_path, csv_folder):
	
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	result_path = csv_folder + 'baseline_per_patient/'
	if not os.path.isdir(result_path):
		os.makedirs(result_path)

	for patient_code_path in acc_full_path:
		code = patient_code_path.split('/')[9]
		csv_save_path = result_path + '{}.csv'.format(code)
		csvfile = open(csv_save_path, 'wb')
		csvwriter = csv.writer(csvfile)
		csvwriter_head = ['Task','Freq','IsPeak']
		csvwriter.writerow(csvwriter_head)
		row = []
		task_paths = util.get_full_path_under_folder(patient_code_path)
		task_paths = sorted(task_paths,key=lambda x: (int(re.sub('\D','',x)),x))
		for task in task_paths:
			task_name = task.split('/')[10]
			freq_txt_path = task + 'freq.txt'
			all_lines = np.loadtxt(freq_txt_path)
			if 'Extra_taak_-__' in task_name:
				simple_task_name = re.sub('Extra_taak_-__','', task_name)
			elif 'Extra_taak_–_' in task_name:
				simple_task_name = re.sub('Extra_taak_–_','', task_name)
			elif 'Extra_taak_-_' in task_name:
				simple_task_name = re.sub('Extra_taak_-_','', task_name)
			else:
				simple_task_name = task_name
			row = [simple_task_name,all_lines[-1,:][1],all_lines[-1,:][0]]
			csvwriter.writerow(row)
		del csvwriter

def write_category_result_to_csv(category,tfd_result_path,acc_result_path, exp2_1_result_path, exp2_2_result_path, window_size,csv_folder):
	"""
	start from every task
	"""
	tasks_path = util.get_full_path_under_folder(tfd_result_path)
	tasks_path = sorted(tasks_path,key=lambda x: (int(re.sub('\D','',x)),x))

	# get all patient_codes
	patient_codes = []
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	for patient_code_path in acc_full_path:
		code = patient_code_path.split('/')[9]
		patient_codes.append(code)

	tasks_csv_folder = csv_folder + 'categories/'
	if not os.path.isdir(tasks_csv_folder):
		os.makedirs(tasks_csv_folder)

	csv_save_path = tasks_csv_folder + '{}.csv'.format(category)
	csvfile = open(csv_save_path, 'wb')
	csvwriter = csv.writer(csvfile)
	csvwriter_head = ['task_name','Patient_code','Freq_acc','Freq_Euler_phase','Freq_Euler_gray','Freq_Lag_with_smooth','Freq_Lag_no_smooth','IsPeak_acc','IsPeak_phase','IsPeak_gray','IsPeak_lag_with_smooth','IsPeak_lag_no_smooth']
	csvwriter.writerow(csvwriter_head)

	for sub_task in Category_dict[category]:
		# sub_task_path = tfd_result_path + sub_task +'_joint_tfd_61/' + 
	# for task_path in tasks_path:
		pc_csvwriter_row = []
		if 'Extra_taak_-__' in sub_task:
			simple_task_name = re.sub('Extra_taak_-__','', sub_task)
		elif 'Extra_taak_–_' in sub_task:
			simple_task_name = re.sub('Extra_taak_–_','', sub_task)
		elif 'Extra_taak_-_' in sub_task:
			simple_task_name = re.sub('Extra_taak_-_','', sub_task)
		else:
			simple_task_name = sub_task
		print simple_task_name

		for index, code in enumerate(patient_codes):
			patient_code = acc_full_path[index].split('/')[9]

			acc_path = acc_result_path + code + '/' + sub_task + '/freq.txt'
			tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(sub_task,window_size) +code + '_tfd/'+ 'freq_psd_txt/'
			phasefreq_txt_path = tfd_folder + 'freq_result.txt'
			grayfreq_txt_path = tfd_folder + 'freq_rgb.txt' # note the path here is rgb not gray !
			exp_2_1_path = exp2_1_result_path + '{}_crop/{}/freq.txt'.format(code,sub_task)
			exp_2_2_path = exp2_2_result_path + '{}_crop/{}/freq.txt'.format(code,sub_task)

			# skip the code which doesn't have this task
			if not os.path.isfile(acc_path) or not os.path.isfile(phasefreq_txt_path) or not os.path.isfile(grayfreq_txt_path):
				continue
			all_phaselines = np.loadtxt(phasefreq_txt_path)
			all_graylines = np.loadtxt(grayfreq_txt_path)
			all_acclines = np.loadtxt(acc_path)
			all_exp2_1_lines = np.loadtxt(exp_2_1_path)
			all_exp2_2_lines = np.loadtxt(exp_2_2_path)

			row = [simple_task_name, code,all_acclines[-1,:][1],all_phaselines[-1,:][1],all_graylines[-1,:][1],all_exp2_2_lines[-1,:][1],all_exp2_1_lines[-1,:][1],\
					all_acclines[-1,:][0],all_phaselines[-1,:][0],all_graylines[-1,:][0],all_exp2_2_lines[-1,:][0],all_exp2_1_lines[-1,:][0]]
			csvwriter.writerow(row)
	del csvwriter


if __name__ == "__main__":

	acc_result_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/baseline_win61/'
	tfd_result_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/tfd_result_one_channel/'
	exp2_1_result_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/exp_2_1_update/'
	exp2_2_result_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/exp_2_2_update/'
	error_analysis_path = '/tudelft.net/staff-bulk/ewi/insy/VisionLab/students/jianzheng/Error_Analysis_per_cate_one_channel/'
	if not os.path.isdir(error_analysis_path):
		os.makedirs(error_analysis_path)
	csv_folder = error_analysis_path +'CSVs/'
	plots_folder = error_analysis_path +'plots/'
	if not os.path.isdir(csv_folder) or not os.path.isdir(plots_folder):
		os.makedirs(csv_folder)
		os.makedirs(plots_folder)

	window_size = '61'

	write_baseline_to_csv(acc_result_path, csv_folder)
	for category in Category_dict:
		print category
		write_category_result_to_csv(category,tfd_result_path,acc_result_path, exp2_1_result_path, exp2_2_result_path, window_size,csv_folder)

	mean_err_per_category_csv_path = csv_folder + 'err_analysis_per_category.csv'

	mean_error_csvfile = open(mean_err_per_category_csv_path, 'wb')
	me_csvwriter = csv.writer(mean_error_csvfile)
	header = ['Category','Mean_err_Euler_phase','Std_Euler_phase','Mean_err_Euler_gray','Std_Euler_gray','Mean_err_Lag_with_smooth','Std_Lag_with_smooth','Mean_err_Lag_no_smooth','Std_Lag_no_smooth']
	me_csvwriter.writerow(header)

	total_length, phase_accurate_length, gray_accurate_length, exp2_1_accurate_length, exp2_2_accurate_length = [], [], [], [], []
	mean_err_phase_cat, mean_err_gray_cat ,mean_err_exp2_1_cat, mean_err_exp2_2_cat = [],[],[],[]
	std_phase_cat, std_gray_cat ,std_exp2_1_cat, std_exp2_2_cat = [],[],[],[]
	std_two_hz_phase_cat, std_two_hz_gray_cat ,std_two_hz_exp2_1_cat, std_two_hz_exp2_2_cat = [],[],[],[]
	std_tremor_list,total_task_count_list = [],[]
	mean_tremor_freq_list_for_plot, mean_freq_phase_list, mean_freq_gray_list, mean_freq_exp2_2_list, mean_freq_exp2_1_list = [], [],[],[],[]
	category_plots_folder = plots_folder + 'categorys/'
	if not os.path.isdir(category_plots_folder):
		os.makedirs(category_plots_folder)

	for category in Category_dict:
		print 'error analysis on category {}'.format(category)
		row = []

		codes_length, phase_accurate_count, gray_accurate_count, exp2_1_accurate_count, exp2_2_accurate_count,\
		mean_err_phase, mean_err_gray, mean_err_exp2_2, mean_err_exp2_1,\
		std_phase, std_gray, std_exp2_2, std_exp2_1,\
		std_two_hz_phase, std_two_hz_gray, std_two_hz_exp2_1, std_two_hz_exp2_2,std_tremor,total_task_count,\
		mean_tremor_freq, mean_freq_phase, mean_freq_gray, mean_freq_exp2_2, mean_freq_exp2_1 = \
					category_err_analysis_all_patients(category,acc_result_path,tfd_result_path,exp2_1_result_path,exp2_2_result_path,window_size,category_plots_folder)
		row = [category,mean_err_phase, std_phase, mean_err_gray,std_gray,mean_err_exp2_2,std_exp2_2, mean_err_exp2_1,std_exp2_1]
		me_csvwriter.writerow(row)
		total_length.append(codes_length) # how many perodic tasks deteced per category
		phase_accurate_length.append(phase_accurate_count)
		gray_accurate_length.append(gray_accurate_count)
		exp2_1_accurate_length.append(exp2_1_accurate_count)
		exp2_2_accurate_length.append(exp2_2_accurate_count)
		mean_err_phase_cat.append(mean_err_phase)
		mean_err_gray_cat.append(mean_err_gray)
		mean_err_exp2_1_cat.append(mean_err_exp2_1)
		mean_err_exp2_2_cat.append(mean_err_exp2_2)
		std_phase_cat.append(std_phase)
		std_gray_cat.append(std_gray)
		std_exp2_1_cat.append(std_exp2_1)
		std_exp2_2_cat.append(std_exp2_2)
		std_two_hz_phase_cat.append(std_two_hz_phase)
		std_two_hz_gray_cat.append(std_two_hz_gray)
		std_two_hz_exp2_1_cat.append(std_two_hz_exp2_1)
		std_two_hz_exp2_2_cat.append(std_two_hz_exp2_2)
		std_tremor_list.append(std_tremor)
		total_task_count_list.append(total_task_count)
		mean_tremor_freq_list_for_plot.append(mean_tremor_freq)
		mean_freq_phase_list.append(mean_freq_phase)
		mean_freq_gray_list.append(mean_freq_gray)
		mean_freq_exp2_2_list.append(mean_freq_exp2_2)
		mean_freq_exp2_1_list.append(mean_freq_exp2_1)

	del me_csvwriter

	categories = Category_dict.keys()

	# write mean freq and std of per cate into CSV for 4methods

	# mean_freq_std_per_category_csv_path = csv_folder + 'mean_freq_std_per_category.csv'
	# mean_freq_std_csvfile = open(mean_freq_std_per_category_csv_path, 'wb')
	# mf_std_csvwriter = csv.writer(mean_freq_std_csvfile)
	# header = ['Category','Mean_freq_Euler_phase (Hz)','Std_Euler_phase (Hz)','Mean_freq_Euler_gray (Hz)',\
	# 		'Std_Euler_gray (Hz)','Mean_freq_Lag_with_smooth (Hz)','Std_Lag_with_smooth (Hz)','Mean_freq_Lag_no_smooth (Hz)','Std_Lag_no_smooth (Hz)']
	# mf_std_csvwriter.writerow(header)
	# for idx, category in enumerate(categories):
	# 	row = [category,mean_freq_phase_list[idx],std_phase_cat[idx],mean_freq_gray_list[idx],std_gray_cat[idx],mean_freq_exp2_2_list[idx],std_exp2_2_cat[idx],mean_freq_exp2_1_list[idx],std_exp2_1_cat[idx]]
	# 	mf_std_csvwriter.writerow(row)
	# del mf_std_csvwriter

	# write dataset statistics 

	# mean_freq_std_acc_category_csv_path = csv_folder + 'dataset_statistics.csv'
	# dataset_csvfile = open(mean_freq_std_acc_category_csv_path, 'wb')
	# dataset_csvwriter = csv.writer(dataset_csvfile)
	# header = ['Category','Mean_freq (Hz)','Std (Hz)']
	# dataset_csvwriter.writerow(header)
	# for idx, category in enumerate(categories):
	# 	row = [category,mean_tremor_freq_list_for_plot[idx],std_tremor_list[idx]]
	# 	dataset_csvwriter.writerow(row)
	# del dataset_csvwriter

	# # To give 0 error element a small value for visualization
	# std_two_hz_list = [std_two_hz_phase_cat, std_two_hz_gray_cat, std_two_hz_exp2_1_cat, std_two_hz_exp2_2_cat]
	# for idx, std_two_hz in enumerate(std_two_hz_list):
	# 	std_two_hz_list[idx] = [0.1 if x < 0.1 else x for x in std_two_hz]
	# std_two_hz_phase_cat, std_two_hz_gray_cat, std_two_hz_exp2_1_cat, std_two_hz_exp2_2_cat = std_two_hz_list[0],std_two_hz_list[1],std_two_hz_list[2],std_two_hz_list[3]

	# plot all results
	# index = np.arange(len(categories))

	# # # plot for accurate quantity
	# # fig, ax = plt.subplots()

	# # bar_width = 0.2
	# # opacity = 0.4
	# # error_config = {'ecolor': '0.3'} 

	# # # sort tasks according to task amount
	# # groups = [[]] * len(categories)
	# # for i in range(0,len(categories)):
	# # 	groups[i] = [total_length[i],phase_accurate_length[i],gray_accurate_length[i],exp2_2_accurate_length[i],exp2_1_accurate_length[i],categories[i]]
	# # groups.sort(key=lambda x: x[1], reverse=True)

	# # total_length = [x[0] for x in groups]
	# # phase_accurate_length = [x[1] for x in groups]
	# # gray_accurate_length = [x[2] for x in groups]
	# # exp2_2_accurate_length = [x[3] for x in groups]
	# # exp2_1_accurate_length = [x[4] for x in groups]
	# # categories_accurate = [x[5] for x in groups]

	# # rects1 = ax.bar(index + 0.5*bar_width, total_length, 4*bar_width,
	# #                 color='None',edgecolor='#2ca02c', # vivid green
	# #                 error_kw=error_config,label='periodic_videos',linewidth=0.8,linestyle='--')

	# # rects2 = ax.bar(index - bar_width, phase_accurate_length, bar_width,
	# #                 alpha=opacity, color='#1f77b4', # dark blue
	# #                 error_kw=error_config,
	# #                 label='Euler_phase')

	# # rects3 = ax.bar(index, gray_accurate_length, bar_width,
	# #                 alpha=opacity, color='#d62728', # dark red
	# #                 error_kw=error_config,
	# #                 label='Euler_gray')

	# # rects4 = ax.bar(index + bar_width, exp2_2_accurate_length, bar_width,
	# #                 alpha=opacity, color='#bcbd22', # dark yellow
	# #                 error_kw=error_config,
	# #                 label='Lag_with_smooth')

	# # rects5 = ax.bar(index + 2*bar_width, exp2_1_accurate_length, bar_width,
	# #                 alpha=opacity, color='#17becf', # light blue
	# #                 error_kw=error_config,
	# #                 label='Lag_no_smooth')

	# # ax.set_xlabel('Category Name')
	# # ax.set_ylabel('Accurate Quantity')
	# # ax.set_title('Accurate Quantity For All Tasks (Abs_Err < 1Hz)')
	# # ax.set_xticks(index + bar_width/2)
	# # plt.xticks(rotation=90)
	# # ax.set_xticklabels(categories_accurate)
	# # ax.legend()

	# # fig.tight_layout()
	# # fig.savefig(plots_folder +'acc_num_per_cate.eps')

	# # plot for errors of Euler comparision

	# groups = [[]] * len(categories)
	# for i in range(0,len(categories)):
	# 	groups[i] = [mean_err_phase_cat[i],mean_err_gray_cat[i],std_two_hz_phase_cat[i],std_two_hz_gray_cat[i],categories[i]]
	# groups.sort(key=lambda x: x[0])

	# comp_mean_err_phase_cat = [x[0] for x in groups]
	# comp_mean_err_gray_cat = [x[1] for x in groups]
	# comp_std_two_hz_phase_cat = [x[2] for x in groups]
	# comp_std_two_hz_gray_cat = [x[3] for x in groups]
	# comp_categories = [x[4] for x in groups]

	# # fig2, ax2 = plt.subplots()
	# # bar_width = 0.4
	# # opacity = 0.4
	# # error_config = {'ecolor': '0.3'} 

	# # rects1 = ax2.bar(index , comp_mean_err_phase_cat , bar_width,
	# #                 alpha=opacity, color='#1f77b4', # dark blue
	# #                 error_kw=error_config,yerr= comp_std_two_hz_phase_cat, 
	# #                 label='Euler_phase')

	# # rects2 = ax2.bar(index + bar_width, comp_mean_err_gray_cat, bar_width,
	# #                 alpha=opacity, color='#d62728', # dark red
	# #                 error_kw=error_config,yerr= comp_std_two_hz_gray_cat,
	# #                 label='Euler_gray')

	# # ax2.set_xlabel('Category Name')
	# # ax2.set_ylabel('Absolute Error (Hz)')
	# # ax2.set_title('Mean Absolute Error of Two Eulerian Based Methods')
	# # ax2.set_xticks(index + bar_width/2)
	# # plt.xticks(rotation=90)
	# # ax2.set_xticklabels(comp_categories)
	# # ax2.legend(loc=2)

	# # fig2.tight_layout()
	# # fig2.savefig(plots_folder +'Euler_2methods_comparision.eps')

	# fig2, ax2 = plt.subplots()
	# bar_width = 0.4
	# opacity = 0.4
	# error_config = {'ecolor': '0.3'} 

	# rects1 = ax2.bar(index , comp_mean_err_phase_cat , bar_width,
	#                 alpha=opacity, color='#1f77b4', # dark blue
	#                 error_kw=error_config,
	#                 label='Euler_phase')

	# rects2 = ax2.bar(index + bar_width, comp_mean_err_gray_cat, bar_width,
	#                 alpha=opacity, color='#d62728', # dark red
	#                 error_kw=error_config,
	#                 label='Euler_gray')

	# ax2.set_xlabel('Category Name')
	# ax2.set_ylabel('Absolute Error (Hz)')
	# ax2.set_title('Mean Absolute Error of Two Eulerian Based Methods')
	# ax2.set_xticks(index + bar_width/2)
	# plt.xticks(rotation=90)
	# ax2.set_xticklabels(comp_categories)
	# ax2.legend(loc=2)

	# fig2.tight_layout()
	# fig2.savefig(plots_folder +'Euler_2methods_comparision.eps')

	# # plot for errors of Lag comparision

	# groups = [[]] * len(categories)
	# for i in range(0,len(categories)):
	# 	groups[i] = [mean_err_exp2_2_cat[i],mean_err_exp2_1_cat[i],std_two_hz_exp2_2_cat[i],std_two_hz_exp2_1_cat[i],categories[i]]
	# groups.sort(key=lambda x: x[0])

	# comp_mean_err_exp2_2_cat = [x[0] for x in groups]
	# comp_mean_err_exp2_1_cat = [x[1] for x in groups]
	# comp_std_two_hz_exp2_2_cat = [x[2] for x in groups]
	# comp_std_two_hz_exp2_1_cat = [x[3] for x in groups]
	# comp_categories = [x[4] for x in groups]

	# # fig3, ax3 = plt.subplots()
	# # bar_width = 0.4
	# # opacity = 0.4
	# # error_config = {'ecolor': '0.3'} 

	# # rects1 = ax3.bar(index , comp_mean_err_exp2_2_cat, bar_width,
	# #                 alpha=opacity, color = '#1f77b4', # dark blue
	# #                 error_kw = error_config,yerr= comp_std_two_hz_exp2_2_cat,
	# #                 label='Lag_with_smooth')

	# # rects2 = ax3.bar(index + bar_width, comp_mean_err_exp2_1_cat, bar_width,
	# #                 alpha = opacity, color = '#d62728', # dark red
	# #                 error_kw = error_config,yerr= comp_std_two_hz_exp2_1_cat,
	# #                 label = 'Lag_no_smooth')

	# # ax3.set_xlabel('Category Name')
	# # ax3.set_ylabel('Absolute Error (Hz)')
	# # ax3.set_title('Mean Absolute Error of Two Lagrangian Based Methods')
	# # ax3.set_xticks(index + bar_width/2)
	# # plt.xticks(rotation=90)
	# # ax3.set_xticklabels(comp_categories)
	# # ax3.legend(loc=2)
	# # fig3.tight_layout()
	# # fig3.savefig(plots_folder +'Lag_2methods_comparision.eps')

	# fig3, ax3 = plt.subplots()
	# bar_width = 0.4
	# opacity = 0.4
	# error_config = {'ecolor': '0.3'} 

	# rects1 = ax3.bar(index , comp_mean_err_exp2_2_cat, bar_width,
	#                 alpha=opacity, color = '#1f77b4', # dark blue
	#                 error_kw = error_config,
	#                 label='Lag_with_smooth')

	# rects2 = ax3.bar(index + bar_width, comp_mean_err_exp2_1_cat, bar_width,
	#                 alpha = opacity, color = '#d62728', # dark red
	#                 error_kw = error_config,
	#                 label = 'Lag_no_smooth')

	# ax3.set_xlabel('Category Name')
	# ax3.set_ylabel('Absolute Error (Hz)')
	# ax3.set_title('Mean Absolute Error of Two Lagrangian Based Methods')
	# ax3.set_xticks(index + bar_width/2)
	# plt.xticks(rotation=90)
	# ax3.set_xticklabels(comp_categories)
	# ax3.legend(loc=2)
	# fig3.tight_layout()
	# fig3.savefig(plots_folder +'Lag_2methods_comparision.eps')

	# # # plot for errors of 4 methods comaprision 

	# # groups = [[]] * len(categories)
	# # for i in range(0,len(categories)):
	# # 	groups[i] = [mean_err_phase_cat[i],mean_err_gray_cat[i],mean_err_exp2_2_cat[i],mean_err_exp2_1_cat[i],std_two_hz_phase_cat[i],std_two_hz_gray_cat[i],std_two_hz_exp2_2_cat[i],std_two_hz_exp2_1_cat[i],categories[i]]
	# # groups.sort(key=lambda x: x[0])

	# # comp_mean_err_phase_cat = [x[0] for x in groups]
	# # comp_mean_err_gray_cat = [x[1] for x in groups]
	# # comp_mean_err_exp2_2_cat = [x[2] for x in groups]
	# # comp_mean_err_exp2_1_cat = [x[3] for x in groups]
	# # comp_std_two_hz_phase_cat = [x[4] for x in groups]
	# # comp_std_two_hz_gray_cat = [x[5] for x in groups]
	# # comp_std_two_hz_exp2_2_cat = [x[6] for x in groups]
	# # comp_std_two_hz_exp2_1_cat = [x[7] for x in groups]
	# # comp_categories = [x[8] for x in groups]

	# # fig4, ax4 = plt.subplots()

	# # bar_width = 0.2
	# # opacity = 0.4
	# # error_config = {'ecolor': '0.3'} 

	# # rects1 = ax4.bar(index - bar_width, comp_mean_err_phase_cat, bar_width,
	# #                 alpha=opacity, color='#1f77b4', # dark blue
	# #                 error_kw=error_config,
	# #                 label='Euler_phase')

	# # rects2 = ax4.bar(index, comp_mean_err_gray_cat, bar_width,
	# #                 alpha=opacity, color='#d62728', # dark red
	# #                 error_kw=error_config,
	# #                 label='Euler_gray')

	# # rects3 = ax4.bar(index + bar_width, comp_mean_err_exp2_2_cat, bar_width,
	# #                 alpha=opacity, color='#bcbd22', # dark yellow
	# #                 error_kw=error_config,
	# #                 label='Lag_with_smooth')

	# # rects4 = ax4.bar(index + 2*bar_width, comp_mean_err_exp2_1_cat, bar_width,
	# #                 alpha=opacity, color='#17becf', # light blue
	# #                 error_kw=error_config,
	# #                 label='Lag_no_smooth')

	# # ax4.set_xlabel('Category Name')
	# # ax4.set_ylabel('Absolute Error (Hz)')
	# # ax4.set_title('Mean Absolute Error of Four Methods')
	# # ax4.set_xticks(index + bar_width/2)
	# # plt.xticks(rotation=90)
	# # ax4.set_xticklabels(comp_categories)
	# # ax4.legend()

	# # fig4.tight_layout()
	# # fig4.savefig(plots_folder +'4methods_comparision.eps')

	# # plot for a histogram of total number of videos recorded per task (including non-periodic videos) 

	# groups = [[]] * len(categories)
	# for i in range(0,len(categories)):
	# 	groups[i] = [total_task_count_list[i],categories[i]]
	# groups.sort(key=lambda x: x[0],reverse=True)

	# comp_total_task_count_list = [x[0] for x in groups]
	# comp_categories = [x[1] for x in groups]

	# fig5, ax5 = plt.subplots()

	# bar_width = 0.45
	# opacity = 0.4
	# error_config = {'ecolor': '0.3'} 

	# rects1 = ax5.bar(index +0.5*bar_width, comp_total_task_count_list, bar_width,
	#                 alpha=opacity, color='#1f77b4', # dark blue
	#                 error_kw = error_config)

	# ax5.set_xlabel('Category Name')
	# ax5.set_ylabel('Total Number ')
	# ax5.set_title('Total Number of Videos Recorded Per Category')
	# ax5.set_xticks(index + bar_width/2)
	# plt.xticks(rotation=90)
	# ax5.set_xticklabels(comp_categories)
	# # ax5.legend()

	# fig5.tight_layout()
	# fig5.savefig(plots_folder +'total_num_of_videos_per_category.eps')


# 	# To give 0 error element a small value for visualization
# 	std_tremor_list = [0.1 if x < 0.1 else x for x in std_tremor_list]
	
# 	# plot for a histogram with the average tremor frequency per task estimated from the accelerometer data.

# 	groups = [[]] * len(categories)
# 	for i in range(0,len(categories)):
# 		groups[i] = [mean_tremor_freq_list_for_plot[i],std_tremor_list[i],categories[i]]
# 	groups.sort(key=lambda x: x[0])

# 	comp_mean_tremor_freq_list_for_plot = [x[0] for x in groups]
# 	comp_std_tremor_list = [x[1] for x in groups]
# 	comp_categories = [x[2] for x in groups]

# 	fig6, ax6 = plt.subplots()

# 	bar_width = 0.5
# 	opacity = 0.4
# 	error_config = {'ecolor': '0.3'} 

# 	rects1 = ax6.bar(index +0.5*bar_width, comp_mean_tremor_freq_list_for_plot, bar_width,
# 	                alpha=opacity, color='#1f77b4', # dark blue
# 	                error_kw=error_config,yerr = comp_std_tremor_list)

# 	ax6.set_xlabel('Category Name')
# 	ax6.set_ylabel('Average Tremor Frequency (Hz)')
# 	ax6.set_title('Average Tremor Frequency Estimated From the Accelerometer Data')
# 	ax6.set_xticks(index + bar_width/2)
# 	plt.xticks(rotation=90)
# 	ax6.set_xticklabels(comp_categories)
# 	# ax5.legend()

# 	fig6.tight_layout()
# 	fig6.savefig(plots_folder +'avg_tremor_freq_per_cate_from_accelerometer_V1.eps')


# # plot for a histogram with the average tremor frequency per task estimated from the accelerometer data.

# 	groups = [[]] * len(categories)
# 	for i in range(0,len(categories)):
# 		groups[i] = [mean_tremor_freq_list_for_plot[i],std_tremor_list[i],categories[i]]
# 	groups.sort(key=lambda x: x[1])

# 	comp_mean_tremor_freq_list_for_plot = [x[0] for x in groups]
# 	comp_std_tremor_list = [x[1] for x in groups]
# 	comp_categories = [x[2] for x in groups]

# 	fig7, ax7 = plt.subplots()

# 	bar_width = 0.5
# 	opacity = 0.4
# 	error_config = {'ecolor': '0.3'} 

# 	rects1 = ax7.bar(index +0.5*bar_width, comp_mean_tremor_freq_list_for_plot, bar_width,
# 	                alpha=opacity, color='#1f77b4', # dark blue
# 	                error_kw=error_config,yerr = comp_std_tremor_list)

# 	ax7.set_xlabel('Category Name')
# 	ax7.set_ylabel('Average Tremor Frequency (Hz)')
# 	ax7.set_title('Average Tremor Frequency Estimated From the Accelerometer Data')
# 	ax7.set_xticks(index + bar_width/2)
# 	plt.xticks(rotation=90)
# 	ax7.set_xticklabels(comp_categories)
# 	# ax5.legend()

# 	fig7.tight_layout()
# 	fig7.savefig(plots_folder +'avg_tremor_freq_per_cate_from_accelerometer_V2.eps')

