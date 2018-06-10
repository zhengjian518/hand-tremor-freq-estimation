# task-wise error analysis
# tfd + joint_cropping + phase + tracker 
# This one for task Rust T001-T041 patient

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

def task_err_analusis_all_patients(acc_result_path,tfd_result_path,task,window_size):
	"""
	start from all patients
	"""
	# step 1
	# read in the results
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	code_num = len(acc_full_path)
	acc_result = []
	phase_result = []
	rgb_result = []
	code_list = []
	peak_mark = []
	phase_accurate_count, rgb_accurate_count = 0,0

	for code_idx in range(0,code_num):
		patient_code = acc_full_path[code_idx].split('/')[5]
		tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(task,window_size) +patient_code + '_tfd/'+ 'freq_psd_txt/'
		phasefreq_txt_path = tfd_folder + 'freq_result.txt'
		rgbfreq_txt_path = tfd_folder + 'freq_rgb.txt'
		if not os.path.isdir(acc_full_path[code_idx] + '{}/'.format(task)) \
			or not os.path.isfile(phasefreq_txt_path)\
			or not os.path.isfile(rgbfreq_txt_path):
			continue # no such task in this patient folder
		
		accfreq_txt_path = acc_full_path[code_idx] + '{}/freq.txt'.format(task)
		all_acclines = np.loadtxt(accfreq_txt_path)
		
		if not int(all_acclines[-1,:][0]) == 0: # periodic detected
			code = patient_code.split('_')[0]
			code_list.append(code)
			acc_result.append(list(all_acclines[-1,:])) # 2 columns
			peak_mark.append(code_idx)
			# tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(task,window_size) +patient_code + '_tfd/'+ 'freq_psd_txt/'
			# phasefreq_txt_path = tfd_folder + 'freq_result.txt'
			all_phaselines = np.loadtxt(phasefreq_txt_path)
			phase_result.append(list(all_phaselines[-1,:])) # 3 columns

			# rgbfreq_txt_path = tfd_folder + 'freq_rgb.txt'
			all_rgblines = np.loadtxt(rgbfreq_txt_path)
			rgb_result.append(list(all_rgblines[-1,:])) # 2 columns

			if abs(all_phaselines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
				phase_accurate_count +=1
			if abs(all_rgblines[-1,:][1] - all_acclines[-1,:][1]) <= 1:
				rgb_accurate_count += 1

	codes_length = len(code_list) # how many patients have this task, and detected periodic


	# step 2
	# calculate abs error
	# abs_err_phase = list(map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1]))
	# abs_err_rgb = list( map(operator.sub, np.array(rgb_result)[:,1], np.array(acc_result)[:,1]))

	abs_err_phase = list( map(abs,map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1])))
	abs_err_rgb = list( map(abs,map(operator.sub, np.array(rgb_result)[:,1], np.array(acc_result)[:,1])))
	

	mean_err_phase = sum(list(map(abs,abs_err_phase)))/len(abs_err_phase)
	mean_err_rgb = sum(list(map(abs,abs_err_rgb)))/len(abs_err_rgb)

	fig, ax = plt.subplots()

	index = np.arange(len(code_list))
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax.bar(index, abs_err_phase, bar_width,
	                alpha=opacity, color='g',
	                error_kw=error_config,
	                label='abs_err_phase')

	rects2 = ax.bar(index + bar_width, abs_err_rgb, bar_width,
	                alpha=opacity, color='r',
	                error_kw=error_config,
	                label='abs_err_rgb')

	ax.set_xlabel('Patient Codes')
	ax.set_ylabel('Absolute Error (HZ)')
	ax.set_title('Absolute errors on {} task for all patients'.format(task))
	ax.set_xticks(index + bar_width/2)
	plt.xticks(rotation=60)
	ax.set_xticklabels(code_list)
	ax.legend()

	fig.tight_layout()
	fig.savefig('/local/guest/joint_postion/tremor-freq-detection/error_analysis/plots/'+ '{}.pdf'.format(task))

	return codes_length, phase_accurate_count, rgb_accurate_count, mean_err_phase, mean_err_rgb

def write_baseline_to_csv(acc_result_path):
	
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	csv_folder = acc_result_path + 'CSVs/'
	if not os.path.isdir(csv_folder):
		os.makedirs(csv_folder)

	for patient_code_path in acc_full_path:
		code = patient_code_path.split('/')[5]
		csv_save_path = csv_folder + '{}.csv'.format(code)
		csvfile = open(csv_save_path, 'wb')
		csvwriter = csv.writer(csvfile)
		csvwriter_head = ['Task','Freq','IsPeak']
		csvwriter.writerow(csvwriter_head)
		row = []
		task_paths = util.get_full_path_under_folder(patient_code_path)
		task_paths = sorted(task_paths,key=lambda x: (int(re.sub('\D','',x)),x))
		for task in task_paths:
			task_name = task.split('/')[6]
			freq_txt_path = task + 'freq.txt'
			all_lines = np.loadtxt(freq_txt_path)
			row = [task_name,all_lines[-1,:][1],all_lines[-1,:][0]]
			csvwriter.writerow(row)
		del csvwriter

def write_task_result_to_csv(tfd_result_path,acc_result_path,window_size):
	"""
	start from every task
	"""
	tasks_path = util.get_full_path_under_folder(tfd_result_path)
	tasks_path = sorted(tasks_path,key=lambda x: (int(re.sub('\D','',x)),x))
	csv_folder = '/local/guest/joint_postion/tremor-freq-detection/error_analysis/CSVs/'
	if not os.path.isdir(csv_folder):
		os.makedirs(csv_folder)

	# get all patient_codes
	patient_codes = []
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	for patient_code_path in acc_full_path:
		code = patient_code_path.split('/')[5]
		patient_codes.append(code)

	patient_codes = patient_codes[1:] # remove folder CSVs/
	# codes_num = len(patient_codes)

	peak_count_csv_path = csv_folder + 'all_tasks_peak_count.csv'
	peak_count_csvfile = open(peak_count_csv_path, 'wb')
	pc_csvwriter = csv.writer(peak_count_csvfile)
	pc_csvwriter_head = ['Task','Total_num','Acc_peak_count','Phase_peak_count','Rgb_peak_count']
	pc_csvwriter.writerow(pc_csvwriter_head)

	for task_path in tasks_path:
		pc_csvwriter_row = []
		task_name = task_path.split('/')[6]
		task_name = re.sub('_joint_tfd_61','',task_name)
		csv_save_path = csv_folder + '{}.csv'.format(task_name)
		csvfile = open(csv_save_path, 'wb')
		csvwriter = csv.writer(csvfile)
		csvwriter_head = ['Patient_code','Freq_acc','Freq_phase','Freq_rgb','IsPeak_acc','IsPeak_phase','IsPeak_rgb']
		csvwriter.writerow(csvwriter_head)
		Total_num = 0
		Acc_peak_count = 0
		Phase_peak_count = 0
		Rgb_peak_count = 0
		for code in patient_codes:

			acc_path = acc_result_path + code + '/' + task_name + '/freq.txt'
			tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(task_name,window_size) +code + '_tfd/'+ 'freq_psd_txt/'
			phasefreq_txt_path = tfd_folder + 'freq_result.txt'
			rgbfreq_txt_path = tfd_folder + 'freq_rgb.txt'
			# skip the code which doesn't have this task
			if not os.path.isfile(acc_path) or not os.path.isfile(phasefreq_txt_path) or not os.path.isfile(rgbfreq_txt_path):
				continue
			Total_num += 1
			all_phaselines = np.loadtxt(phasefreq_txt_path)
			all_rgblines = np.loadtxt(rgbfreq_txt_path)
			all_acclines = np.loadtxt(acc_path)
			# row = ['Patient_code','Freq_acc','Freq_phase','Freq_rgb','IsPeak_acc','IsPeak_phase','IsPeak_rgb']
			row = [code,all_acclines[-1,:][1],all_phaselines[-1,:][1],all_rgblines[-1,:][1],\
					all_acclines[-1,:][0],all_phaselines[-1,:][0],all_rgblines[-1,:][0]]
			csvwriter.writerow(row)
			if not all_acclines[-1,:][0] == 0:
				Acc_peak_count += 1
			if not all_phaselines[-1,:][0] == 0:
				Phase_peak_count += 1
			if not all_rgblines[-1,:][0] == 0:
				Rgb_peak_count += 1

		pc_csvwriter_row = [task_name,Total_num,Acc_peak_count,Phase_peak_count,Rgb_peak_count]
		pc_csvwriter.writerow(pc_csvwriter_row)
		del csvwriter
	del pc_csvwriter

if __name__ == "__main__":

	acc_result_path = '/local/guest/benchmark_acc/baseline_win61/'
	tfd_result_path = '/local/guest/joint_postion/tremor-freq-detection/result/'
	csv_folder = '/local/guest/joint_postion/tremor-freq-detection/error_analysis/CSVs/'
	window_size = '61'

	# write_baseline_to_csv(acc_result_path)
	# write_task_result_to_csv(tfd_result_path,acc_result_path,window_size)

	task_list = ['100-7','2_hz_lager','2_hz_hoger','Duimen_omhoog','Fingertap','Handen_in_pronatie',\
				'Maanden_terug','Pianospelen','Rust','Schrijven_links','Schrijven_rechts',\
				'Spiraal_links','Spiraal_rechts','Top_neus_links','Top_neus_rechts','Top-top','Volgen']

	mean_err_per_task_csv_path = csv_folder + 'mean_err_per_task.csv'
	# mean_err_per_task_csv_path = csv_folder + 'mean_err_per_task(only periodic).csv'

	mean_error_csvfile = open(mean_err_per_task_csv_path, 'wb')
	me_csvwriter = csv.writer(mean_error_csvfile)
	header = ['Task','Mean_err_phase','Mean_err_rgb']
	me_csvwriter.writerow(header)
	
	phase = []
	rgb = []
	total_length, phase_accurate_length, rgb_accurate_length = [], [], []
	for task in task_list:
		print 'error analysis on task {}'.format(task)
		row = []
		codes_length, phase_accurate_count, rgb_accurate_count, mean_err_phase, mean_err_rgb =\
					 task_err_analusis_all_patients(acc_result_path,tfd_result_path,task,window_size)
		row = [task,mean_err_phase,mean_err_rgb]
		me_csvwriter.writerow(row)
		total_length.append(codes_length) 
		phase_accurate_length.append(phase_accurate_count)
		rgb_accurate_length.append(rgb_accurate_count)
		phase.append(mean_err_phase)
		rgb.append(mean_err_rgb)

	del me_csvwriter
	print sum(phase)/len(phase),sum(rgb)/len(rgb)

	fig, ax = plt.subplots()

	index = np.arange(len(task_list))
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax.bar(index, total_length, bar_width,
	                alpha=opacity, color='g',
	                error_kw=error_config,
	                label='total_length')

	rects2 = ax.bar(index + bar_width, phase_accurate_length, bar_width,
	                alpha=opacity, color='r',
	                error_kw=error_config,
	                label='phase_accurate_length')

	rects3 = ax.bar(index + 2*bar_width, rgb_accurate_length, bar_width,
	                alpha=opacity, color='r',
	                error_kw=error_config,
	                label='rgb_accurate_length')

	ax.set_xlabel('Task Name')
	ax.set_ylabel('Accurate Number')
	ax.set_title('Accurate Number For All Tasks (Err < 1 HZ)')
	ax.set_xticks(index + 3*bar_width/2)
	plt.xticks(rotation=60)
	ax.set_xticklabels(task_list)
	ax.legend()

	fig.tight_layout()
	fig.savefig('/local/guest/joint_postion/tremor-freq-detection/error_analysis/plots/acc_num_per_task.pdf'