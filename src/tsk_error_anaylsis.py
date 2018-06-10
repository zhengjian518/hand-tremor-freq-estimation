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

def task_err_analusis_all_patients(acc_result_path,tfd_result_path,task,window_size):

	# step 1
	# read in the results
	acc_full_path = util.get_full_path_under_folder(acc_result_path)
	acc_full_path = sorted(acc_full_path,key=lambda x: (int(re.sub('\D','',x)),x))
	task_num = len(acc_full_path)
	acc_result = []
	phase_result = []
	rgb_result = []
	code_list = []
	peak_mark = []
	for task_idx in range(0,task_num):
		task_code = acc_full_path[task_idx].split('/')[5]
		if not os.path.isdir(acc_full_path[task_idx] + '{}/'.format(task)) \
			or not os.path.isdir(tfd_result_path + '{}_joint_tfd_{}/'.format(task,window_size) +task_code + '_tfd/'):
			continue # no such task in this patient folder
		
		accfreq_txt_path = acc_full_path[task_idx] + '{}/freq.txt'.format(task)
		all_acclines = np.loadtxt(accfreq_txt_path)
		
		if not int(all_acclines[-1,:][0]) == 0: # periodic detected
			code = task_code.split('_')[0]
			code_list.append(code)
			acc_result.append(list(all_acclines[-1,:])) # 2 columns
			peak_mark.append(task_idx)
			tfd_folder = tfd_result_path + '{}_joint_tfd_{}/'.format(task,window_size) +task_code + '_tfd/'+ 'freq_psd_txt/'
			phasefreq_txt_path = tfd_folder + 'freq_result.txt'
			all_phaselines = np.loadtxt(phasefreq_txt_path)
			phase_result.append(list(all_phaselines[-1,:])) # 3 columns

			rgbfreq_txt_path = tfd_folder + 'freq_rgb.txt'
			all_rgblines = np.loadtxt(rgbfreq_txt_path)
			rgb_result.append(list(all_rgblines[-1,:])) # 2 columns

	# step 2
	# calculate abs error
	# abs_err_phase = list(map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1]))
	# abs_err_rgb = list( map(operator.sub, np.array(rgb_result)[:,1], np.array(acc_result)[:,1]))

	abs_err_phase = list( map(abs,map(operator.sub, np.array(phase_result)[:,1], np.array(acc_result)[:,1])))
	abs_err_rgb = list( map(abs,map(operator.sub, np.array(rgb_result)[:,1], np.array(acc_result)[:,1])))

	mean_err_phase = sum(list(map(abs,abs_err_phase)))/len(abs_err_phase)
	mean_err_rgb = sum(list(map(abs,abs_err_rgb)))/len(abs_err_rgb)

	print mean_err_phase,mean_err_rgb
	print ((mean_err_rgb-mean_err_phase)/mean_err_rgb)

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


if __name__ == "__main__":

	acc_result_path = '/local/guest/benchmark_acc/baseline_win61/'
	tfd_result_path = '/local/guest/joint_postion/tremor-freq-detection/result/'
	# task = '100-7'
	# task = 'Rust'
	task_list = ['100-7','2_hz_lager','2 hz_hoger','Duimen_omhoog','Fingertap','Handen_in_pronatie',\
				'Maanden_terug','Pianospelen','Rust','Schrijven_links','Schrijven_rechts',\
				'Spiraal_links','Spiraal_rechts','Top_neus_links','Top_neus_rechts','Top-top','Volgen']
	window_size = '61'
	for task in task_list:
		print 'error analysis on task {}'.format(task)
		task_err_analusis_all_patients(acc_result_path,tfd_result_path,task,window_size)