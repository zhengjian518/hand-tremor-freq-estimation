import numpy as np
import os 
import util

all_tasks_name = []
path = '/media/tremor-data/TremorData_split/Tremor_data/'

all_patient_code_path = util.get_full_path_under_folder(path)

for i in range(0,len(all_patient_code_path)):
	task_names = util.get_dir_list(all_patient_code_path[i])
	all_tasks_name = all_tasks_name + task_names

all_name = set(all_tasks_name)
name_list = sorted(all_name,key=lambda x: (int(re.sub('\D','',x)),x))

print name_list
