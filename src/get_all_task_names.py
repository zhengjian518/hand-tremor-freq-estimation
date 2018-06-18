# -*- coding: utf-8 -*-
import numpy as np
import os 
import util
import re
all_tasks_name = []
path = '/media/tremor-data/TremorData_split/Tremor_data/'

all_patient_code_path = util.get_full_path_under_folder(path)

for i in range(0,len(all_patient_code_path)):
        task_names = util.get_dir_list(all_patient_code_path[i])
        all_tasks_name = all_tasks_name + task_names
        print all_patient_code_path[i]
all_name = list(set(all_tasks_name))
names = []

for task in all_name:
	if ('Extra_taak_–_') in task:
		a = re.sub('Extra_taak_–_','',task)
		names.append(a)
	elif 'Extra_taak_-_' in task:
		b = re.sub('Extra_taak_-_','',task)
		names.append(b)
	else:
		names.append(task)


alll_names = list(set(names))
sorted(alll_names)

print alll_names
print len(alll_names)
