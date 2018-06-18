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
all_name = set(all_tasks_name)
for all_task in all_name:

sorted(all_name)
print all_name
print len(all_name)