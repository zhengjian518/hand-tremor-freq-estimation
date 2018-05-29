import csv
import numpy as np

data = np.genfromtxt("../results/T004_Links_crop/Rust/conf_arr/conf_7_1.txt",dtype=None,)

data = [list(i) for i in data]

print type(data),len(data)
print len(data[0])
a = np.array(data)
print a.shape
print type(a[0][0])