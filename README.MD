# Phase-Based Method For Tremor Frequency Detection From Video
The experiment setup is as follows:
* Exp 1.1) Eulerian freq estimation: pose estimation + tracker + phase-hands + FFT over phase-hands.
* Exp 1.2) Eulerian freq estimation: pose estimation + tracker + RGB + FFT over RGB.
Compare exp1.1 with 1.2 to show the effect of phase.

* Exp 2.1) Lagrangian freq estimation: pose estimation at all frames + FFT over joints (x,y) ( NO SMOOTH).
* Exp 2.2) Lagrangian freq estimation: pose estimation at all frames + tracker+ FFT over joints (x,y) ( plus tracker).
Compare exp1.1 with 2.1 to show the importatnce of using eulerian coordinate.
## Package Required: 
* [OpenCV v3.2.0 or higher version](http://opencv.org/releases.html) - for image processing, please compile from [source code](https://github.com/opencv/opencv), otherwise you will miss some video processing functions. 
* [Caffe](http://caffe.berkeleyvision.org/installation.html) - for pose estimation part, if you want to retrain the CPM network, please install customized version from the [repository](https://github.com/shihenw/convolutional-pose-machines-release).
* [Perceptual](https://github.com/andreydung/Steerable-filter) - using complex steerable pyramid to generate phase images, install with: sudo pip install perceptual.
* [Filterpy](https://github.com/rlabbe/filterpy) - for smoothing the joint trajectory using Kalman Filter, install with: sudo pip install filterpy.
* [Numpy & Scipy](https://www.scipy.org)
* [ConfigObj]() 

## Installation
Run code to get model files.
```sh
$ src/get_model.sh
```
If above method failed, please contact me for "file_id" on Google drive, and download model files by:
```
$ dl_model_from_gd.py file_id, ../model/_trained_MPI/ OR ../model/_trained_person_MPI/
```

## Basic use
The code in folder 'src/' for the project mainly includes three procedures, which are pose estimation, frequency estimation and error analysis.
1. For pose estimation, change the 'patient_code_folder' in 'pe_and_crop.py' to a path of patient folder. it can generate all results for all tasks in this folder by running below code. Apart from processing for all tasks, it can perform pose estimation for scattered videos by changing 'full_video_list' in 'pe_and_crop.py'.
```sh
$ python src/pe_and_crop.py
```
The results for each task include:
* 'conf_arr/' --- fixed number (same to the video frame number) of .bin files which contains cropped confidence map(56*56) from CPM.
* 'prediction_arr/' --- fixed number (same to the video frame number) of .txt files which contains joint positions from CPM. 
Each row in the txt file corresponds to a joint in the joints_string = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank']. So the 5th row for "Rwri", and 8th row for "Lwri".
In the txt file, the first column is y coordinate(Top->Down), the second is x coordinate(Left->Right).
* 'segment_img/' --- Contains the cropped joint frames.
2. For frequency estimation, assume that the first step 'pose estimation' has done, change the 'task_to_process' in 'tfd_phase_joint_kal_y_cor.py' to a task name, for example 'task_to_process = 'Spiraal_links', then run:
```sh
$ python src/tfd_phase_joint_kal_y_cor.py
```
In the result folders, the 'freq_psd_txt/' folder contains the frequency and PSD information.
* freq_result.txt --- The numbers in the first column are counts of strides, the numbers in the second column are frequency results in each temporal window, the numbers in the third column records the channel index for the freq_result in column 2.
The last row in freq_result.txt records the freq_result for the whole video, the number in column 1 indicates whether it is a peak frequency.
* psd.txt --- Records a mixed PSD of all channels for the whole video.
* freq_rgb.txt --- Records the frequency information from RGB channel.
* psd.txt --- Records the PSD of RGB channel for the whole video.

3. To get the smoothed trajectory of joint positions overtime for exp2_2, run 
```sh
$ python src/exp2_2_prepare.py
```
The results will be recorded in folder 'smooth_trajectory/'.

4. For performing error analysis, the baseline frequency of each video will be calculated using the data collected from the accelerator. Running the code below can get the frequency and PSD results for exp2_1 and exp2_2:
```sh
$ matlab src_matlab/exp2_1.m
$ matlab src_matlab/exp2_2.m
```
5. After step 1 to step 4 have been finished, error analysis can start by running:
```sh
$ python src/tsk_error_anaylsis.py
```
The error results are collected in files under folder 'Error_Analysis/'.
Notice when changing the result pathes in 'tsk_error_anaylsis.py' if needed, the 'task_name' will also need to be carefully modified, otherwise it report errors about missing files.


## Contact
Jian Zheng - zhengjian.tud@gmail.com
