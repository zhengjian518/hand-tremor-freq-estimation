import matplotlib
matplotlib.use('TkAgg')
from video_in_frame import VideoInFrame
from video import Video
from io_video import IOVideo
from fftm import FFTM
from video_preprocessing import Video_Preprocessing

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import types
import math
from PIL import Image # for make colorful ball video
from random import randint,seed
from collections import deque
from scipy import io
from cycler import cycler
class Error_anlysis():
    @staticmethod
    def plt_pixel_intensity(img_path,pos_list,save_to_file_on=False,
                            save_path=''):
        """Plot pixel intensities of positions along time.

        Args:
            img_path: a string indicating the path of images.
            pos_list: a list of position in dictionary format, 
                        e.g.[{'x':0,'y':0}].
            save_to_file_on: boolean indicating whether save image to file.
            save_path: a string indicating the path of saving path.
        """        
        for pos in pos_list:
            pixel_int = pixel_intensity(img_path,[pos])[0]

            plt.figure()
            plt.plot(pixel_int)
            plt.xlabel('Frame No.')
            plt.ylabel('Pixel Intensity')
            plt.title('Intensity Curve at ({},{})'.format(str(pos['x']),
                                                            str(pos['y'])))
            plt.axis([0,len(pixel_int),0,255])

            if save_to_file_on:
                assert save_path != ''
                plt.savefig(save_path + str(pos['x']) +'_'+ str(pos['y'])
                                                                        +'.png')
                plt.close()
            else:
                plt.show(block=True)

    @staticmethod
    def plt_fft_pixel_intensity(img_path,pos_list,save_to_file_on=False,
                                save_path=''):
        for pos in pos_list:
            pixel_int = pixel_intensity(img_path,[pos])[0]


    @staticmethod
    def run(debug_type='pixel',
            video_path='../data/video/pe_data/Handen_in_pronatie/seg_7_fixed/',
            fps=30.0,freq_tremor=10,freq_noise=0,disturb_on=False,
            disturb_mag=0,window_size=120):
        from tpd import TPD
        
        video_name = video_path.split('/')[4] # , {'x':16,'y':16}
        save_path = '../results/error_analysis/' + video_name + '/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if debug_type=='pixel':
            save_path += 'pixel_intensity/'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            # pos_list = [ {'x':0,'y':0},{'x':18,'y':20}]
            pos_list=[]
            for i in range(32):
                # for j in range(32):
                pos_list.append({'x':i,'y':i})
            Error_anlysis.plt_pixel_intensity(video_path,pos_list,
                                                save_to_file_on=True,
                                                save_path=save_path)

        elif debug_type=='make_ball':
            save_path = '../results/error_analysis/ball_fps{}'.format(fps)
            save_path+= '_disturb' if disturb_on else ''
            save_path+= '_noise{}hz'.format(freq_noise) if freq_noise!=0 else ''

            for i in range(1,int(fps/2)):
                save_path_ = save_path + '/{}Hz/'.format(i)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                make_ball_video(i,freq_noise,save_path_,fps=fps,
                                disturb_on=disturb_on,disturb_mag=disturb_mag)

        elif debug_type=='make_ball_size':
            save_path = '../results/error_analysis/ball_fps{}_size'.format(fps)
            
            for i in range(3,12):
                save_path_ = save_path + '/r_{}/'.format(i)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                make_ball_video(freq_tremor,freq_noise,save_path_,fps=fps,
                                disturb_on=disturb_on,disturb_mag=disturb_mag,
                                ball_radius=i)            

        elif debug_type=='make_ball_disturb':
            save_path = \
                '../results/error_analysis/ball_fps{}_disturbmag'.format(fps)

            for i in range(0,16):
                save_path_ = save_path + '/disturb_{}/'.format(i)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                make_ball_video(freq_tremor,freq_noise,save_path_,fps=fps,
                                disturb_on=disturb_on,disturb_mag=i)
        # new built
        elif debug_type=='make_color_ball_disturb':
            save_path = \
                '../results/error_analysis/color_ball_fps{}_disturbmag'.format(fps)

            for i in range(0,16):
                save_path_ = save_path + '/disturb_{}/'.format(i)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                make_color_ball_video(freq_tremor,freq_noise,save_path_,fps=fps,
                                disturb_on=disturb_on,disturb_mag=i)
                
        elif debug_type=='make_ball_noise':
            save_path = '../results/error_analysis/ball_fps{}_noise'.format(fps)

            for i in range(0,21):
                save_path_ = save_path + '/noise_{}/'.format(i*0.1)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                if i==0:
                    make_ball_video(freq_tremor,freq_noise,save_path_,fps=fps)
                else:
                    make_ball_video(freq_tremor,freq_noise,save_path_,fps=fps,
                                noise_on=True,noise_sigma=i*0.1) #i*0.1

        # new built 
        elif debug_type=='make_color_ball_noise':
            save_path = \
                '../results/error_analysis/color_ball_fps{}_noise'.format(fps)

            for i in range(0,21):
                save_path_ = save_path + '/noise_{}/'.format(i*0.1)
                if not os.path.isdir(save_path_):
                    os.makedirs(save_path_)
                if i==0:
                    make_color_ball_video(freq_tremor,freq_noise,save_path_,fps=fps)
                else:
                    make_color_ball_video(freq_tremor,freq_noise,save_path_,fps=fps,
                                noise_on=True,noise_sigma=i*0.1) #i*0.1

        elif debug_type=='ball_test':

            video_path = '../results/error_analysis/ball_fps{}'.format(fps)
            video_path+= '_disturb' if disturb_on else ''
            video_path+= '_noise{}hz'.format(freq_noise) if freq_noise!=0 \
                                                                        else ''

            MSE = []
            for i in range(2,int(fps/2)):
                segment_img_path = video_path + '/{}Hz/'.format(i)
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(i),2)
                MSE.append(mse/len(freq_results))

            freq_range = np.arange(2,fps/2)
            MSE = np.array(MSE)
            plt.figure()
            plt.plot(freq_range,MSE)
            plt.ylabel('Mean Square Error')
            # plt.ylim([-0.3,5])
            plt.xlabel('Frequency (Hz)')
            plt.title('MSE,fps={},window_size={}'.format(fps,window_size))
            plt.show()

        elif debug_type=='ball_test_size':

            video_path = '../results/error_analysis/ball_fps{}_size'.format(fps)

            MSE = []
            for i in range(3,12):
                segment_img_path = video_path + '/r_{}/'.format(i)
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(10),2)
                MSE.append(mse/len(freq_results))

            ball_radius_range = np.arange(3,12)
            MSE = np.array(MSE)
            plt.figure()
            plt.plot(ball_radius_range,MSE)
            plt.ylabel('Mean Square Error')
            # plt.ylim([-0.3,5])
            plt.xlabel('Ball Radius (Pixel)')
            plt.title('MSE vs Ball Size, fps={}, window_size={}'.format(fps,
                                                                window_size))
            plt.show()

        elif debug_type=='ball_test_disturb':
            video_path = \
                '../results/error_analysis/ball_fps{}_disturbmag'.format(fps)

            MSE = []
            for i in range(0,16):
                segment_img_path = video_path + '/disturb_{}/'.format(i)
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(6),2)
                MSE.append(mse/len(freq_results))

            disturb_range = np.arange(0,16)
            MSE = np.array(MSE)
            plot_to_file(x=disturb_range,y=MSE,ylabel='Mean Square Error',
                xlabel='Disturb Magnitude (Pixel)',
                title='MSE vs Disturb Magnitude, fps={}, window_size={}'.format(
                                                            fps,window_size),
                save_path=video_path+'/MSE.eps')
            return MSE

        # new built
        elif debug_type=='color_ball_test_disturb':
            video_path = \
                '../results/error_analysis/color_ball_fps{}_disturbmag'.format(fps)

            MSE = []
            for i in range(0,16):
                color_ball_path = video_path + '/disturb_{}/'.format(i)
                Video_Preprocessing.video_preprocessing(color_ball_path+'color_ball.avi', filter_on = False,\
                 norm_mode = 'normalization',debug_type ='dis_and_noise')
                segment_img_path = color_ball_path+'precessed_frames/'
                # print segment_img_path
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(6),2)
                MSE.append(mse/len(freq_results))

            disturb_range = np.arange(0,16)
            MSE = np.array(MSE)
            plot_to_file(x=disturb_range,y=MSE,ylabel='Mean Square Error',
                xlabel='Disturb Magnitude (Pixel)',
                title='MSE vs Disturb Magnitude(preprocessed), fps={}, window_size={}'.format(
                                                            fps,window_size),
                save_path=video_path+'/MSE.eps')
            return MSE

        # new built
        elif debug_type=='ball_phase_test_disturb':
            video_path = \
                '../results/error_analysis/ball_fps{}_disturbmag'.format(fps)

            path_list = list()
            for i in range(0,16):
                print 'makeing phase images in disturb_{}'.format(i)
                ball_path = video_path + '/disturb_{}/'.format(i)
                scl_ori_path = processing.get_phase_images(ball_path + 'ball.avi',ball_path,debug_type,i)
                path_list.append(scl_ori_path)


            MSE_list = list()
            num_scale = 5
            num_orient  = 4

            print 'Start frequency analysizing: '
            for scl in range (1,num_scale-1):
                for ori in range(0,num_orient):
                    MSE = []
                    print 'FA: scl_{}'.format(scl)+'_ori_{}'.format(ori)
                    for i in range(0,16):
                        path_scl_ori = '../results/error_analysis/phase_ball_fps30.0_disturbmag/disturb_{}/'.format(i)
                        path = path_scl_ori + 'scale_{}_ori_{}/'.format(scl,ori)
                        # path = path_list[i]
                        freq_results = Error_anlysis.tfd_controled_test(
                                                            path,window_size,
                                                            window_size/2,fps,
                                                            is_ball_video=True)
                        mse = 0.0
                        for freq in freq_results:
                            mse += math.pow(float(freq)-float(6),2)
                        MSE.append(mse/len(freq_results))
                    # print MSE
                    MSE_list.append(MSE)

            
            # disturb_range = np.arange(0,16)
            # for plt in range(0,3):
            #     MSE_ori_list = []
            #     legends=['0','1/4 pi','1/2 pi','3/4 pi']
            #     for ori in range(0,4):
            #         leftpop = MSE_list.pop(0)
            #         MSE_ori_list.append(leftpop)
            #     n_plot_to_file(disturb_range,MSE_ori_list,xlabel = 'Disturb Magnitude (Pixel)',ylabel ='Mean Square Error',\
            #             legends = legends,\
            #             title = 'MSE vs Disturb Magnitude (phase scale_{}), fps={}, window_size={}'.format(plt,fps,window_size),\
            #             plot_path = '../results/error_analysis/phase_ball_fps30.0_disturbmag/'+ 'MSE_scale{}.eps'.format(plt))

            return MSE_list


        elif debug_type=='ball_test_disturb_sim':
            video_path = '../results/error_analysis/ball_fps{}_disturbmag'.format(fps)

            MSE,MSE_sim = [],[]
            for i in range(0,16):
                segment_img_path = video_path + '/disturb_{}/'.format(i)
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                freq_results_sim = TPD.tremor_period_detec_sim_fft_ball(
                                                segment_img_path,window_size,
                                                window_size/2,fps)
                mse,mse_sim = 0.0, 0.0

                for freq in freq_results:
                    mse += math.pow(float(freq)-6.0,2)
                for freq in freq_results_sim:
                    mse_sim += math.pow(float(freq)-6.0,2)
                MSE.append(mse/len(freq_results))
                MSE_sim.append(mse_sim/len(freq_results))

            disturb_range = np.arange(0,16)
            MSE = np.array(MSE)
            MSE_sim = np.array(MSE_sim)
            multi_plot_to_file(x1=disturb_range,y1=MSE,
                x2=disturb_range,y2=MSE_sim,
                ylabel='Mean Square Error',
                xlabel='Disturb Magnitude (Pixel)',
                legends=['TFD','Similarity'],
                title='MSE vs Disturb Magnitude, fps={}, window_size={}'.format(
                                                            fps,window_size),
                save_path=video_path+'/MSE.eps')
            return MSE,MSE_sim

        elif debug_type=='ball_test_noise':
            video_path ='../results/error_analysis/ball_fps{}_noise'.format(fps)

            MSE = []
            for i in range(0,21):
                segment_img_path = video_path + '/noise_{}/'.format(0.1*i)
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(6),2)
                MSE.append(mse/len(freq_results))

            sigma_range = np.arange(0,21) * 0.1
            MSE = np.array(MSE)
            plot_to_file(x=sigma_range,y=MSE,ylabel='Mean Square Error',
                xlabel='Guassian Noise '+r'$\sigma$',
                title='MSE vs Guassian Noise '+r'$\sigma$'+\
                            ', fps={}, window_size={}'.format(fps,window_size),
                save_path=video_path+'/MSE.eps')

            return MSE

        # new built
        elif debug_type=='color_ball_test_noise':
            video_path ='../results/error_analysis/color_ball_fps{}_noise'.format(fps)

            MSE = []
            for i in range(0,21):
                color_ball_path = video_path + '/noise_{}/'.format(0.1*i)
                Video_Preprocessing.video_preprocessing(color_ball_path+'color_ball.avi', filter_on = False,\
                                                         norm_mode = 'normalization',debug_type = 'dis_and_noise')
                segment_img_path = color_ball_path+'precessed_frames/'
                # print segment_img_path
                freq_results = Error_anlysis.tfd_controled_test(
                                                segment_img_path,window_size,
                                                window_size/2,fps,
                                                is_ball_video=True)
                mse = 0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(6),2)
                MSE.append(mse/len(freq_results))

            sigma_range = np.arange(0,21) * 0.1
            MSE = np.array(MSE)
            plot_to_file(x=sigma_range,y=MSE,ylabel='Mean Square Error',
                xlabel='Guassian Noise '+r'$\sigma$',
                title='(preprocessed) MSE vs Guassian Noise '+r'$\sigma$'+\
                            ', fps={}, window_size={}'.format(fps,window_size),
                save_path=video_path+'/preprocessed_MSE.eps')

            return MSE

        # new built
        elif debug_type=='ball_phase_test_noise':
            video_path = \
                '../results/error_analysis/ball_fps{}_noise'.format(fps)

            path_list = list()
            for i in range(0,21):
                print 'makeing phase images in noise_{}'.format(i*0.1)
                ball_path = video_path + '/noise_{}/'.format(0.1*i)
                scl_ori_path = Video_Preprocessing.get_phase_images(ball_path + 'ball.avi',ball_path,debug_type,i)
                path_list.append(scl_ori_path)


            MSE_list = list()
            num_scale = 5
            num_orient  = 4

            print 'Start frequency analysizing: '

            for scl in range (1,num_scale-1):
                for ori in range(0,num_orient):
                    MSE = []
                    print 'FA: scl_{}'.format(scl)+'_ori_{}'.format(ori)
                    for i in range(0,21):
                        path_scl_ori = '../results/error_analysis/phase_ball_fps30.0_noise/noise_{}/'.format(0.1*i)
                        path = path_scl_ori + 'scale_{}_ori_{}/'.format(scl,ori)
                        # path = path_list[i]
                        freq_results = Error_anlysis.tfd_controled_test(
                                                            path,window_size,
                                                            window_size/2,fps,
                                                            is_ball_video=True)
                        mse = 0.0
                        for freq in freq_results:
                            mse += math.pow(float(freq)-float(6),2)
                        MSE.append(mse/len(freq_results))
                    # print MSE
                    MSE_list.append(MSE)

            
            # sigma_range = np.arange(0,21) *0.1
            # for plt in range(0,3):
            #     MSE_ori_list = []
            #     legends=['0','1/4 pi','1/2 pi','3/4 pi']
            #     for ori in range(0,4):
            #         leftpop = MSE_list.pop(0)
            #         MSE_ori_list.append(leftpop)
            #     n_plot_to_file(sigma_range,MSE_ori_list,xlabel = 'Guassian Noise '+r'$\sigma$',ylabel ='Mean Square Error',\
            #             legends = legends,\
            #             title = 'MSE vs Guassian Noise '+r'$\sigma$ '+' (phase scale_{}), '.format(plt)+\
            #                     'fps={}, window_size={}'.format(fps,window_size),\
            #             plot_path = '../results/error_analysis/phase_ball_fps30.0_noise/'+ 'MSE_scale{}.eps'.format(plt))

            return MSE_list

        elif debug_type=='ball_test_noise_sim':
            video_path = '../results/error_analysis/ball_fps{}_noise'.format(fps)

            MSE,MSE_sim = [],[]
            for i in range(0,21):
                segment_img_path = video_path + '/noise_{}/'.format(0.1*i)
                freq_results = Error_anlysis.tfd_controled_test(
                                            segment_img_path,window_size,
                                            window_size/2,fps,
                                            is_ball_video=True)
                freq_results_sim = TPD.tremor_period_detec_sim_fft_ball(
                                            segment_img_path,window_size,
                                            window_size/2,fps)
                mse,mse_sim = 0.0,0.0
                for freq in freq_results:
                    mse += math.pow(float(freq)-float(6),2)
                for freq in freq_results_sim:
                    mse_sim += math.pow(float(freq)-float(6),2)
                MSE.append(mse/len(freq_results))
                MSE_sim.append(mse_sim/len(freq_results))

            sigma_range = np.arange(0,21) * 0.1
            MSE = np.array(MSE)
            MSE_sim = np.array(MSE_sim)
            multi_plot_to_file(x1=sigma_range,y1=MSE,x2=sigma_range,y2=MSE_sim,
                ylabel='Mean Square Error',
                xlabel='Guassian Noise '+r'$\sigma$',
                legends=['TFD','Similarity'],
                title='MSE vs Guassian Noise '+r'$\sigma$'+\
                            ', fps={}, window_size={}'.format(fps,window_size),
                save_path=video_path+'/MSE.eps')

            return MSE,MSE_sim


    # TFD Controlled Experiment: single-frame confidence approach + accumulated-PSD
    @staticmethod
    def tfd_controled_test(segment_img_path,window_size,noverlap,joint_No,
                            fps=30,is_ball_video=False):
        """This version assumes that joint image has already been cropped and 
            saved.
        
        Args:
            segment_img_path: a path string to the cached images.
            window_size, noverlap: integers of set-up.
            joint_No: No. of the joint to test.
            fps: video frame rate.
            is_ball_video: boolean indicating whether it is a ball video.
        """

        stride = window_size - noverlap
        video_lwri = VideoInFrame(segment_img_path,'png',fps)
        io_video = IOVideo()  # blurring_on=True,sigmaX=5,sigmaY=5,ksize=0
        video_name = segment_img_path.split('/')[5]
        video_code = segment_img_path.split('/')[4]
        fftm = FFTM(window_size,video_lwri.FPS)

        stride_count = stride-1
        freq_maps = deque([])
        conf_maps = deque([])
        # freq_maps = deque([])

        if is_ball_video:
            conf_map = np.zeros((video_lwri.HEIGHT,video_lwri.WIDTH))
            sigma_gauss = 21
            for x in range(video_lwri.WIDTH):
                for y in range(video_lwri.HEIGHT):
                    dist_sq = (x-video_lwri.WIDTH/2)*(x-video_lwri.WIDTH/2) + \
                                (y-video_lwri.HEIGHT/2)*(y-video_lwri.HEIGHT/2)
                    exponent = dist_sq / 2.0 / sigma_gauss / sigma_gauss
                    conf_map[y,x] = math.exp(-exponent)
        else:
            conf_map = np.loadtxt( \
                '../data/video/pe_data_/{}/{}/conf_{}/conf_{}_1.txt'.format( \
                                    video_code,video_name,joint_No,joint_No))
        # rows,cols = conf_map.shape
        # M = np.float32([[1,0,0],[0,1,15]])
        # conf_map = cv2.warpAffine(conf_map,M,(cols,rows))

        # Plot Conf Map 
        # plt.figure()
        # plt.imshow(conf_map)
        # plt.show(block=True)
        # conf_map = np.ones((video_lwri.HEIGHT,video_lwri.WIDTH))

        freq_results = []
        gap = []
        len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
        freq_series = np.fft.fftfreq( window_size, 
                                        d=1/float(video_lwri.FPS) )[0:len_half]

        weighted_fft_seq_sum = None
        while(video_lwri.next_frame_index<video_lwri.FRAME_COUNT):
            # cv2.GaussianBlur(io_video.get_video_frames(video_lwri,1),(0,0),5)
            # fftm.add_frames(cv2.GaussianBlur(
                            # io_video.get_video_frames(video_lwri,1),(0,0),5)) # when add frame at the end it will remove the frame at the head
            

            fftm.add_frames(io_video.get_video_frames(video_lwri,1))# changed in 27th Feb
            

            # print 'FFTM has {} frames'.format(video_lwri.next_frame_index)

            if stride-1<=video_lwri.next_frame_index<= \
                    int(video_lwri.FRAME_COUNT)-1-stride and \
                    (video_lwri.next_frame_index+1)%stride == 0:
                # print '    Conf!'
                if not is_ball_video:
                    conf_map = np.loadtxt(
                        '../data/video/pe_data_/{}/{}/conf_{}/conf_{}_{}.txt'.format(
                                    video_code,video_name,joint_No,joint_No,
                                    video_lwri.next_frame_index-1))
                    if conf_map.shape[0]>video_lwri.HEIGHT or \
                                            conf_map.shape[1]>video_lwri.WIDTH:
                        center_y = conf_map.shape[0]/2
                        center_x = conf_map.shape[1]/2
                        height,width = video_lwri.HEIGHT/2, video_lwri.WIDTH/2
                        conf_map = conf_map[center_y-height:center_y+height,
                                            center_x-width:center_x+width]
                        conf_map = conf_map / np.sum(conf_map)
                    elif conf_map.shape[0]<video_lwri.HEIGHT or \
                                            conf_map.shape[1]<video_lwri.WIDTH:
                        print("Warning: Confidence map {} smaller than Freq map {}! EXIT!".format(
                                            conf_map.shape,freq_map.shape[-2:]))
                        return

                conf_maps.append(conf_map)

            if fftm.frame_num == window_size: # always 120 frames after reach 120
                stride_count += 1
                if stride_count == stride: # FA every 60 new frames
                    stride_count = 0
                    # print '    FFT!'
                    # freq_map = (fftm.fft_frames_sequence(filter_on=True))[0] 
                    freq_map = (fftm.fft_frames_sequence(filter_on=True,threshold_on=True))[0] 
                    freq_maps.append(freq_map)
        
            if len(freq_maps)!=0 and len(conf_maps)!=0:
                fft_sequences = freq_maps.popleft()
                conf_mapp = conf_maps.popleft()
                weighted_fft_sequence = fft_sequences * conf_mapp # window_size/2 *width*height * width*height
                weighted_fft_sequence = np.sum(weighted_fft_sequence,axis=(1,2))
                if weighted_fft_seq_sum is None:
                    weighted_fft_seq_sum = weighted_fft_sequence
                else:
                    weighted_fft_seq_sum += weighted_fft_sequence

                # PSD
                # plt.figure()
                # plt.plot(freq_series,weighted_fft_sequence/weighted_fft_sequence.max())
                # plt.xlabel('Frequency (Hz)')
                # plt.ylabel('Power Spectral Density')
                # # plt.title('F={},window_size={},overlap={},fps={}'.format(6,window_size,noverlap,video_lwri.FPS))
                # # plt.title('PSD-Noise')
                # plt.show()

                # freq_i = freq_series[np.argmax(weighted_fft_sequence)] 
                freq_i = freq_series[np.argmax(weighted_fft_seq_sum)] # changed in 11th March, better result

                freq_results.append(str(freq_i))

                # print('Freq: {} Hz'.format(freq_i))
                max_weighted_fft_seq_sum = np.max(weighted_fft_seq_sum)/weighted_fft_seq_sum.max()
                mean_weighted_fft_seq_sum = np.mean(weighted_fft_seq_sum,0)/weighted_fft_seq_sum.max()
                gap_element = max_weighted_fft_seq_sum - mean_weighted_fft_seq_sum
                gap.append(gap_element)


        # Avg PSD
        if is_ball_video:
            plot_to_file(x=freq_series,
                y=weighted_fft_seq_sum/weighted_fft_seq_sum.max(),
                xlabel='Frequency (Hz)',ylabel='Power Spectral Density',
                title='Average PSD {}'.format(os.path.basename(\
                                        os.path.normpath(segment_img_path))),
                save_path=segment_img_path+'avgpsd_.eps')
        print freq_results
        print '\nGaps between peak and mean are:\n {} '.format(gap)
        print '\nAverage gap between peak and mean are:\n {}'.format(np.sum(gap)/len(gap))
        # return [freq_results,weighted_fft_seq_sum/weighted_fft_seq_sum.max(),freq_series]
        return freq_results

        # t = np.arange(len(freq_results))
        # ground_truth = np.ones(len(freq_results))*12
        # freq_results = np.array(freq_results)
        # plt.figure()
        # plt.plot(t,freq_results,t,ground_truth)
        # plt.ylim([0,15])
        # plt.legend(['Estimation','Ground Truth'])
        # plt.ylabel('Frequency/Hz')
        # plt.xlabel('Time/s')
        # plt.title('Frequency Estimation')
        # plt.show()

    # TFD Controlled Experiment: single-frame confidence approach + accumulated-PSD
    @staticmethod
    def tfd_controled_test_sliding_pe(segment_img_path,window_size,noverlap,
                                        joint_No,fps=30):
        """This version assumes that joint prediction has been done and saved."""
        video_code = segment_img_path.split('/')[4]
        video_name = segment_img_path.split('/')[5]

        video_lwri = VideoInFrame(segment_img_path,'png',fps)
        fftm = FFTM(window_size,video_lwri.FPS)

        stride = int(window_size - noverlap)
        len_half = window_size/2 if window_size%2==0 else (window_size+1)/2
        freq_series = np.fft.fftfreq( window_size, 
                                        d=1/float(video_lwri.FPS) )[0:len_half]
        pred = np.loadtxt('../data/video/pe_data_/{}/{}/prediction_arr/pred_{}.txt'.format(
                                                    video_code,video_name,1))
        box_size = int(math.sqrt( math.pow(pred[0,0]-pred[1,0],2) + \
                                     math.pow(pred[0,1]-pred[1,1],2) ) / 2 )

        freq_results = []
        weighted_fft_seq_sum = None

        for i in range(0,int(video_lwri.FRAME_COUNT/stride)-1):
            # print("No_start:{},frame_num:{},No_pe:{},box_size:{}".format(i*stride,window_size,i*stride+window_size/2,box_size))

            # No_start:0-based, No_pe: 0-based
            fftm.add_frames(get_cropped_frames(video_code,video_name,joint_No,
                                    No_start=i*stride,frame_num=window_size,
                                    No_pe=i*stride+window_size/2,
                                    box_size=box_size))
            freq_map = (fftm.fft_frames_sequence(filter_on=True))[0]

            conf_map = np.loadtxt(
                '../data/video/pe_data_/{}/{}/conf_{}/conf_{}_{}.txt'.format(
                                    video_code,video_name,joint_No,joint_No,
                                    i*stride+window_size/2))
            if conf_map.shape[0]>freq_map.shape[1] or \
                                         conf_map.shape[1]>freq_map.shape[2]:
                center_y,center_x = conf_map.shape[0]/2, conf_map.shape[1]/2
                height,width = freq_map.shape[1]/2, freq_map.shape[2]/2
                conf_map = conf_map[center_y-height:center_y+height,
                                                center_x-width:center_x+width]
            elif conf_map.shape[0]<freq_map.shape[1] or \
                                            conf_map.shape[1]<freq_map.shape[2]:
                print("Warning: Confidence map-No.{} {} smaller than Freq map {}! EXIT!".format(
                    i*stride+window_size/2,conf_map.shape,freq_map.shape[-2:]))
                return

            weighted_fft_sequence = freq_map * conf_map # window_size/2 *width*height * width*height
            weighted_fft_sequence = np.sum(weighted_fft_sequence,axis=(1,2))
            if weighted_fft_seq_sum is None:
                weighted_fft_seq_sum = weighted_fft_sequence
            else:
                weighted_fft_seq_sum += weighted_fft_sequence

            # PSD
            # plt.figure()
            # plt.plot(freq_series,weighted_fft_sequence/weighted_fft_sequence.max())
            # plt.xlabel('Frequency/Hz')
            # plt.ylabel('Power Spectral Density')
            # plt.title('f={},window_size={},overlap={},fps={}'.format(6,window_size,noverlap,video_lwri.FPS))
            # plt.show()

            freq_i = freq_series[np.argmax(weighted_fft_sequence)]
            freq_results.append(str(freq_i))
            # print('Freq: {} Hz'.format(freq_i))

        # Avg PSD
        # plt.figure()
        # plt.plot(freq_series,weighted_fft_seq_sum/weighted_fft_seq_sum.max())
        # plt.xlabel('Frequency/Hz')
        # plt.ylabel('Power Spectral Density')
        # plt.title('Accumulated PSD')
        # plt.show()
        print freq_results
        return freq_results

        # t = np.arange(len(freq_results))
        # ground_truth = np.ones(len(freq_results))*12
        # freq_results = np.array(freq_results)
        # plt.figure()
        # plt.plot(t,freq_results,t,ground_truth)
        # plt.ylim([0,15])
        # plt.legend(['Estimation','Ground Truth'])
        # plt.ylabel('Frequency/Hz')
        # plt.xlabel('Time/s')
        # plt.title('Frequency Estimation')
        # plt.show()

def pixel_intensity(img_path,pos_list):
    """Return pixel intensity of a list of postion.

    Args:
        img_path: a string indicating the path of images.
        pos_list: a list of position in dictionary format, e.g.[{'x':0,'y':0}].

    Return:
        pixel_int_series: a list of pixel intensities at positions.
    """
    assert isinstance(pos_list,types.ListType)

    video_lwri = VideoInFrame(img_path,'png',30)
    io_video = IOVideo()
    pixel_int_series = []
    for pos in pos_list:
        assert 0<=pos['x']<=video_lwri.WIDTH and 0<=pos['y']<=video_lwri.HEIGHT
        pixel_int_series.append([])

    while(video_lwri.next_frame_index<video_lwri.FRAME_COUNT):
        frame = io_video.get_video_frames(video_lwri,1)       
        pixel_int_it = iter(pixel_int_series)
        for pos in pos_list:
            pixel_int = next(pixel_int_it)
            pixel_int.append(frame[pos['x'],pos['y']])

    return pixel_int_series

def make_ball_video(freq_tremor,freq_noise,save_path,fps=30.0,
                    disturb_on=False,disturb_mag=2,noise_on=False,
                    noise_sigma=1,ball_radius=8):
    """Make ball video with sine tremor.

    Args:
        freq_tremor: a float indicating the tremor frequency.
        freq_noise: a float indicating the tremor frequency.
        save_path: a string indicating the path of saving path.
        fps: a float number indicating frame rate.
        disturb_on: a boolean value indicating if adding disturbance to ball 
                    tremor.
        ball_radius: an integer indicating the radius of the ball.
    """
    # Ball Image Param
    video_height,video_width = 32,32 # pixel
    bkg_clr = 220.0 # 0-255 grayscale
    ball_clr = 130.0 # 0-255 grayscale

    # Dynamic Param
    f_tremor = freq_tremor # Hz
    a_tremor = 5 # pixel
    offset = disturb_mag if disturb_on else 0
    f_noise = freq_noise
    a_noise = 2
    # a_noise = 0  # changed to 0 in 22ed Feb

    noise_sigma = noise_sigma if noise_on else 0

    # Video Param
    video_fps = fps
    len_video_sec = 10 # sec
    len_video_frame = len_video_sec * int(video_fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(save_path+'ball.avi',fourcc, video_fps, 
                                        (int(video_width),int(video_height)))

    for i in range(len_video_frame):
        frame = np.ones((video_height,video_width),dtype=np.float32) * bkg_clr
        
        offset_x,offset_y = randint(-offset,offset),randint(-offset,offset)
        center_x,center_y = video_width/2+offset_x ,video_height/2 + \
                int(a_tremor * math.sin(2*math.pi*f_tremor * (i/video_fps))) +\
                int(a_noise * math.sin(2*math.pi*f_noise * (i/video_fps))) + \
                offset_y
        # Single Way
        # center_x = video_width/2+offset_x
        # center_y = video_height/2  + int(a_tremor * math.cos(2*math.pi*f_tremor * ( i%(video_fps/f_tremor/2) /video_fps)))

        # cv2.circle(frame,(center_x,center_y),ball_radius,ball_clr,-1)
        cv2.circle(frame,(center_x,center_y),ball_radius,ball_clr,-1)
        if noise_on:
            frame = frame + 255*np.random.normal(0, noise_sigma, size=frame.shape)
        cv2.imwrite(save_path+'{}.png'.format(i),frame)
        out_video.write(cv2.applyColorMap(np.uint8(frame), cv2.COLORMAP_BONE))

def make_color_ball_video(freq_tremor,freq_noise,save_path,fps=30.0,
                    disturb_on=False,disturb_mag=2,noise_on=False,
                    noise_sigma=1,ball_radius=8):
        """Make ball video with sine tremor.

        Args:
            freq_tremor: a float indicating the tremor frequency.
            freq_noise: a float indicating the tremor frequency.
            save_path: a string indicating the path of saving path.
            fps: a float number indicating frame rate.
            disturb_on: a boolean value indicating if adding disturbance to ball 
                        tremor.
            ball_radius: an integer indicating the radius of the ball.
        """
        # Ball Image Param
        video_height,video_width = 32,32 # pixel
        bkg_clr = 220.0 # 0-255 grayscale
        # ball_clr = 130.0 # 0-255 grayscale
        ball_clr = (102,178,255) # color ball
        # Dynamic Param
        f_tremor = freq_tremor # Hz
        a_tremor = 5 # pixel
        offset = disturb_mag if disturb_on else 0
        f_noise = freq_noise
        a_noise = 2

        noise_sigma = noise_sigma if noise_on else 0

        # Video Param
        video_fps = fps
        len_video_sec = 10 # sec
        len_video_frame = len_video_sec * int(video_fps)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(save_path+'color_ball.avi',fourcc, video_fps, 
                                            (int(video_width),int(video_height)), isColor = True) # isColor = True for makeing colorful ball video

        for i in range(len_video_frame):
            # print i
            frame = np.ones((video_height,video_width,3),dtype=np.float32) * bkg_clr
            
            offset_x,offset_y = randint(-offset,offset),randint(-offset,offset)
            center_x,center_y = video_width/2+offset_x ,video_height/2 + \
                    int(a_tremor * math.sin(2*math.pi*f_tremor * (i/video_fps))) +\
                    int(a_noise * math.sin(2*math.pi*f_noise * (i/video_fps))) + \
                    offset_y
            # Single Way
            # center_x = video_width/2+offset_x
            # center_y = video_height/2  + int(a_tremor * math.cos(2*math.pi*f_tremor * ( i%(video_fps/f_tremor/2) /video_fps)))

            cv2.circle(frame,(center_x,center_y),ball_radius,ball_clr,-1)
            if noise_on:
                frame = frame + 255*np.random.normal(0, noise_sigma, size=frame.shape)
                
            # color_frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            cv2.imwrite(save_path+'{}.png'.format(i),frame)
            out_video.write(np.uint8(frame))
            # out_video.write(cv2.applyColorMap(np.uint8(frame), cv2.COLORMAP_HSV))


def get_cropped_frames(video_code,video_name,joint_No,No_start,
                        frame_num,No_pe,box_size):
    """Get cropped joint frames from video.

    Args:
        video_name: a string indicating the name of video in T008 series, 
                    e.g. 'Rust'.
        No_start: an integer indicating which frame to start from, 0-based.
        frame_num: an integer indicating how many frames to extract.
        No_pe: an integer indicating which frame used as prediction,0-based.
        box_size: an integer indicating half size of the box.
    Return:
        frames: a numpy array including cropped frames in gray scale, 
                shape:frame_num*height*width. 
    """

    video_path = '../data/video/{}/{}/kinect.avi'.format(video_code,video_name)
    video_pe = Video(video_path)
    io_video = IOVideo(resizing_on=True,scale=368/video_pe.HEIGHT,
                fps=30,height=368,width=368*video_pe.WIDTH/video_pe.HEIGHT)
    video_pe.set_next_frame_index(No_start)
    
    pred = np.loadtxt(
            '../data/video/pe_data_/{}/{}/prediction_arr/pred_{}.txt'.format(
                                                video_code,video_name,No_pe+1)) # add 1 because cached file is 1-based
    pred_x,pred_y = int(pred[joint_No,1]),int(pred[joint_No,0])

    frames = []
    while(len(frames)<frame_num):
        frame = io_video.get_video_frames(video_pe,1,grayscale_on=True)
        joint_frame = frame[pred_y-box_size:pred_y+box_size,
                            pred_x-box_size:pred_x+box_size]
        # cv2.imwrite('../results/tmp/{}.png'.format(len(frames)+1),joint_frame)
        joint_frame = cv2.GaussianBlur(joint_frame,(0,0),5)
        frames.append(joint_frame)
    return np.array(frames)

def plot_to_file(x,y,xlabel,ylabel,title,save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['b']));
    ax.plot( x, y )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(save_path, format='eps')
    plt.close(fig)

def multi_plot_to_file(x1,y1,x2,y2,xlabel,ylabel,legends,title,save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(cycler('color', ['b', 'r']));
    ax.plot( x1, y1, x2, y2 )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legends)
    fig.savefig(save_path, format='eps')
    plt.close(fig)

def n_plot_to_file(run_time,x,list,xlabel,ylabel,legends,title,plot_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(cycler('color', ['b', 'r']))
    lenth = len(list)
    for i in range(0,lenth):
        MSE = list[i]
        y = np.array(MSE)/run_time
        ax.plot(x,y,label = 'i')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legends)
    fig.savefig(plot_path, format='eps')
    plt.close(fig)