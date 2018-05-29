import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy import signal
from cycler import cycler
import os

class FFTM:
    """Class for Fast Fourior Transform Machine for grayscale images.
    
    Attributes:
        frames_sequence: a numpy array consisting of several grayscale frames.
        frame_num: the number of frames in the sequence.
        WINDOW_SIZE: a constant integer indicating the maximum number of frames 
                        that the machine could keep.
        SAMPLE_FREQ: a constant integer indicating the sampling frequency.
    """
    def __init__(self,window_size,sample_freq,frame_sequence=None):
        """Init FFTM."""

        self.frame_sequence = frame_sequence
        if frame_sequence is not None:
            self.frame_num = self.frame_sequence.shape[0]
        else:
            self.frame_num = 0
        self.WINDOW_SIZE = int(window_size)
        self.SAMPLE_FREQ = int(sample_freq)

    def add_frames(self,frames):
        """Add frames at the end of the frame sequence.

        Args:
            frames: a numpy array consisting of several grayscale frames.
        """ 
        num_frames_to_add = 1 if len(frames.shape)<=2 else frames.shape[0]

        if self.frame_sequence is not None:
            assert frames.shape[-2:] == self.frame_sequence.shape[-2:]

            if self.frame_num + num_frames_to_add>self.WINDOW_SIZE:
                self.remove_frames(self.frame_num + num_frames_to_add - \
                                    self.WINDOW_SIZE)
            self.frame_sequence = np.insert(self.frame_sequence, 
                                self.frame_sequence.shape[0], frames, axis=0)    
        else:
            self.frame_sequence = np.expand_dims(frames, axis=0) \
                                            if num_frames_to_add==1 else frames
        self.frame_num += num_frames_to_add

    def remove_frames(self,num_to_remove):
        """Remove frames from the head of the frame sequence.

        Args:
            num_to_remove: an integer indicating how many frames to remove.
        """

        assert type(num_to_remove)==int and num_to_remove <= self.frame_num

        self.frame_sequence = np.delete(self.frame_sequence, 
                                        np.s_[0:num_to_remove:1], axis=0)
        self.frame_num -= num_to_remove

    def fft_frames_sequence(self,filter_on=False,threshold_on=False):
        """Do FFT to frame sequence.

        Args:
            filter_on: a boolean value indicating whether apply a butter-worth 
                        filter.
            threshold_on: a boolean value indicating whether thresholding on FFT
                         amplitude.
        
        Returns:
            psd: a 3D matrix indicating PSD of 2D time series.
            freq: a 1D vector indicating frequencies.
            psd_max: a 2D matrix indicating maximum power in pixel series.
            freq_max_psd: a 2D frequency matrix with the maximum psd amplitude.
        """

        # Normalize
        frame_sequence_norm = np.subtract(self.frame_sequence, 
                                        np.mean(self.frame_sequence, axis=0)) # along y = 0 axis
        
        # Bandpass Filter
        if filter_on:
            order,low,high = 4,2/(0.5 * self.SAMPLE_FREQ), \
                                                    14.5/(0.5 *self.SAMPLE_FREQ)
            b, a = signal.butter(order, [low, high], btype='band')
            frame_sequence_norm = signal.lfilter(b, a, frame_sequence_norm,
                                                                        axis=0)

        # length of half series
        len_half = self.frame_num/2 if self.frame_num%2==0 else \
                                                            (self.frame_num+1)/2

        # Tukey window
        alpha = (self.SAMPLE_FREQ / 2) * 2 / (self.frame_num - 1);
        window = np.expand_dims(np.expand_dims( \
                                    signal.tukey(self.frame_num,alpha), 1) , 2 )
        # window = np.expand_dims(np.expand_dims(np.hamming(self.WINDOW_SIZE), 
        #                                                            1) , 2 )
        window = np.tile( window,(1,frame_sequence_norm.shape[1],
                                    frame_sequence_norm.shape[2]) )
        frame_sequence_norm = frame_sequence_norm * window

        # FFT amplitude and take half (Notice: numpy [0:2] = [0,1] )
        psd = np.fft.fft(frame_sequence_norm,axis=0)   # calculate along time series axis = 0
        freq = np.fft.fftfreq(self.frame_num, 
                                d=1/float( self.SAMPLE_FREQ))[0:len_half]

        # PSD
        psd = psd[0:len_half,:,:]
        psd = np.power(np.abs(psd),2) / (self.SAMPLE_FREQ*self.frame_num)
        psd[1:len_half-1,:,:] = np.multiply(psd[1:len_half-1,:,:],2)

        # Get maximum amplitude
        psd_max = psd.max(axis=0)
        psd_sum = np.sum(psd,axis=0)

        # Generate frequency squence, half
        freq_max_psd = np.argmax(psd, axis=0)
        freq_max_psd = np.take(freq,freq_max_psd)

        # Add threshold
        if threshold_on:
            threshold = np.mean(psd,0)+3*np.std(psd,0)
            mask = psd_max>threshold
            psd = psd*mask
        
        # The height between peak and mean
        # mean_psd = np.mean(psd,0)
        # gap = 
        
        # Normalize
        np.divide(psd,psd_sum)
        return [psd, freq, psd_max, freq_max_psd]