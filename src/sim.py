import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
class Sim:

    def __init__(self,window_size,sample_freq,frame_sequence=None):
        """Init Sim."""
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
                                            self.frame_sequence.shape[0], 
                                            frames, axis=0)    
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

    def acf(self,sim_metric='MI'):
        """Calculate similarities on frame sequence and estimate frequency using
             auto-correlation.
            Notice: always take the first frame as reference frame.

        Args:
            sim_metric: a string indicating the similarity metric to use, mutual
                         information as default or absolute correlation.
        """
        frame_ref = self.frame_sequence[0,:,:]

        sim = []
        for i in range(0,self.frame_num):
            frame_current = self.frame_sequence[i,:,:]
            if sim_metric == 'MI':
                sim.append( self.MI(frame_ref,frame_current) )
            else:
                sim.append( self.absolute_correlation(frame_ref,frame_current) )
        
        sim = np.array(sim,dtype=np.float)
        if sim_metric != 'MI':
            sim = 1 - sim / sim.max()
        autocorr = self.autocorrelation(sim)
        autocorr[0:2] = 0
        if np.argmax(autocorr)==0:
            print autocorr
            plt.figure()
            plt.plot(autocorr)
            plt.show(block=True)
        est_freq = float(self.SAMPLE_FREQ) / float(np.argmax(autocorr))

        return [est_freq,autocorr]

    def fft(self,sim_metric='AC',filter_on=True):
        """Calculate similarities on frame sequence and estimate frequency using
             fft.
            Notice: take each frame as reference frame and take average PSD, 
                    Hanning window.

        Args:
            sim_metric: a string indicating the similarity metric to use, mutual
                         information or absolute correlation as default.
            filter_on: a boolean value if use band-pass filter, default keep 
                        only 2-14.5Hz.
        """
        accum_fft = None
        for i in range(self.WINDOW_SIZE):
            frame_ref = self.frame_sequence[i,:,:]
            sim = []
            for j in range(self.WINDOW_SIZE):
                frame_current = self.frame_sequence[j,:,:]
                if sim_metric=='MI':
                    sim.append( self.MI(frame_ref,frame_current) )
                else:
                    sim.append( self.absolute_correlation(frame_ref,
                                                                frame_current)) 
            sim = np.array(sim)
            sim = np.subtract(sim, np.mean(sim))
            len_half = self.frame_num/2 if self.frame_num%2==0 else \
                                                            (self.frame_num+1)/2

            if filter_on:
                order,low,high = 4,2/(0.5 * self.SAMPLE_FREQ),14.5/(0.5 * \
                                                            self.SAMPLE_FREQ)
                b, a = signal.butter(order, [low, high], btype='band')
                sim = signal.lfilter(b, a, sim)

            window_hann = np.hanning(self.WINDOW_SIZE)
            sim = sim * window_hann

            fft_sequence_ampl = np.power( np.abs(np.fft.fft(sim)),2 ) / \
                                        float(self.SAMPLE_FREQ*self.WINDOW_SIZE)
            fft_sequence_ampl = fft_sequence_ampl[0:len_half]
            fft_sequence_ampl[1:len_half-1] = np.multiply(
                                            fft_sequence_ampl[1:len_half-1],2)
            if accum_fft is None:
                accum_fft = fft_sequence_ampl
            else:
                accum_fft += fft_sequence_ampl

        accum_fft = accum_fft / float(self.WINDOW_SIZE)
        freq = np.fft.fftfreq( 
                        self.WINDOW_SIZE, d=1.0/self.SAMPLE_FREQ )[0:len_half]

        est_freq=freq[np.argmax(accum_fft)]

        return [est_freq,accum_fft]

    def entropy(self,array_A,array_B=None):
        """Calculate entropy.

        Args:
            array_A,array_B: one or two 2D numpy arrays.
        """
        grayscale_bin = np.arange(256+1) # 0-256 since histogram count [254-255)
        
        if array_B is None:
            hist, bin_edges = np.histogram(array_A,bins=grayscale_bin,
                                                                density=True)
            
        else:
            assert array_A.shape == array_B.shape
            hist_A, bin_edges_A = np.histogram(array_A,bins=grayscale_bin,
                                                                density=True)
            hist_B, bin_edges_B = np.histogram(array_B,bins=grayscale_bin,
                                                                density=True)
            hist = (hist_A+hist_B)/2
        
        # hist = hist + ~(hist>0); entropy = - np.sum( hist * np.log(hist) )
        entropy = stats.entropy(hist) 
        return entropy

    def MI(self,array_A,array_B):
        """Calculate mutual information between two arrays.

        Args:
            array_A,array_B: one or two 2D numpy arrays.
        """
        
        return self.entropy(array_A)+self.entropy(array_B)-\
                                                self.entropy(array_A,array_B)

    def autocorrelation(self,x):
        """Calculate autocorrelation of array.

        Args:
            x: a 1D numpy vector.
        """
        n = len(x)
        variance = x.var()
        x = x-x.mean()
        r = np.correlate(x, x, mode = 'full')[-n:]
        result = r/(variance) # *(np.arange(n, 0, -1)))
        return np.absolute(result)

    def absolute_correlation(self,array_A, array_B):
        """Calculate mutual information between two arrays.
        
        Args:
            array_A,array_B: two 2D numpy arrays with the same shape.
        """
        assert array_A.shape == array_B.shape
        return np.sum(np.absolute(array_A - array_B))