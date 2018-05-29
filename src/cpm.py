import cv2 as cv 
import numpy as np
import scipy
import caffe
import time
from configobj import ConfigObj
import util
import copy
import math
import os

class CPM:
    """Class for convolutional pose machine.

    Attributes:
        _param: a series of integer parameters describing the hardware 
                configuration, check config for details.
        _model: a series of integer and string parameters describing the network
                 model, check config for details.
        _person_net: a caffe CNN to locate persons' position in the image.
        _pose_net: a caffe CNN to locate persons' joints.
    """

    def __init__(self):
        """Init CPM networks."""

        self._param, self._model = self._read_cpm_config()
        if self._param['use_gpu']: 
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        caffe.set_device(self._param['GPUdeviceNumber']) # set to your device!
        self._person_net = caffe.Net(self._model['deployFile_person'], 
                                    self._model['caffemodel_person'], 
                                    caffe.TEST )
        self._pose_net = caffe.Net(self._model['deployFile'], 
                                    self._model['caffemodel'], caffe.TEST)

    def locate_person(self,image):
        """Locate person position in an image.

        Args:
            image: a numpy array decribing the image in BGR.

        Returns:
            x,y: a float numpy vector indicating persons' positions in the 
                image, if not given, program will run locate_person.
        """

        image_padded, pad = util.pad_right_down_corner(image)

        # Shape blob size
        self._person_net.blobs['image'].reshape(*(1, 3, image_padded.shape[0], 
                                                image_padded.shape[1]))
        self._person_net.reshape()
        # Dry run to avoid GPU synchronization later in caffe
        self._person_net.forward(); 

        # Feed image
        self._person_net.blobs['image'].data[...] = np.transpose(np.float32( \
                        image_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;

        # Run and time evalution
        start_time = time.time()
        output_blobs = self._person_net.forward()
        time_person_net = time.time() - start_time
        # print('Person net took %.2f ms. \n' % (1000 * (time_person_net)))

        # Get person confidence map
        person_map = np.squeeze( \
                            self._person_net.blobs[output_blobs.keys()[0]].data)
        person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, 
                                        interpolation=cv.INTER_CUBIC)

        # Locate person position
        data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
        maxima = (person_map_resized == data_max)
        diff = (data_max > 0.5)
        maxima[diff == 0] = 0
        x = np.nonzero(maxima)[1]
        y = np.nonzero(maxima)[0]

        # Add plus tricky
        if x.size>1:
            x_center,y_center=0,0
            dist_min,dist_min_idx = 100000,-1
            for i in range(x.size):
                dist = math.sqrt(math.pow(x[i]-image_padded.shape[1]/2,2)+ \
                                math.pow(y[i]-image_padded.shape[0]/2,2))
                if dist < dist_min:
                    dist_min_idx = i
                    dist_min = dist
            return [ np.expand_dims(np.array(x[dist_min_idx]),0), 
                    np.expand_dims(np.array(y[dist_min_idx]),0) ]

        return [x,y]

    def locate_joints(self, image, x=None, y=None, return_conf_map=False):
        """Locate joints position with CPM.

        Args:
            image: a numpy array decribing the image in BGR.
            x,y: a float numpy array indicating persons' position in the image, 
                if not given, program will run locate_person.
            return_conf_map: a boolean indicating whether return prediction 
                confidence map for 14 joints.
        
        Returns:
            prediction: a numpy float64 array (14*2) showing the joints' 
                        predition (x,y).
            conf_map(optional): a numpy float64 array(h*w*14) showing the 
                            confidence map of the prediction for 14 joints.

        """

        # Resize image
        scale = self._model['boxsize']/(image.shape[0] * 1.0)
        image = cv.resize(image, (0,0), fx=scale, fy=scale, 
                            interpolation=cv.INTER_CUBIC)
        
        # Check if person location is given, if not 
        if x is None or y is None:
            x, y = self.locate_person(image)

        # Crop region of interest for each person, pixel-wise
        num_people = x.size
        person_image = np.ones((self._model['boxsize'], self._model['boxsize'], 
                                3, num_people)) * 128
        for p in range(num_people):
            for ch in range(3):
                person_image[:,:,ch,p] = util.map_matl_to_mats(
                                                person_image[:,:,ch,p],
                                                image[:,:,ch],x[p],y[p])

        # Generate a guassian map around person position 
        # Put at the 4th channel of each cropped region for the input of CPM
        gaussian_map = np.zeros((self._model['boxsize'],self._model['boxsize']))
        for x_p in range(self._model['boxsize']):
            for y_p in range(self._model['boxsize']):
                dist_sq = (x_p - self._model['boxsize']/2) * \
                            (x_p - self._model['boxsize']/2) + \
                            (y_p - self._model['boxsize']/2) *  \
                            (y_p - self._model['boxsize']/2)
                exponent = dist_sq / 2.0 / self._model['sigma'] / \
                                    self._model['sigma']
                gaussian_map[y_p, x_p] = math.exp(-exponent)

        # Dry run to avoid GPU synchronization later in caffe
        self._pose_net.forward()
        # The input is 4-channel image, 4th channel is gaussian distributed 
        # confidence map from person net
        # Run CPM and get confidence map for each person
        output_blobs_array = [dict() for dummy in range(num_people)]
        for p in range(num_people):
            input_4ch = np.ones((self._model['boxsize'], self._model['boxsize'],
                                 4))
            # normalize to [-0.5, 0.5]
            input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 
            input_4ch[:,:,3] = gaussian_map
            self._pose_net.blobs['data'].data[...] = np.transpose(np.float32( \
                                        input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
            start_time = time.time()
            output_blobs_array[p] = copy.deepcopy( \
                                    self._pose_net.forward()['Mconv5_stage6'])
            # print('For person %d, pose net took %.2f ms.' % (p, 1000 * \ 
            #                                       (time.time() - start_time)))

        # To get predictions upscale the confidence maps by 8x, 
        # and pick the strongest peak as prediction.
        prediction = np.zeros((14, 2, num_people))
        conf_map = np.zeros([image.shape[0],image.shape[1],14])
        for p in range(num_people):
            for part in range(14):
                part_map = output_blobs_array[p][0, part, :, :]
                # Confidence map is not in image size but cropped 368x368
                part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, 
                                            interpolation=cv.INTER_CUBIC)
                prediction[part,:,p]=np.unravel_index(part_map_resized.argmax(),
                                                         part_map_resized.shape)

                if return_conf_map:
                    conf_map[:,:,part] = util.map_mats_to_matl(part_map_resized,
                                                conf_map[:,:,part],x[p],y[p])
                    
            # mapped back on full image
            prediction[:,0,p] = prediction[:,0,p] - (self._model['boxsize']/2) \
                                                                        + y[p]
            prediction[:,1,p] = prediction[:,1,p] - (self._model['boxsize']/2) \
                                                                        + x[p]
        
        # Return confidence map in full image scale
        if return_conf_map:
            return [prediction,conf_map]
        else:
            return prediction

    def _read_cpm_config(self):
        """CPM config file reader for file 'config' in the same folder."""

        config = ConfigObj('config')

        param = config['param']
        model_id = param['modelID']
        model = config['models'][model_id]
        model['boxsize'] = int(model['boxsize'])
        model['np'] = int(model['np'])
        num_limb = len(model['limbs'])/2
        model['limbs'] = np.array(model['limbs']).reshape((num_limb, 2))
        model['limbs'] = model['limbs'].astype(np.int)
        model['sigma'] = float(model['sigma'])
        model['joints_string'] = model['part_str']
        param['use_gpu'] = int(param['use_gpu'])
        param['GPUdeviceNumber'] = int(param['GPUdeviceNumber'])

        return param, model