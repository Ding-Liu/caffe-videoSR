#! /usr/bin/env python

import numpy as np
import cPickle as pickle
import utils
import time
from SCNSR import SRBase


class ESPCNSR (SRBase):
    def __init__(self, params):
        # import only necessary for Caffe
        import caffe

        caffe.set_mode_gpu()
        caffe.set_device(params['gpu'])
        self.model = caffe.Net(params['def_file'], params['trn_file'], caffe.TEST)

        self.input_blob = params['input_blob']
        self.output_blob = params['output_blob']

        params['input_size'] = self.model.blobs[self.input_blob].data.shape[2]
        params['output_size'] = (params['input_size'] - params['border_size']) * params['mdl_scale']

        self.params = params
        # print params

    def upscale_alg(self, im_l_y, s):
        # assert im_l_y.ndim == 2, 'Input image has to be a grayscale image!'

        if im_l_y.ndim == 2:  # input is single frame
            im_l_y = utils.ExtendBorder(im_l_y, self.params['border_size'])
            self.model.blobs[self.input_blob].data[...] = np.reshape(im_l_y, (1, 1)+im_l_y.shape)
        else:  # input is multiple frames
            im_tmp = np.empty([im_l_y.shape[0], im_l_y.shape[1]+2*self.params['border_size'], im_l_y.shape[2]+2*self.params['border_size']])
            for j in range(im_l_y.shape[0]):
                im_tmp[j, ...] = utils.ExtendBorder(np.squeeze(im_l_y[j, ...]), self.params['border_size'])
            im_l_y = im_tmp
            self.model.blobs[self.input_blob].data[...] = np.expand_dims(im_l_y, axis=0)

        t_start = time.time()
        fea = self.model.forward(blobs=[self.output_blob])
        t_end = time.time()
        im_h_y = fea[self.output_blob][0, 0, :, :]
        print 'caffe time: {}'.format(t_end - t_start)

        return im_h_y

