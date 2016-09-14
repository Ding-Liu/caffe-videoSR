# Super-resolution test for multi-frame inputs
caffe_root = '../'
import sys

sys.path.insert(0, caffe_root + 'python')

import os
import math
import numpy as np
import scipy.io as sio
from PIL import Image
sys.path.append('./py_routines')
# from ImageSR import SCNCaffe, SCNCaffeUpsc
from ESPCNSR import ESPCNSR
import utils
import scipy.io

if __name__ == '__main__':
    IMAGES = '/ws/ifp-06_1/dingliu2/data/SR/TestImage_x4_vsr6_ESPCN.mat'
    SCALE = 4.0
    SHAVE = 0  # set 1 to be consistent with SRCNN
    SAVE_FLAG = 0  # set 1 to save SR results in .mat file (SHAVE needs to be 0)

    MDL_PARAMS = {
                  'def_file': './config/ESPCN_res_5F_deploy.prototxt',
                  'trn_file': './mdl/5F/espcn_iter_10000000.caffemodel',

                  # 'def_file': './config/ESPCN_res_deploy.prototxt',
                  # 'trn_file': './mdl/1F/espcn_iter_2300000.caffemodel',

                  # 'def_file': './config/ESPCN_tanh_deploy.prototxt',
                  # 'trn_file': './mdl/tanh-f2/espcn_iter_3000000.caffemodel',

                  'mdl_scale': SCALE,
                  'border_size': 4,
                  'input_blob': 'data',
                  'output_blob': 'sum',  # 'conv3r' or 'sum'
                  'gpu': 0
    }

    d = sio.loadmat(IMAGES)
    im_gt_y = d['Output'][0]
    im_l_y = d['Input'][0]

    n = len(im_l_y)
    res = {'rmse': np.zeros((n,)), 'psnr': np.zeros((n,))}

    im_h_y = []

    print 'testing model', MDL_PARAMS['trn_file']
    for i in range(n):
        print 'upscaling from', im_l_y[i].shape, 'to', im_gt_y[i].shape

        with open(MDL_PARAMS['def_file'], 'r') as f:
            data = f.readlines()

        # write image channel in deploy.prototxt
        if im_l_y[i].ndim == 3:
            data[3] = 'input_dim: {}\n'.format(im_l_y[i].shape[0])
        else:
            data[3] = 'input_dim: 1\n'
        # write image height in deploy.prototxt
        data[4] = 'input_dim: {}\n'.format(im_l_y[i].shape[-2] + 2 * MDL_PARAMS['border_size'])
        # write image width in deploy.prototxt
        data[5] = 'input_dim: {}\n'.format(im_l_y[i].shape[-1] + 2 * MDL_PARAMS['border_size'])
        # write upscaling factor in deploy.prototxt
        # data[82] = '    num_output: {}\n'.format(int(SCALE ** 2))
        # data[111] = '    kernel_size: {}\n'.format(int(2 * SCALE - SCALE % 2))
        # data[112] = '    stride: {}\n'.format(int(SCALE))
        # data[115] = '    pad: {}\n'.format(int(SCALE // 2))
        # data[129] = '            offset: {}\n'.format(int(SCALE * MDL_PARAMS['border_size']))

        with open(MDL_PARAMS['def_file'], 'w') as f:
            f.writelines(data)

        # initialize ESPCN model
        espcn = ESPCNSR(MDL_PARAMS)

        im_h_y.append(espcn.upscale_alg(im_l_y[i]))

        if SHAVE == 1:
            im_gt_y[i] = utils.shave(im_gt_y[i], int(SCALE))
            im_h_y[-1] = utils.shave(im_h_y[-1], int(SCALE))

        # data range 0~255
        im_h_y_uint8 = np.rint( np.clip(im_h_y[-1], 0, 255))
        im_gt_y_uint8 = np.rint( np.clip(im_gt_y[i], 0, 255))

        # data range 0~1
        # im_h_y_uint8 = np.rint( np.clip(im_h_y * 255, 0, 255))
        # im_gt_y_uint8 = np.rint( np.clip(im_gt_y[i] * 255, 0, 255))

        diff = np.abs(im_h_y_uint8 - im_gt_y_uint8).flatten()
        res['rmse'][i] = np.sqrt(np.mean(np.square(diff)))
        res['psnr'][i] = 20*np.log10(255.0/res['rmse'][i])
        print 'rmse={}, psnr={}'.format(res['rmse'][i], res['psnr'][i])

    print 'mean psnr: {}'.format(np.mean(res['psnr'])) 

    if SAVE_FLAG == 1:
        save_dir = './results/' + IMAGES.split('/')[-1][:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dic = {}
        dic['im_h_y'] = im_h_y
        scipy.io.savemat(os.path.join(save_dir, 'SR_results.mat'), dic)
