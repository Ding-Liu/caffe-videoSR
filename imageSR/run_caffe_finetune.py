caffe_root = '../'
import sys

sys.path.insert(0, caffe_root + 'python')

import os.path
import argparse
import numpy as np
import cPickle as pickle
import caffe
from caffe.io import caffe_pb2
import py_data_layers
import scipy.io

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="GPU ID", type=int)
    parser.add_argument("--save-path", help="folder to save check points", type=str)
    parser.add_argument("--solver-param", help="solver parameter file", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    basename = os.path.basename(__file__)
    print 'Here are the arguments in {}: {}'.format(basename, args)

    solver_dir = args.save_path
    os.chdir(solver_dir)
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    solver = caffe.SGDSolver(args.solver_param)

    # initialize from single image SR model
    refnet = caffe.Net('../x4_res/ESPCN_test_res.prototxt', '../x4_res/espcn_iter_10000000.caffemodel', caffe.TEST)

    layers_list = ['conv1', 'conv2', 'conv3']
    for l in layers_list:
        if l == 'conv1':
            s = solver.net.params[l][0].data.shape  # [n, c, w, h]
            # print 'model required shape of weights: ', s
            for i in range(s[0]):  # each filter
                solver.net.params[l][0].data[i, ...] = np.zeros(s[1:])
                solver.net.params[l][0].data[i, s[1]/2, ...] = refnet.params[l][0].data[i, ...]
        else:
            solver.net.params[l][0].data[...] = refnet.params[l][0].data

    # save
    #solver.net.save('scnet-dic48.caffemodel')

    #train
    solver.solve()

