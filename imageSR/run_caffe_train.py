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
    # print 'Here are the arguments in run_caffe_train.py: ', args

    solver_dir = args.save_path
    os.chdir(solver_dir)
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    solver = caffe.SGDSolver(args.solver_param)
    # save
    #solver.net.save('scnet-dic48.caffemodel')

    #train
    solver.solve()

