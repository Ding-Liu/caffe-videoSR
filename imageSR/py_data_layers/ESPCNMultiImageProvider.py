# Efficient sub-pixel convolutional network (ESPCN) data provider
# obtain samples by online sampling from images

import os
import sys
import numpy as np
import scipy.misc
import scipy.io


class ESPCNMultiImageProvider:
    def __init__(self, paramfile):
        self.param = {'paramfile': paramfile, 'rotate': 1}
        with open(paramfile, 'r') as fid:
            plines = fid.readlines()

        for l in plines:
            l = l.rstrip().split()
            self.param[l[0]] = l[1]
        self.param['inpsize'] = int(self.param['inpsize'])
        self.param['outsize'] = int(self.param['outsize'])
        self.param['scale'] = int(self.param['scale'])
        self.param['mrgsize'] = int(self.param['mrgsize'])
        assert (self.param['inpsize'] > 2 * self.param['mrgsize']), 'Input size should be larger than 2 * margin size!'
        assert((self.param['inpsize'] - 2 * self.param['mrgsize']) * self.param['scale'] == self.param['outsize']), \
            'Error in input size, output size, scale & margin size!'
        self.param['smplside'] = int(self.param['smplPerSide'])
        self.param['smplvideo'] = int(self.param['smplPerVideo'])
        self.param['inpdepth'] = int(self.param['inpdepth'])
        self.param['outdepth'] = int(self.param['outdepth'])
        assert self.param['inpdepth'] >= self.param['outdepth'], 'Input frame No. should be not less than output frame No.'
        assert (self.param['inpdepth'] - self.param['outdepth']) % 2 == 0
        self.param['rotate'] = max(1, min(16, int(self.param['rotate'])))
        MAT_IN_VAR = self.param['MAT_IN_VAR']
        MAT_OUT_VAR = self.param['MAT_OUT_VAR']

        basename = os.path.basename(__file__) 
        print os.path.splitext(basename)[0], " with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        self.input = []
        self.output = []
        datadir = self.param['imgdata']
        datafiles = [f for f in os.listdir(datadir) if f.endswith('.mat')]
        for f in datafiles:
            d = scipy.io.loadmat(datadir + f)
            assert d[MAT_IN_VAR].ndim == 3
            self.input.append(d[MAT_IN_VAR])
            self.output.append(d[MAT_OUT_VAR])

        self.imgnum = len(self.input)
        self.pertnum = self.param['smplvideo'] * (self.param['smplside']**2)*self.param['rotate']
        self.datanum = self.imgnum * self.pertnum
        self.indim = (self.param['inpsize']**2) * self.param['inpdepth']
        self.outdim = (self.param['outsize']**2) * self.param['outdepth']

        if self.param['train'] == '0':
            self.output = [_*0 for _ in self.output]
        print '%d video clips, %d samples found' % (self.imgnum, self.datanum)

    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input_depth(self):
        return self.param['inpdepth']

    def get_output_depth(self):
        return self.param['outdepth']

    def get_input(self, idx):
        img_id = idx / self.pertnum
        pert_id = idx % self.pertnum
        rot_id = pert_id % self.param['rotate']
        off_id = pert_id / self.param['rotate']
        [c, h, w] = self.input[img_id].shape
        [dz, dy, dx] = self.get_offset(c, h, w, off_id)
        res = self.input[img_id][dz:dz+self.param['inpdepth'], dy:dy+self.param['inpsize'], dx:dx+self.param['inpsize']]
        if rot_id / 2 == 0:
            res = res[::int(rot_id%2*2-1), :, :]
        elif rot_id / 2 == 1:
            res = res[::int(rot_id%2*2-1), :, ::-1]
        elif rot_id / 2 == 2:
            res = res[::int(rot_id%2*2-1), ::-1, :]
        elif rot_id / 2 == 3:
            res = np.transpose(res[::int(rot_id%2*2-1), :, :], (0, 2, 1))
        elif rot_id / 2 == 4:
            res = np.transpose(res[::int(rot_id%2*2-1), :, ::-1], (0, 2, 1))
        elif rot_id / 2 == 5:
            res = np.transpose(res[::int(rot_id%2*2-1), ::-1, :], (0, 2, 1))
        elif rot_id / 2 == 6:
            res = res[::int(rot_id%2*2-1), ::-1, ::-1]
        elif rot_id / 2 == 7:
            res = np.transpose(res[::int(rot_id%2*2-1), ::-1, ::-1], (0, 2, 1))
        return res

    def get_output(self, idx):
        img_id = idx / self.pertnum
        pert_id = idx % self.pertnum
        rot_id = pert_id % self.param['rotate']
        off_id = pert_id / self.param['rotate']
        # [h, w]=self.output[img_id].shape
        [c, h, w] = self.input[img_id].shape
        [dz, dy, dx] = self.get_offset(c, h, w, off_id)
        dz += (self.param['inpdepth'] - self.param['outdepth']) / 2
        dy = (dy + self.param['mrgsize']) * self.param['scale']
        dx = (dx + self.param['mrgsize']) * self.param['scale']
        res = self.output[img_id][dz:dz+self.param['outdepth'], dy:dy+self.param['outsize'], dx:dx+self.param['outsize']]
        if rot_id / 2 == 0:
            res = res[::int(rot_id%2*2-1), :, :]
        elif rot_id / 2 == 1:
            res = res[::int(rot_id%2*2-1), :, ::-1]
        elif rot_id / 2 == 2:
            res = res[::int(rot_id%2*2-1), ::-1, :]
        elif rot_id / 2 == 3:
            res = np.transpose(res[::int(rot_id%2*2-1), :, :], (0, 2, 1))
        elif rot_id / 2 == 4:
            res = np.transpose(res[::int(rot_id%2*2-1), :, ::-1], (0, 2, 1))
        elif rot_id / 2 == 5:
            res = np.transpose(res[::int(rot_id%2*2-1), ::-1, :], (0, 2, 1))
        elif rot_id / 2 == 6:
            res = res[::int(rot_id%2*2-1), ::-1, ::-1]
        elif rot_id / 2 == 7:
            res = np.transpose(res[::int(rot_id%2*2-1), ::-1, ::-1], (0, 2, 1))
        # if rot_id == 1:
        #     res = res[:, :, ::-1]
        # elif rot_id == 2:
        #     res = res[:, ::-1, :]
        # elif rot_id == 3:
        #     res = np.transpose(res, (0, 2, 1))
        # elif rot_id == 4:
        #     res = np.transpose(res[:, :, ::-1], (0, 2, 1))
        # elif rot_id == 5:
        #     res = np.transpose(res[:, ::-1, :], (0, 2, 1))
        # elif rot_id == 6:
        #     res = res[:, ::-1, ::-1]
        # elif rot_id == 7:
        #     res = np.transpose(res[:, ::-1, ::-1], (0, 2, 1))
        return res
    
    def get_offset(self, c, h, w, i):
        iz = i / (self.param['smplside']**2)
        iy = i % (self.param['smplside']**2) / self.param['smplside']
        ix = i % (self.param['smplside']**2) % self.param['smplside']
        dz = round(iz*float(c-self.param['inpdepth'])/(self.param['smplvideo']-1))
        dy = round(iy*float(h-self.param['inpsize'])/(self.param['smplside']-1))
        dx = round(ix*float(w-self.param['inpsize'])/(self.param['smplside']-1))
        return dz, dy, dx


def test(param):
    ts = ESPCNMultiImageProvider(param)
    print "{} images in total".format(ts.get_num_images())
    # for i in range(728,748,1):
    #     im=ts.get_input(i)
    #     y=ts.get_output(i)
    #     print "i={}, input={},\toutput={}".format(i, im.shape, y.shape)
    #     scipy.misc.imsave('./img/{}_in.png'.format(i), im);
    #     scipy.misc.imsave('./img/{}_out.png'.format(i), y);
    # print 'image shape:', np.shape(im)

if __name__ == '__main__':
    basename = os.path.basename(__file__)
    print 'testing {}!'.format(basename)
    assert(len(sys.argv) == 2)
    test(sys.argv[1])

