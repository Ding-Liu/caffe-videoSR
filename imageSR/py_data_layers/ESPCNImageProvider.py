# Efficient sub-pixel convolutional network (ESPCN) data provider
# obtain samples by online sampling from images

import os
import sys
import numpy as np
import scipy.misc
import scipy.io


class ESPCNImageProvider:
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
        self.param['rotate'] = max(1, min(8, int(self.param['rotate'])))
        MAT_IN_VAR = self.param['MAT_IN_VAR']
        MAT_OUT_VAR = self.param['MAT_OUT_VAR']

        basename = os.path.basename(__file__) 
        print os.path.splitext(basename)[0], " with paramfile:", paramfile, " (", MAT_IN_VAR, MAT_OUT_VAR, ")"

        d = scipy.io.loadmat(self.param['imgdata'])
        self.X = d[MAT_IN_VAR]
        self.Z = d[MAT_OUT_VAR]
        assert(len(np.shape(self.X)) == 2)
        self.imgnum = self.X.shape[1]
        self.pertnum = (self.param['smplside']**2)*self.param['rotate']
        self.datanum = self.imgnum*self.pertnum

        self.indim = self.param['inpsize']**2 #np.prod(np.shape(self.X[0,0]))
        self.outdim = self.param['outsize']**2 #np.prod(np.shape(self.Z[0,0]))
        self.input = [self.X[0, i] for i in range(self.imgnum)]
        self.output = [self.Z[0, i] for i in range(self.imgnum)]
        if self.param['train'] == '0':
            self.output = [_*0 for _ in self.output]
        print '%d whole images, %d samples found' % (self.imgnum, self.datanum)

    def get_num_images(self):
        return self.datanum
    
    def get_input_dim(self):
        return self.indim

    def get_output_dim(self):
        return self.outdim

    def get_input(self, idx):
        img_id = idx / self.pertnum
        pert_id = idx % self.pertnum
        rot_id = pert_id % self.param['rotate']
        off_id = pert_id / self.param['rotate']
        [h, w] = self.input[img_id].shape
        [dy, dx] = self.get_offset(h, w, off_id)
        res = self.input[img_id][dy:dy+self.param['inpsize'], dx:dx+self.param['inpsize']]
        if rot_id == 1:
            res = np.fliplr(res)
        elif rot_id == 2:
            res = np.flipud(res)
        elif rot_id == 3:
            res = res.T
        elif rot_id == 4:
            res = np.fliplr(res).T
        elif rot_id == 5:
            res = np.flipud(res).T
        elif rot_id == 6:
            res = np.rot90(res, 2)
        elif rot_id == 7:
            res = np.rot90(res, 2).T
        return res

    def get_output(self, idx):
        img_id = idx / self.pertnum
        pert_id = idx % self.pertnum
        rot_id = pert_id % self.param['rotate']
        off_id = pert_id / self.param['rotate']
        # [h, w]=self.output[img_id].shape
        [h, w] = self.input[img_id].shape
        [dy, dx] = self.get_offset(h, w, off_id)
        # dy+=self.param['mrgsize']
        # dx+=self.param['mrgsize']
        dy = (dy + self.param['mrgsize']) * self.param['scale']
        dx = (dx + self.param['mrgsize']) * self.param['scale']
        res = self.output[img_id][dy:dy+self.param['outsize'], dx:dx+self.param['outsize']]
        if rot_id == 1:
            res = np.fliplr(res)
        elif rot_id == 2:
            res = np.flipud(res)
        elif rot_id == 3:
            res = res.T
        elif rot_id == 4:
            res = np.fliplr(res).T
        elif rot_id == 5:
            res = np.flipud(res).T
        elif rot_id == 6:
            res = np.rot90(res, 2)
        elif rot_id == 7:
            res = np.rot90(res, 2).T
        return res
    
    def get_offset(self, h, w, i):
        iy = i / self.param['smplside']
        ix = i % self.param['smplside']
#        dy=np.floor(iy*float(h-self.param['inpsize'])/(self.param['smplside']-1))
#        dx=np.floor(ix*float(w-self.param['inpsize'])/(self.param['smplside']-1))
        dy = round(iy*float(h-self.param['inpsize'])/(self.param['smplside']-1))
        dx = round(ix*float(w-self.param['inpsize'])/(self.param['smplside']-1))
        return dy, dx


def test(param):
    ts = ESPCNImageProvider(param)
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

