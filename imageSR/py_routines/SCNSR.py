#! /usr/bin/env python

import numpy as np
import cPickle as pickle
import utils
import time


class SRBase(object):
    def __init__(self):
        pass

    def upscale(self, im_l, s):
        """
        % im_l: LR image, float np array in [0, 255]
        % im_h: HR image, float np array in [0, 255]
        """
        im_l = im_l/255.0
        if len(im_l.shape)==3 and im_l.shape[2]==3:
            im_l_ycbcr = utils.rgb2ycbcr(im_l)
        else:
            im_l_ycbcr = np.zeros([im_l.shape[0], im_l.shape[1], 3])
            im_l_ycbcr[:, :, 0] = im_l
            im_l_ycbcr[:, :, 1] = im_l
            im_l_ycbcr[:, :, 2] = im_l

        im_l_y = im_l_ycbcr[:, :, 0]*255 #[16 235]
        im_h_y = self.upscale_alg(im_l_y, s)

        # recover color
        #print 'recover color...'
        if len(im_l.shape)==3:
            im_ycbcr = utils.imresize(im_l_ycbcr, s);
            im_ycbcr[:, :, 0] = im_h_y/255.0; #[16/255 235/255]
            im_h = utils.ycbcr2rgb(im_ycbcr)*255.0
        else:
            im_h = im_h_y

        #print 'clip...'
        im_h = np.clip(im_h, 0, 255)
        im_h_y = np.clip(im_h_y, 0, 255)
        return im_h,im_h_y

    def upscale_alg(self, im_l_y, s):
        pass

class Bicubic(SRBase):
    def upscale_alg(self, im_l_y, s):
        im_h_y = utils.imresize(im_l_y, s)
        return im_h_y


class SCNCaffe(SRBase):
    def __init__(self, params):
        # import only necessary for Caffe
        import caffe

        self.input_blob='data'
        self.output_blob = 'sum2'

#        self.input_blob='Python1'
#        self.output_blob = 'Eltwise7' # for 2 cluster2
#        self.output_blob = 'Eltwise13' # for 4 clusters


        caffe.set_mode_gpu()
        caffe.set_device(params['gpu'])
        self.model = caffe.Net(params['def_file'][0], params['trn_file'][0], caffe.TEST)
        params['batch'] = self.model.blobs[self.input_blob].data.shape[0]
        params['input_size'] = self.model.blobs[self.input_blob].data.shape[2]
        params['output_size'] = params['input_size'] - params['border_size']*2
        if 'matlab_bic' not in params.keys():
            params['matlab_bic'] = 0
        self.params = params
        print params

        #initial dummy running
        self.model.blobs[self.input_blob].data[...] = 1
#        fea = self.model.forward(end=params['layer'], blobs=[params['layer'], 'data'])
        fea = self.model.forward(blobs=[self.output_blob])
        #print fea[params['layer']].shape

    def upscale_alg(self, im_l_y, s):
        h_gt, w_gt = im_l_y.shape[0]*s, im_l_y.shape[1]*s
#        hpsz = self.params['patch_size']/2
        itr_all = int(np.ceil(np.log(s)/np.log(self.params['mdl_scale'])))
        idxdata = np.zeros((2, self.params['batch']), dtype=np.float32)
        indata = np.zeros(self.model.blobs[self.input_blob].data.shape, dtype=np.float32)

        for itr in range(itr_all):
            print 'itr:', itr
            if self.params['matlab_bic']==1:
                im_y = utils.imresize_bic2(im_l_y, self.params['mdl_scale'])
            else:
                im_y = utils.imresize(im_l_y, self.params['mdl_scale'])
            im_y = utils.ExtendBorder(im_y, self.params['border_size'])
#            im_y = utils.RepeatBorder(im_y, self.params['border_size'])
            h, w = im_y.shape

            Height_idx = range(0, h-self.params['input_size'], self.params['output_size'])
            Width_idx  = range(0, w-self.params['input_size'], self.params['output_size'])
            Height_idx += [h-self.params['input_size']]
            Width_idx  += [w-self.params['input_size']]

            bcnt = 0
            im_h_y = np.zeros((h, w), dtype=np.float32)
            t0 = time.time()
            #fea = self.model.forward(data=indata, end=self.params['layer'], blobs=[self.params['layer'], 'data'])
            for i in Height_idx:
                for j in Width_idx:
                    idxdata[0, bcnt] = i
                    idxdata[1, bcnt] = j
                    tmp = im_y[i:i+self.params['input_size'], j:j+self.params['input_size']]
                    indata[bcnt, 0, :, :] = np.reshape(tmp, (1, 1)+tmp.shape)
                    bcnt += 1
                    if bcnt==self.params['batch'] or (i==Height_idx[-1] and j==Width_idx[-1]):
#                        fea = self.model.forward(data=indata, end=self.params['layer'], blobs=[self.params['layer'], 'data'])
                        self.model.blobs[self.input_blob].data[...] = indata
                        fea = self.model.forward(blobs=[self.output_blob])

                        for b in range(bcnt):
                            si = idxdata[0, b]+self.params['border_size']
                            sj = idxdata[1, b]+self.params['border_size']
                            im_h_y[si:si+self.params['output_size'], sj:sj+self.params['output_size']] = \
                                    np.reshape(fea[self.output_blob][b, 0, :, :], (self.params['output_size'], self.params['output_size']))
                        bcnt = 0
            t1 = time.time()
            print 'caffe time: {}'.format(t1-t0)

            im_h_y = im_h_y[self.params['border_size']:-self.params['border_size'], self.params['border_size']:-self.params['border_size']]
            im_l_y = im_h_y

        # shrink size to gt
        if (im_h_y.shape[0]>h_gt):
            print 'downscale from {} to {}'.format(im_h_y.shape, (h_gt, w_gt))
#            if self.params['matlab_bic']==1:
            if 0:
                im_h_y = utils.imresize_bic2(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            else:
                im_h_y = utils.imresize(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            assert(im_h_y.shape[1]==w_gt)

        return im_h_y

class SCNCaffeUpsc(SRBase):
    def __init__(self, params):
        # import only necessary for Caffe
        import caffe

#        self.input_blob='data'
#        self.output_blob = 'sum2'

        self.input_blob='Python1'
#        self.output_blob = 'Eltwise7' # for 2 cluster2
        self.output_blob = 'Eltwise13' # for 4 clusters

        caffe.set_mode_gpu()
        caffe.set_device(params['gpu'])
        self.model = caffe.Net(params['def_file'][0], params['trn_file'][0], caffe.TEST)
        params['batch'] = self.model.blobs[self.input_blob].data.shape[0]
        params['input_size'] = self.model.blobs[self.input_blob].data.shape[2]
        params['output_size'] = params['input_size'] - params['border_size']*2
        if 'matlab_bic' not in params.keys():
            params['matlab_bic'] = 0
        self.params = params
#        print params

        #initial dummy running
        self.model.blobs[self.input_blob].data[...] = 1
        fea = self.model.forward(blobs=[self.output_blob])
        #print fea[params['layer']].shape

    def upscale(self, im_l, s):
        """
        % im_l: LR image, float np array in [0, 1]
        % im_h: HR image, float np array in [0, 1]
        """
        """
#        im_l = im_l/255.0
        if len(im_l.shape)==3 and im_l.shape[2]==3:
            im_l_ycbcr = utils.rgb2ycbcr(im_l)
        else:
            im_l_ycbcr = np.zeros([im_l.shape[0], im_l.shape[1], 3])
            im_l_ycbcr[:, :, 0] = im_l
            im_l_ycbcr[:, :, 1] = im_l
            im_l_ycbcr[:, :, 2] = im_l

        im_l_y = im_l_ycbcr[:, :, 0]*255 #[16 235]
        """
        im_l_y = im_l
        im_h_y = self.upscale_alg(im_l_y, s)
        """
        # recover color
        #print 'recover color...'
        if len(im_l.shape)==3:
            im_ycbcr = utils.imresize(im_l_ycbcr, s);
            im_ycbcr[:, :, 0] = im_h_y/255.0; #[16/255 235/255]
            im_h = utils.ycbcr2rgb(im_ycbcr)*255.0
        else:
            im_h = im_h_y

        #print 'clip...'
        im_h = np.clip(im_h, 0, 255)
        im_h_y = np.clip(im_h_y, 0, 255)
        """
        im_h = None
        return im_h,im_h_y

    def upscale_alg(self, im_l_y, s):
#        h_gt, w_gt = im_l_y.shape[0]*s, im_l_y.shape[1]*s
        if len(im_l_y.shape)==2:
            im_l_y = im_l_y[None,...] 
        h_gt, w_gt = im_l_y.shape[1], im_l_y.shape[2]
        itr_all = int(np.ceil(np.log(s)/np.log(self.params['mdl_scale'])))
        idxdata = np.zeros((2, self.params['batch']), dtype=np.float32)
        indata = np.zeros(self.model.blobs[self.input_blob].data.shape, dtype=np.float32)
#        print 'indata shape: ', indata.shape

        for itr in range(itr_all):
            print 'itr:', itr
            """
            if self.params['matlab_bic']==1:
                im_y = utils.imresize_bic2(im_l_y, self.params['mdl_scale'])
            else:
                im_y = utils.imresize(im_l_y, self.params['mdl_scale'])
            """
            im_y = im_l_y
            im_y_tmp = np.empty([im_y.shape[0], im_y.shape[1]+2*self.params['border_size'], im_y.shape[2]+2*self.params['border_size']])
            for j in range(im_y.shape[0]):
                im_y_tmp[j,...] = utils.ExtendBorder( np.squeeze(im_y[j,...]), self.params['border_size'])
            im_y = im_y_tmp

            # if run comparison experiments
            if 0:
                im_y = im_y[im_y.shape[0]/2,...][None,...]

#            im_y = utils.RepeatBorder(im_y, self.params['border_size'])
            depth, h, w = im_y.shape
#            print 'im_y shape: ', im_y.shape
            Height_idx = range(0, h-self.params['input_size'], self.params['output_size'])
            Width_idx  = range(0, w-self.params['input_size'], self.params['output_size'])
            Height_idx += [h-self.params['input_size']]
            Width_idx  += [w-self.params['input_size']]

            bcnt = 0
            im_h_y = np.zeros((h, w), dtype=np.float32)
            t0 = time.time()
            #fea = self.model.forward(data=indata, end=self.params['layer'], blobs=[self.params['layer'], 'data'])
            for i in Height_idx:
                for j in Width_idx:
                    idxdata[0, bcnt] = i
                    idxdata[1, bcnt] = j
#                    tmp = im_y[i:i+self.params['input_size'], j:j+self.params['input_size']]
                    tmp = im_y[:, i:i+self.params['input_size'], j:j+self.params['input_size']]
#                    indata[bcnt, 0, :, :] = np.reshape(tmp, (1, 1)+tmp.shape)
                    indata[bcnt, :, :, :] = np.reshape(tmp, tmp[None,...].shape)
                    bcnt += 1
                    if bcnt==self.params['batch'] or (i==Height_idx[-1] and j==Width_idx[-1]):
#                        fea = self.model.forward(data=indata, end=self.params['layer'], blobs=[self.output_blob, self.input_blob])
                        self.model.blobs[self.input_blob].data[...] = indata
                        fea = self.model.forward(blobs=[self.output_blob])

                        for b in range(bcnt):
                            si = idxdata[0, b]+self.params['border_size']
                            sj = idxdata[1, b]+self.params['border_size']
                            im_h_y[si:si+self.params['output_size'], sj:sj+self.params['output_size']] = \
                                    np.reshape(fea[self.output_blob][b, 0, :, :], (self.params['output_size'], self.params['output_size']))
#                                    np.reshape(fea[self.params['layer']][b, 0, :, :], (self.params['output_size'], self.params['output_size']))
                        bcnt = 0
            t1 = time.time()
            print 'caffe time: {}'.format(t1-t0)

            im_h_y = im_h_y[self.params['border_size']:-self.params['border_size'], self.params['border_size']:-self.params['border_size']]
            im_l_y = im_h_y

#        print 'im_h_y shape: ', im_h_y.shape
        # shrink size to gt
        if (im_h_y.shape[0]>h_gt):
            print 'downscale from {} to {}'.format(im_h_y.shape, (h_gt, w_gt))
            if self.params['matlab_bic']==1:
                im_h_y = utils.imresize_bic2(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            else:
                im_h_y = utils.imresize(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            assert(im_h_y.shape[1]==w_gt)

        return im_h_y


class SCNConvNet(SRBase):
    def __init__(self, params):
        # import only necessary for ConvNet
        from gpumodel import IGPUModel
        from shownet import ShowConvNet
        import options

        load_dic = IGPUModel.load_checkpoint(params['mdl_file'][0])
        op = load_dic['op']
        op2 = ShowConvNet.get_options_parser()
        op.merge_from(op2)
        op.eval_expr_defaults()
        op.set_value('load_file', params['mdl_file'][0])
        op.set_value('gpu', [params['gpu']], False)
        op.set_value('minibatch_size', params['batch'], False)
        op.set_value('write_features', params['layer'], False)
        #for k in op.options.keys():
        #    print k, op.options[k].value

        self.model = ShowConvNet(op, load_dic)
        layers = [_['name'] for _ in self.model.layers]
        params['layer_idx'] = layers.index(params['layer'])
        params['output_size'] = params['input_size'] - params['border_size']*2
        self.params = params

        #initial dummy running
        indata = np.zeros((self.params['input_size']**2, self.params['batch']), dtype=np.float32)
        outdata = np.zeros((self.params['output_size']**2, self.params['batch']), dtype=np.float32)
        data = [indata, outdata]
        ftrs = np.zeros((self.params['batch'], self.params['output_size']**2), dtype=np.float32)
        self.model.do_write_one_feature(data, ftrs, self.params['layer_idx'])

    def upscale_alg(self, im_l_y, s):
        h_gt, w_gt = im_l_y.shape[0]*s, im_l_y.shape[1]*s
#        hpsz = self.params['patch_size']/2
        itr_all = int(np.ceil(np.log(s)/np.log(self.params['mdl_scale'])))

        indata = np.zeros((self.params['input_size']**2, self.params['batch']), dtype=np.float32)
        outdata = np.zeros((self.params['output_size']**2, self.params['batch']), dtype=np.float32)
        idxdata = np.zeros((2, self.params['batch']), dtype=np.float32)
        ftrs = np.zeros((self.params['batch'], self.params['output_size']**2), dtype=np.float32)

        for itr in range(itr_all):
            print 'itr:', itr
            if self.params['matlab_bic']==1:
                im_y = utils.imresize_bic2(im_l_y, self.params['mdl_scale'])
            else:
                im_y = utils.imresize(im_l_y, self.params['mdl_scale'])
            im_y = utils.ExtendBorder(im_y, self.params['border_size'])
            h, w = im_y.shape

            Height_idx = range(0, h-self.params['input_size'], self.params['output_size'])
            Width_idx  = range(0, w-self.params['input_size'], self.params['output_size'])
            Height_idx += [h-self.params['input_size']]
            Width_idx  += [w-self.params['input_size']]

            bcnt = 0
            im_h_y = np.zeros((h, w), dtype=np.float32)
            t0 = time.time()
            for i in Height_idx:
                for j in Width_idx:
                    idxdata[0, bcnt] = i
                    idxdata[1, bcnt] = j
                    tmp = im_y[i:i+self.params['input_size'], j:j+self.params['input_size']]
                    indata[:, bcnt] = np.reshape(tmp, (indata.shape[0], ))
                    bcnt += 1
                    if bcnt==self.params['batch'] or (i==Height_idx[-1] and j==Width_idx[-1]):
                        self.model.do_write_one_feature([indata, outdata], ftrs, self.params['layer_idx'])
                        for b in range(bcnt):
                            si = idxdata[0, b]+self.params['border_size']
                            sj = idxdata[1, b]+self.params['border_size']
                            im_h_y[si:si+self.params['output_size'], sj:sj+self.params['output_size']] = \
                                    np.reshape(ftrs[b, :], (self.params['output_size'], self.params['output_size']))
                        bcnt = 0
            t1 = time.time()
            print 'convnet time: {}'.format(t1-t0)

            im_h_y = im_h_y[self.params['border_size']:-self.params['border_size'], self.params['border_size']:-self.params['border_size']]
            im_l_y = im_h_y

        # shrink size to gt
        if (im_h_y.shape[0]>h_gt):
            print 'downscale from {} to {}'.format(im_h_y.shape, (h_gt, w_gt))
            if self.params['matlab_bic']==1:
                im_h_y = utils.imresize_bic2(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            else:
                im_h_y = utils.imresize(im_h_y, 1.0*h_gt/im_h_y.shape[0])
            assert(im_h_y.shape[1]==w_gt)

        return im_h_y

