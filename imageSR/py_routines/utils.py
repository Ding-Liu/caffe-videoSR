#! /usr/bin/env python

import numpy as np
import cv2
import scipy.io

def modcrop(im, modulo):
    sz = im.shape
    h = int(sz[0]/modulo)*modulo
    w = int(sz[1]/modulo)*modulo

    ims = im[0:h, 0:w, ...]
    return ims

def imresize(im_l, s):
    if s<1:
        im_l = cv2.GaussianBlur(im_l, (7,7), 0.5)
    im_h = cv2.resize(im_l, (0,0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return im_h

def cubic(x):
    """
    % See Keys, "Cubic Convolution Interpolation for Digital Image
    % Processing," IEEE Transactions on Acoustics, Speech, and Signal
    % Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
    """
    absx = np.abs(x)
    absx2 = absx*absx
    absx3 = absx*absx2

    f = (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1<absx)*(absx<=2))
    return f

def correlate(A, f):
    [h, w] = A.shape
    [hf, wf] = f.shape
    B = np.zeros((h-hf+1, w-wf+1))
    [hr, wr] = B.shape

    for i in range(hf):
        for j in range(wf):
            dh = i
            dw = j
            B = B+f[i,j]*A[dh:hr+dh, dw:wr+dw]

    return B
      

def imresize_bic2(im_l, s):
    assert(s==2)
    [h, w] = im_l.shape
    B = np.zeros((h*2, w*2))

    # make bicubic filters
    h1 = cubic(-1.25+np.array(range(4)));
    h2 = cubic(-1.75+np.array(range(4)));
    h1 = h1.reshape((4,1))/np.sum(h1);
    h2 = h2.reshape((4,1))/np.sum(h2);

    filters = []
    filters += [np.dot(h1,h1.T)]
    filters += [np.dot(h2,h1.T)]
    filters += [np.dot(h1,h2.T)]
    filters += [np.dot(h2,h2.T)]

    """
    filters_mat = np.zeros( (4, 16));
    for i in range( len(filters)):
        filters_mat[i,:] = filters[i].flatten()
    d1 = {'fltr_bic': filters_mat}
    scipy.io.savemat('./data/fltrs_bic.mat', d1) 
    """

    imf = RepeatBorder(im_l, 2)
    tmp = correlate(imf, filters[0])
    B[1::2, 1::2] = tmp[1:, 1:]
    tmp = correlate(imf, filters[1])
    B[0::2, 1::2] = tmp[0:-1, 1:]
    tmp = correlate(imf, filters[2])
    B[1::2, 0::2] = tmp[1:, 0:-1]
    tmp = correlate(imf, filters[3])
    B[0::2, 0::2] = tmp[0:-1, 0:-1]

    return B

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*255.0-16)/(235-16) #to [0, 1]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*255.0-16)/(240-16) #to [0, 1]
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def shave(im, border):
    if isinstance(border, int):
        border=[border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im

def RepeatBorder(im, offset):
    sz = im.shape
    assert(len(sz)==2)

    im2 = np.zeros([sz[0]+offset*2, sz[1]+offset*2])
    im2[ offset:-offset, offset:-offset ] = im
    im2[ 0:offset, offset:-offset ] = im[0, :]
    im2[ -offset:, offset:-offset ] = im[-1, :]
    im2[ :, 0:offset] = im2[:, offset:offset+1]
    im2[ :, -offset:] = im2[:, (-offset-1):(-offset)]
    return im2

def ExtendBorder(im, offset):
    sz = im.shape
    assert(len(sz)==2)

    im2 = np.zeros([sz[0]+offset*2, sz[1]+offset*2])
    im2[ offset:-offset, offset:-offset ] = im
    im2[ offset:-offset, 0:offset ] = im[:, offset:0:-1]
    im2[ offset:-offset, -offset: ] = im[:, -2:-(offset+2):-1]
    im2[ 0:offset, :] = im2[2*offset:offset:-1, :]
    im2[ -offset:, :] = im2[-(offset+2):-(2*offset+2):-1, :]

    return im2

def ExtrConvFea(im, fltrs):
    """
    % extract convoluation features from whole image output
    % fea: [mxnxf], where f is the number of features used
    """
    m,n = im.shape
    nf = fltrs.shape[1]
    fs = int(np.round(np.sqrt(fltrs.shape[0])))
    hfs = fs/2
    fea = np.zeros([m-fs+1, n-fs+1, nf])
    for i in range(nf):
        fltr = fltrs[:, i].reshape([fs, fs])
        acts = cv2.filter2D(im, -1, fltr)
        fea[:, :, i] = acts[hfs:-hfs, hfs:-hfs]
    return fea

def ShLU(a, th):
    return np.sign(a)*np.maximum(0, np.abs(a)-th)

