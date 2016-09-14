import caffe
# from caffe.io import caffe_pb2
import sys
import numpy as np
import scipy.misc
import json
from ESPCNImageProvider import ESPCNImageProvider
import scipy.io

"""Some util functions
"""
"""
Load data from either datum or from original image
Here, I used caffe.io.read_datum_from_image to resolve all format transformation,
instead of the io functions provided by caffe python wrapper
"""


def extract_sample_from_datum(datum, image_mean, resize=-1):
    # extract numpy array from datum, then substract the mean_image
    img_data = decode_datum(datum)
    img_data = subtract_mean(img_data, image_mean)
    if resize == -1:
        return img_data
    else:
        # resize image:
        # first transfer back to h * w * c, -> resize -> c * h * w
        return resize_img_data(img_data, resize)


def resize_img_data(img_data, resize):
    img_data = img_data.transpose(1, 2, 0)
    img_data = scipy.misc.imresize(img_data, (resize, resize))
    return img_data.transpose(2, 0, 1)


def decode_datum(datum):
    if datum.encoded is True:
        datum = caffe.io.decode_datum(datum)
    img_data = caffe.io.datum_to_array(datum)
    return img_data


def extract_sample_from_image(image_fn, image_mean):
    # extract image data from image file directly,
    # return a numpy array, organized in the format of channel * height * width
    datum = caffe.io.read_datum_from_image(image_fn, encoding='')
    img_data = caffe.io.datum_to_array(datum)
    img_data = subtract_mean(img_data, image_mean)
    return img_data


def subtract_mean(img, image_mean):
    """
    Subtract image mean from data sample
    image_mean is a numpy array, either 1 * 3 or of the same size as input image
    """
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img


class BatchCounter:
    """
    Batch Counter
    """
    def __init__(self, batch_size, data_size, random_shuffle=True):
        self._batch_size = batch_size
        self._data_size = data_size
        self._random_shuffle = random_shuffle  # shuffle for training only

        # initialize id list and counter
        self.reset()

    def reset(self):
        self._data_ids = range(self._data_size)
        if self._random_shuffle is True:
            np.random.shuffle(self._data_ids)
        self._curr_epoch = 0
        self._curr_batch = 0
        self._curr_data = 0

    def advance_batch(self):
        next_data = self._curr_data+self._batch_size
        batch = self._data_ids[self._curr_data:next_data]
        if next_data >= self._data_size:  # warp
            next_data = next_data-self._data_size
            if self._random_shuffle is True:
                np.random.shuffle(self._data_ids)
            batch = batch + self._data_ids[0:next_data]
            self._curr_epoch += 1
            self._curr_batch = 0
            if self._random_shuffle is False:
                next_data = 0
        else:
            self._curr_batch += 1
        self._curr_data = next_data
        return (self._curr_epoch, self._curr_batch, batch)


class ESPCNImagePairDataLayer(caffe.Layer):
    """
    ESPCN Image Pair Data Layer
    """
    def setup(self, bottom, top):
#        layer_params = json.loads(self.param_str_.replace("'", '"'))
        layer_params = json.loads(self.param_str.replace("'", '"'))
        print 'ImagePair data layer param:', layer_params
        self._batch_size = int(layer_params['batch_size'])
        self._batch_num = int(layer_params['batch_num'])
        self._test = int(layer_params['test']) == 1

        self.dp = ESPCNImageProvider(layer_params['config_file'])
        self._image_num = self.dp.get_num_images()
        print 'total epoches:', self._batch_num*self._batch_size*1.0/self._image_num
        self.bc = BatchCounter(self._batch_size, self._image_num, random_shuffle=(not self._test))

        self._data_dim = int(np.sqrt(self.dp.get_input_dim()))
        self._label_dim = int(np.sqrt(self.dp.get_output_dim()))

        # (batch_size, channels, height, width)
        top[0].reshape(self._batch_size, 1, self._data_dim, self._data_dim)
        top[1].reshape(self._batch_size, 1, self._label_dim, self._label_dim)
        self._top_data_shape = top[0].data.shape
        self._top_label_shape = top[1].data.shape

        sys.stdout.flush()

    def reshape(self, bottom, top):
        pass

    def get_next_batch(self):
        epoch, batch_num, data_idx = self.bc.advance_batch()
        
        data_batch = np.zeros(self._top_data_shape)
        label_batch = np.zeros(self._top_label_shape)

        for i, idx in enumerate(data_idx):
            dat = self.dp.get_input(idx)
            lbl = self.dp.get_output(idx)
            data_batch[i, ...] = dat.reshape(self._top_data_shape[1:])
            label_batch[i, ...] = lbl.reshape(self._top_label_shape[1:])

        return data_batch, label_batch

    def forward(self, bottom, top):
        data_batch, label_batch = self.get_next_batch()
    
        top[0].data[...] = data_batch.astype(np.float32, copy=False)
        top[1].data[...] = label_batch.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass
