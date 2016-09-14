import caffe
import sys
import numpy as np
import json
from ESPCNMultiImageProvider import ESPCNMultiImageProvider
from ImagePairLayer import BatchCounter


class ESPCNMultiImagePairDataLayer(caffe.Layer):
    """
    ESPCN Multi-Image Pair Data Layer
    """
    def setup(self, bottom, top):
        layer_params = json.loads(self.param_str.replace("'", '"'))
        print 'ImagePair data layer param:', layer_params
        self._batch_size = int(layer_params['batch_size'])
        self._batch_num = int(layer_params['batch_num'])
        self._test = int(layer_params['test']) == 1

        self.dp = ESPCNMultiImageProvider(layer_params['config_file'])
        self._image_num = self.dp.get_num_images()
        print 'total epoches:', self._batch_num*self._batch_size*1.0/self._image_num
        self.bc = BatchCounter(self._batch_size, self._image_num, random_shuffle=(not self._test))

        self._data_dim = int(np.sqrt(self.dp.get_input_dim()/float(self.dp.get_input_depth())))
        self._label_dim = int(np.sqrt(self.dp.get_output_dim()/float(self.dp.get_output_depth())))
        self._data_depth = self.dp.get_input_depth()
        self._label_depth = self.dp.get_output_depth()

        # (batch_size, channels, height, width)
        top[0].reshape(self._batch_size, self._data_depth, self._data_dim, self._data_dim)
        top[1].reshape(self._batch_size, self._label_depth, self._label_dim, self._label_dim)
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

            data_batch[i, ...] = dat
            label_batch[i, ...] = lbl

        return data_batch, label_batch

    def forward(self, bottom, top):
        data_batch, label_batch = self.get_next_batch()
    
        top[0].data[...] = data_batch.astype(np.float32, copy=False)
        top[1].data[...] = label_batch.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass
