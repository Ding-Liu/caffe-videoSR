#include <vector>

#include "caffe/layers/rearrange_layer.hpp"
//#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RearrangeForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels, const int depth,
    const int height, const int width, const int scale,
    const int scaled_height, const int scaled_width,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const int j = c % scale;
    const int i = (c / scale) % scale;    
    const int d = c / scale / scale;

    int height_idx = h * scale + i;
    int width_idx = w * scale + j;
    int top_idx = ((n*depth+d)*scaled_height+height_idx)*scaled_width+width_idx;

    top_data[top_idx] = bottom_data[index];    
  }
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  int count = top[0]->count();

  RearrangeForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom_data, num_, channels_, depth_, height_, width_, scale_, scaled_height_, scaled_width_, top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void RearrangeBackward(const int nthreads, const Dtype* const top_diff,
    const int num,
    const int channels, const int depth, const int height, const int width, const int scale,
    const int scaled_height, const int scaled_width,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    const int j = c % scale;
    const int i = (c / scale) % scale;
    const int d = c / scale / scale;

    int height_idx = h * scale + i;
    int width_idx = w * scale + j;
    int top_idx = ((n*depth+d)*scaled_height+height_idx)*scaled_width+width_idx;

    bottom_diff[index] = top_diff[top_idx];
  }
}


template <typename Dtype>
void RearrangeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  RearrangeBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, top_diff, num_, channels_, depth_, height_, width_, scale_, scaled_height_, scaled_width_, bottom_diff);

  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(RearrangeLayer);


}  // namespace caffe
