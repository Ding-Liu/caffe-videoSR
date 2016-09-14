#ifndef CAFFE_REARRANGE_LAYER_HPP_
#define CAFFE_ERARRANGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

//#include "caffe/layers/conv_layer.hpp"

namespace caffe {

/**
 * @brief Rearrange layer
 */

template <typename Dtype>
class RearrangeLayer : public Layer<Dtype> {
public:
    explicit RearrangeLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Rearrange"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
       virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
       virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    // data members
    // convolution parameter
//        LayerParameter bicubic_conv_param;

    // channels of output blob
    int depth_;
    int num_;
    int channels_;
    int height_;
    int width_;
    // upscaling scale
    int scale_;
    int scaled_height_;
    int scaled_width_;

    // intermediate variables
    // float multiple_;
    // float multiple_sqrt_;
    Dtype scaled_sqr_;
};  // class RearrangeLayer

}  // namespace caffe

#endif  // CAFFE_REARRANGE_LAYER_HPP_
