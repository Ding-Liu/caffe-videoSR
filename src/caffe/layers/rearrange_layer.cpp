#include <vector>
#include <math.h>

#include "caffe/layers/rearrange_layer.hpp"
//#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RearrangeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom.size(), 1) << "Local Layer takes a single blob as input.";
	CHECK_EQ(top.size(), 1) << "Local Layer takes a single blob as output.";

	depth_ = this->layer_param_.rearrange_param().depth();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	CHECK_EQ(channels_ % depth_, 0) << "input channels need to be a multiple of output channels.";
	// multiple = channels_ \ depth_;
	// multiple_sqrt = sqrt(multiple);
	scaled_sqr_ = Dtype(channels_) / Dtype(depth_);
	// CHECK_EQ(multiple_sqrt, int(multiple_sqrt)) << "the multiple needs to be a perfect square.";
	CHECK_EQ(sqrt(scaled_sqr_), int(sqrt(scaled_sqr_))) << "the upscaling scale has to be an integer.";
	scale_ = int(sqrt(scaled_sqr_));
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {

	scaled_height_ = scale_ * bottom[0]->height();
	scaled_width_ = scale_ * bottom[0]->width();

	top[0]->Reshape(bottom[0]->num(), depth_, scaled_height_, scaled_width_);
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	for (int n = 0; n < num_; ++n) {
		for (int d = 0; d < depth_; ++d) {
			for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
					for (int i = 0; i < scale_; ++i) {
						for (int j = 0; j < scale_; ++j) {
							int height_idx = h * scale_ + i;
							int width_idx = w * scale_ + j;
							*(top_data + top[0]->offset(n, d, height_idx, width_idx)) =
							    *(bottom_data + bottom[0]->offset(n, (d * scale_ + i) * scale_ + j, h, w));
						}
					}
				}
			}
		}
	}
}

template <typename Dtype>
void RearrangeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const Dtype* top_diff = top[0]->cpu_diff();
//   	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	for (int n = 0; n < num_; ++n) {
		for (int d = 0; d < depth_; ++d) {
			for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
					for (int i = 0; i < scale_; ++i) {
						for (int j = 0; j < scale_; ++j) {
							int height_idx = h * scale_ + i;
							int width_idx = w * scale_ + j;
							*(bottom_diff + bottom[0]->offset(n, (d * scale_ + i) * scale_ + j, h, w)) =
							    *(top_diff + top[0]->offset(n, d, height_idx, width_idx));
						}
					}
				}
			}
		}
	}
}

INSTANTIATE_CLASS(RearrangeLayer);
REGISTER_LAYER_CLASS(Rearrange);
}  // namespace caffe
