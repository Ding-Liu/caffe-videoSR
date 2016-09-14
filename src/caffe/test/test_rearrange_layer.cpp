#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rearrange_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RearrangeLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  protected:
  RearrangeLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_->Reshape(2, 18, 4, 3);
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~RearrangeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForwardDepth1() {
  	LayerParameter layer_param;
	RearrangeParameter* rearrange_param = layer_param.mutable_rearrange_param();
	rearrange_param->set_depth(1);  	

	blob_bottom_->Reshape(2, 4, 3, 2);

	for (int i = 0; i < 2; i += 4*3*2) {
		blob_bottom_->mutable_cpu_data()[i + 0] = 0;
		blob_bottom_->mutable_cpu_data()[i + 1] = 10;
		blob_bottom_->mutable_cpu_data()[i + 2] = 20;
		blob_bottom_->mutable_cpu_data()[i + 3] = 30;
		blob_bottom_->mutable_cpu_data()[i + 4] = 40;
		blob_bottom_->mutable_cpu_data()[i + 5] = 50;
		blob_bottom_->mutable_cpu_data()[i + 6] = 60;
		blob_bottom_->mutable_cpu_data()[i + 7] = 70;
		blob_bottom_->mutable_cpu_data()[i + 8] = 80;
		blob_bottom_->mutable_cpu_data()[i + 9] = 90;
		blob_bottom_->mutable_cpu_data()[i + 10] = 0;
		blob_bottom_->mutable_cpu_data()[i + 11] = 1;
		blob_bottom_->mutable_cpu_data()[i + 12] = 2;
		blob_bottom_->mutable_cpu_data()[i + 13] = 3;
		blob_bottom_->mutable_cpu_data()[i + 14] = 4;
		blob_bottom_->mutable_cpu_data()[i + 15] = 5;
		blob_bottom_->mutable_cpu_data()[i + 16] = 6;
		blob_bottom_->mutable_cpu_data()[i + 17] = 7;
		blob_bottom_->mutable_cpu_data()[i + 18] = 8;
		blob_bottom_->mutable_cpu_data()[i + 19] = 9;
		blob_bottom_->mutable_cpu_data()[i + 20] = 10;
		blob_bottom_->mutable_cpu_data()[i + 21] = 11;
		blob_bottom_->mutable_cpu_data()[i + 22] = 12;
		blob_bottom_->mutable_cpu_data()[i + 23] = 13;
	}
    RearrangeLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), 2);
    EXPECT_EQ(blob_top_->channels(), 1);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 4);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    for (int i = 0; i < 2; i += 4*3*2) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 60);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 70);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 20);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 80);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 30);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 90);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 50);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 13);
    }
  }
  void TestForwardDepth2() {
  	LayerParameter layer_param;
	RearrangeParameter* rearrange_param = layer_param.mutable_rearrange_param();
	rearrange_param->set_depth(2);  	

	blob_bottom_->Reshape(2, 8, 3, 2);

	for (int i = 0; i < 2*2; i += 4*3*2) {
		blob_bottom_->mutable_cpu_data()[i + 0] = 0;
		blob_bottom_->mutable_cpu_data()[i + 1] = 10;
		blob_bottom_->mutable_cpu_data()[i + 2] = 20;
		blob_bottom_->mutable_cpu_data()[i + 3] = 30;
		blob_bottom_->mutable_cpu_data()[i + 4] = 40;
		blob_bottom_->mutable_cpu_data()[i + 5] = 50;
		blob_bottom_->mutable_cpu_data()[i + 6] = 60;
		blob_bottom_->mutable_cpu_data()[i + 7] = 70;
		blob_bottom_->mutable_cpu_data()[i + 8] = 80;
		blob_bottom_->mutable_cpu_data()[i + 9] = 90;
		blob_bottom_->mutable_cpu_data()[i + 10] = 0;
		blob_bottom_->mutable_cpu_data()[i + 11] = 1;
		blob_bottom_->mutable_cpu_data()[i + 12] = 2;
		blob_bottom_->mutable_cpu_data()[i + 13] = 3;
		blob_bottom_->mutable_cpu_data()[i + 14] = 4;
		blob_bottom_->mutable_cpu_data()[i + 15] = 5;
		blob_bottom_->mutable_cpu_data()[i + 16] = 6;
		blob_bottom_->mutable_cpu_data()[i + 17] = 7;
		blob_bottom_->mutable_cpu_data()[i + 18] = 8;
		blob_bottom_->mutable_cpu_data()[i + 19] = 9;
		blob_bottom_->mutable_cpu_data()[i + 20] = 10;
		blob_bottom_->mutable_cpu_data()[i + 21] = 11;
		blob_bottom_->mutable_cpu_data()[i + 22] = 12;
		blob_bottom_->mutable_cpu_data()[i + 23] = 13;
	}
    RearrangeLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), 2);
    EXPECT_EQ(blob_top_->channels(), 2);
    EXPECT_EQ(blob_top_->height(), 6);
    EXPECT_EQ(blob_top_->width(), 4);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    for (int i = 0; i < 2*2; i += 4*3*2) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 60);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 70);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 20);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 80);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 30);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 90);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 0);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 50);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 13);
    }


  }

};

TYPED_TEST_CASE(RearrangeLayerTest, TestDtypesAndDevices);

TYPED_TEST(RearrangeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RearrangeParameter* rearrange_param = layer_param.mutable_rearrange_param();
  rearrange_param->set_depth(2);
  // rearrange_param->set_stride(2);
  RearrangeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
//    EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 12);
  EXPECT_EQ(this->blob_top_->width(), 9);
}

TYPED_TEST(RearrangeLayerTest, TestForward) {
  this->TestForwardDepth1();
  this->TestForwardDepth2();
  // this->TestForwardRectHigh();
//  this->TestForwardRectWide();
}

TYPED_TEST(RearrangeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RearrangeParameter* rearrange_param = layer_param.mutable_rearrange_param();
  rearrange_param->set_depth(2);

  RearrangeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer,
      this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}