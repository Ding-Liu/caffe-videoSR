#!/bin/bash
# deepsr training on caffe
# arguments from condor job: $(Process) 

jobid=$1

gpu_id=1

codedir=/home/dingliu2/Documents/caffe/videoSR
datdir=/ws/ifp-06_1/dingliu2/data
savepath=${codedir}/mdl/${jobid}
caffe=${codedir}/run_caffe_train.py
#caffe=${codedir}/run_caffe_finetune.py
#caffe=${codedir}/run_caffe_loadmdl.py

# data provider
data_module=py_data_layers.ImagePairLayer
data_layer=ESPCNImagePairDataLayer

# network specifications
solver_file=${codedir}/config/ESPCN_solver.prototxt

# Relu activation
#train_layer_file=${codedir}/config/ESPCN_train_res.prototxt
#test_layer_file=${codedir}/config/ESPCN_test_res.prototxt
# Parametric relu activation
#train_layer_file=${codedir}/config/ESPCN_prelu_train.prototxt
#test_layer_file=${codedir}/config/ESPCN_prelu_test.prototxt
# Tanh activation
train_layer_file=${codedir}/config/ESPCN_tanh_train.prototxt
test_layer_file=${codedir}/config/ESPCN_tanh_test.prototxt

batch_sz=64
train_batches=10000000 #over all epoches
test_batches=8 # number of batches used during testing
test_int=10000
snap_int=100000

inpSize=20 # input size
scale=2 # upscaling factor
mrgSize=4 # margin size
outSize=$((($inpSize-2*$mrgSize)*$scale)) # output size
smplPerSide=50 #50 for natural image, 200 for scan document
rotPerImg=8 #4 for natural image, 1 for scan document

# data range 0~255
#imgdata_tr=${datdir}/SR/TrainingImage_x3_91+BSD200s5_ESPCN.mat
#imgdata_ts=${datdir}/SR/TestImage_x3_set5s1_ESPCN.mat
# data range 0~1 (for tanh activation)
imgdata_tr=${datdir}/SR/TrainingImage_x2_91s5_ESPCN-1.mat
imgdata_ts=${datdir}/SR/TestImage_x2_set5s1_ESPCN-1.mat

# prepare dp configurations
mkdir -p $savepath
rm -r $savepath/*
train_cfg=${savepath}/param_tr.cfg
test_cfg=${savepath}/param_ts.cfg
#rm -r ${train_cfg}
echo "imgdata $imgdata_tr" >> $train_cfg
echo "train 1" >> $train_cfg
echo "inpsize $inpSize" >> $train_cfg
echo "outsize $outSize" >> $train_cfg
echo "scale $scale" >> $train_cfg
echo "mrgsize $mrgSize" >> $train_cfg
echo "smplPerSide $smplPerSide" >> $train_cfg
echo "rotate $rotPerImg" >> $train_cfg
echo 'MAT_IN_VAR Input' >> $train_cfg
echo 'MAT_OUT_VAR Output' >> $train_cfg
#rm -r ${test_cfg}
echo "imgdata $imgdata_ts" >> $test_cfg
echo "train 1" >> $test_cfg
echo "inpsize $inpSize" >> $test_cfg
echo "outsize $outSize" >> $test_cfg
echo "scale $scale" >> $test_cfg
echo "mrgsize $mrgSize" >> $test_cfg
echo "smplPerSide 10" >> $test_cfg
echo 'MAT_IN_VAR Input' >> $test_cfg
echo 'MAT_OUT_VAR Output' >> $test_cfg

# prepare network/solver
#layer_target=$savepath/$(basename $layer_file)
train_layer_target=$savepath/$(basename $train_layer_file)
test_layer_target=$savepath/$(basename $test_layer_file)
solver_target=$savepath/$(basename $solver_file)

#cp $layer_file $layer_target
cp $train_layer_file $train_layer_target
cp $test_layer_file $test_layer_target
cp $solver_file $solver_target

#sed -i 's:<NET_HOLDER>:./'$(basename $layer_target)':g' $solver_target
sed -i 's:<TRAIN_NET_HOLDER>:./'$(basename $train_layer_target)':g' $solver_target
sed -i 's:<TEST_NET_HOLDER>:./'$(basename $test_layer_target)':g' $solver_target
sed -i 's/<MAX_ITR_HOLDER>/'${train_batches}'/g' $solver_target
sed -i 's/<TEST_ITR_HOLDER>/'${test_batches}'/g' $solver_target
sed -i 's/<TEST_INT_HOLDER>/'${test_int}'/g' $solver_target
sed -i 's/<SNAP_INT_HOLDER>/'${snap_int}'/g' $solver_target

sed -i 's/<PYTHON_MODULE_HOLDER>/'${data_module}'/g' $train_layer_target
sed -i 's/<PYTHON_LAYER_HOLDER>/'${data_layer}'/g' $train_layer_target
sed -i 's/<BATCH_SIZE_HOLDER>/'${batch_sz}'/g' $train_layer_target
sed -i 's/<PYTHON_MODULE_HOLDER>/'${data_module}'/g' $test_layer_target
sed -i 's/<PYTHON_LAYER_HOLDER>/'${data_layer}'/g' $test_layer_target
sed -i 's/<BATCH_SIZE_HOLDER>/'${batch_sz}'/g' $test_layer_target
sed -i 's/<TRAIN_BATCH_NUM_HOLDER>/'${train_batches}'/g' $train_layer_target
sed -i 's/<TEST_BATCH_NUM_HOLDER>/'${test_batches}'/g' $test_layer_target
sed -i 's:<TRAIN_CONFIG_FILE_HOLDER>:'${train_cfg}':g' $train_layer_target
sed -i 's:<TEST_CONFIG_FILE_HOLDER>:'${test_cfg}':g' $test_layer_target

# when using resNet structure
sed -i 's:<CHANNEL_NUM>:'$(($scale * $scale))':g' $train_layer_target
sed -i 's:<CHANNEL_NUM>:'$(($scale * $scale))':g' $test_layer_target
sed -i 's:<DECONV_KERNAL_SIZE>:'$((2*$scale-$scale%2))':g' $train_layer_target
sed -i 's:<DECONV_KERNAL_SIZE>:'$((2*$scale-$scale%2))':g' $test_layer_target
sed -i 's:<STRIDE>:'$scale':g' $train_layer_target
sed -i 's:<STRIDE>:'$scale':g' $test_layer_target
sed -i 's:<PAD_SIZE>:'$(($scale/2))':g' $train_layer_target
sed -i 's:<PAD_SIZE>:'$(($scale/2))':g' $test_layer_target
sed -i 's:<CROP_SIZE>:'$(($scale*$mrgSize))':g' $train_layer_target
sed -i 's:<CROP_SIZE>:'$(($scale*$mrgSize))':g' $test_layer_target

# execution
arg="$caffe --gpu=$gpu_id --save-path=$savepath --solver-param=$solver_target"
echo $arg
python $arg

exit




