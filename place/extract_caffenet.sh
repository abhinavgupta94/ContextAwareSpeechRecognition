#!/usr/bin/env sh

export LD_LIBRARY_PATH=/data/MM2/iyu/base/anaconda/lib:/data/MM2/iyu/base/local/cuda-6.5/lib64:/data/MM2/iyu/base/lib64:/data/MM2/iyu/base/lib:/data/MM2/iyu/libraries/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
export PATH=/data/ASR5/babel/ymiao/Install2/caffe/build/tools:$PATH

# data_root: the root directory where the JPEG images are saved

rm -rf db_leveldb

data_root=/data/ASR5/abhinav5/train_360h/raw/

convert_imageset \
    --resize_height=256 \
    --resize_width=256 \
    --backend=leveldb \
    $data_root \
    ../train_360h/list_of_images \
    db_leveldb

image_num=`cat ../train_360h/list_of_images | wc -l`

extract_features.bin places205CNN_iter_300000.caffemodel deploy.prototxt prob features_prob $image_num leveldb GPU 2> Log

./read_leveldb.py features_prob feature

./get_leveldb_ark.py
