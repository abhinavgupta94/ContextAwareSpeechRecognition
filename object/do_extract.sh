#!/bin/bash

export LD_LIBRARY_PATH=/data/ASR5/babel/ymiao/Install/cudalib/lib64:/data/ASR5/babel/ymiao/Install/glog-0.3.3/lib:/data/ASR5/babel/ymiao/Install/gflags/lib:/home/ymiao/local/lib:/data/ASR5/babel/ymiao/Install/leveldb-1.15.0:/data/ASR5/babel/ymiao/Install/snappy-1.1.2/lib:/data/ASR5/babel/ymiao/Install/mdb-mdb/libraries/liblmdb/lib:/data/ASR5/babel/ymiao/Install/opencv/lib:/data/ASR5/babel/ymiao/Install/gcc-4.6.3/lib:/data/ASR5/babel/ymiao/Install/opencv_libraries/lib:$LD_LIBRARY_PATH

n=$no

cat alexnet_val.prototxt | sed "s/img_list/full_image_list.$n/g" > alexnet_val.prototxt.$n

batch_num=`cat full_image_list.$n | wc -l`
blob_name=prob

rm -rf features_${blob_name}_$n

/data/ASR5/babel/ymiao/tools/caffe-dev/build/tools/extract_features.bin bvlc_alexnet.caffemodel alexnet_val.prototxt.$n ${blob_name} features_${blob_name}_$n $batch_num CPU 2> Log.$n

./read_leveldb.py features_${blob_name}_$n feature.$n


