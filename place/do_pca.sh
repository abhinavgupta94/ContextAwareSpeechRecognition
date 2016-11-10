#!/bin/bash

# convert feature.* files into a single feat.ark

#./get_leveldb_ark.py

# est pca transform, may take a while

pca_dim=100   # dimension of PCA

/data/ASR5/babel/ymiao/Install/kaldi-latest/src/bin/est-pca --dim=$pca_dim --read-vectors=true ark:feat.ark pca.mat.d$pca_dim


# apply pca.mat.d* to the orginal features
# the final features will be saved in "final.ark" and "final.scp"

/data/ASR5/babel/ymiao/Install/kaldi-latest/src/bin/transform-vec pca.mat.d$pca_dim ark:feat.ark ark,scp:final.ark,final.scp

# remove feat.ark

#rm -f feat.ark

