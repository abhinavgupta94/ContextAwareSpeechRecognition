#!/bin/bash

string=""

for n in `seq 1 40`; do
  string="$string all_images.$n"
done

/data/ASR5/babel/ymiao/Install/kaldi-latest/egs/swbd/s5b_nochange/utils/split_scp.pl all_images $string
