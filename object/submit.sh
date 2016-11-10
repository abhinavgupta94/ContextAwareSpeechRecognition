#!/bin/bash

for n in `seq 1 40`; do
  qsub -j eo -S /bin/bash -o . -N decode_$n -l nodes=1:ppn=1,walltime=36:00:00 -d . -v no=$n ./convert.sh
done
