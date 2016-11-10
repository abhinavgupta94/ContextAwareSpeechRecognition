#!/data/ASR1/tools/python27/bin/python

import os
import sys

import leveldb
import caffe_pb2
import numpy as np

if __name__ == '__main__':

    db_path = sys.argv[1]
    out_path = sys.argv[2]

    fout = open(out_path, 'w')

    db = leveldb.LevelDB(db_path)
    size =  len(list(db.RangeIter(include_value = False)))
    for n in xrange(size):
        key = str(n)
        for m in xrange(10 - len(key)):
            key = '0' + key
        datum = caffe_pb2.Datum.FromString(db.Get(key))
#        array = np.array(datum.float_data).astype(float).reshape(datum.height,)
        array = np.array(datum.float_data).astype(float)      
        output = ''
        for m in xrange(len(array)):
            output = output + str(array[m]) + ' '
        fout.write(output.strip() + '\n')

   fout.close() 
