#!/data/ASR1/tools/python27/bin/python

import os
import sys

if __name__ == '__main__':


   fout = open('feat.ark', 'w')

   keys = []
   for line in open('../train_360h/full_image_list','r'):
       path = line.replace('\n','').split(' ')[0]
       key = path.split('/')[-1].replace('.JPEG','')
       keys.append(key)

   index = 0
   for line in open('feature','r'):
       line = line.replace('\n','').strip()
       fout.write(keys[index] + ' [ ' + line + ' ]\n')
       index = index + 1

   fout.close() 
