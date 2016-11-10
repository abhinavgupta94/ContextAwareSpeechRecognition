#!/data/ASR1/tools/python27/bin/python

import os
import sys

if __name__ == '__main__':


   fout = open('feat.ark', 'w')

   for n in range(1, 40):
       keys = []
       for line in open('full_image_list.' + str(n),'r'):
           path = line.replace('\n','').split(' ')[0]
           key = path.split('/')[-1].replace('.JPEG','')
           keys.append(key)

       index = 0
       for line in open('feature.' + str(n),'r'):
           line = line.replace('\n','').strip()
           fout.write(keys[index] + ' [ ' + line + ' ]\n')
           index = index + 1

   fout.close() 
