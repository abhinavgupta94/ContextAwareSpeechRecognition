import os
import sys
import random

if __name__ == '__main__':

   random.seed(88997)
   fout = open('all_images','w')
   for line in open('segments'):
       elements = line.replace('\n','').split(' ')
       utt_ID = elements[0]
       video_ID = elements[1]
       start = float(elements[2])
       end = float(elements[3])
       cut_point = start + random.random() * (end - start)
       fout.write(utt_ID + ' ' + video_ID + ' ' + str(int(cut_point)) + '\n')
   fout.close()
