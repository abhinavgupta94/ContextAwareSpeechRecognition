#!/data/ASR1/tools/python27/bin/python

import os
import sys
import cv2

if __name__ == '__main__':

   input_list = sys.argv[1]
   output_dir = sys.argv[2]
   resize_dim = int(sys.argv[3])

   number = 0
   for line in open(input_list, 'r'):
       line = line.replace('\n','').strip()
       input_img = line
       elements = line.split('/')
#       output_img = output_dir + '/' + elements[-2] + '/' + elements[-1]
       output_img = output_dir + '/' + elements[-1]

       img = cv2.imread(input_img)
       height, width, depth = img.shape
       new_height = resize_dim
       new_width = resize_dim
       if height > width:
           new_height = resize_dim * height / width
       else:
           new_width = resize_dim * width / height
       resized_img = cv2.resize(img, (new_width, new_height))
       height_offset = (new_height - resize_dim) / 2
       width_offset = (new_width - resize_dim) / 2
       cropped_img = resized_img[height_offset:height_offset + resize_dim,
                             width_offset:width_offset + resize_dim]
#      cv2.imwrite(output_img, resized_img)
       cv2.imwrite(output_img, cropped_img)
       number = number + 1
       if number % 1000 == 0:
           print number

