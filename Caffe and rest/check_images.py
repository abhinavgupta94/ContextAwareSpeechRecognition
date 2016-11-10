import numpy as np
import sys
import os
import cv2

file = sys.argv[1]

for line in open(file, 'r'):
  strings = line.replace('\n','').strip().split(' ')
  file_path = strings[0]
  img = cv2.imread(file_path)
  if img is None:
    continue
  size = img.shape[0] * img.shape[1]
  if size <= 0:
    continue
  print line.replace('\n','')    

