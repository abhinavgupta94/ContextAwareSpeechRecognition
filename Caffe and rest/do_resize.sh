#!/bin/bash

n=$no

python resize_img.py imglist.$n resize 256 >& Log/log.$n 

