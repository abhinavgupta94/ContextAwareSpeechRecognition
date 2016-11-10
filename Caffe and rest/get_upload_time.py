import os
import sys
import urllib2
import time

if __name__ == '__main__':
   
    fout = open('video_year','w')

    for line in open('spk2utt'):
        video_id = line.split(' ')[0]
        url = 'http://gdata.youtube.com/feeds/api/videos/' + video_id + '?v=2'
        try:
            webtxt = urllib2.urlopen(url).read()
        except:
            print 'Error on video %s' % (video_id)
            continue
        elements = webtxt.split('published>')
        year = elements[1].split('-')[0]
        fout.write(video_id + ' ' + year + '\n')
 #       time.sleep(2)
   fout.close()