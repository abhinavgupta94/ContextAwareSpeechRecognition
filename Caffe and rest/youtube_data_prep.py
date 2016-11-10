
import os
import sys

def process_text(input_text):
    text = input_text
    if input_text.endswith('.') or input_text.endswith(',') or input_text.endswith('?'):
        text = input_text[:-1]
    text = text.replace('. ', ' ').replace(', ', ' ').replace('?', ' ').replace('!', ' ').replace('\"', ' ').replace('`', '')
    text = text.replace('(', '').replace(')', '')
    text = text.replace(' \'', ' ').replace('\' ', ' ')
    text = text.replace(';', ' ').replace(':', ' ')
    text = text.replace(' - ', ' ')
    text = text.replace('\xe2\x80\x99', '\'')
    # remove ......
    pattern = '.'
    for i in range(2, 30):
        pattern = pattern + '.'
        text = text.replace(pattern, ' ')
    text = text.replace('..', ' ')
    text = text.replace(' .', ' ')
    text = text.replace('. ', ' ')
    for i in range(1, 100):
        text = text.replace('  ', ' ') 
    text = text.strip()
    return text.upper()

def convert_2_sec(input_str):
    strs = input_str.split(',')
    elements = strs[0].split(':')
    return str(3600 * int(elements[0]) + 60 * int(elements[1]) + int(elements[2])) + '.00'

if __name__ == '__main__':

   import sys
   if len(sys.argv) != 2:
       exit(1)

   fout_utt2spk = open('utt2spk','w')
   fout_text = open('text','w')
   fout_seg = open('segments','w')
   fout_raw = open('text.raw','w')

   srt_dir = sys.argv[1]
   for file in os.listdir(srt_dir):
       if (not file.endswith('.srt')):
           continue
       video_id = file.replace('.srt','')
       content = []
       for line in open(os.path.join(srt_dir, file)):
           line = line.strip()
           if line != '':
               content.append(line)
       utt_num = len(content) / 3
       # we are discarding the final sentense
              
       trans = ''
       raw = ''
       for n in xrange(utt_num):
           # uttID
       #    utt_id = video_id + '_' + content[3*n]
           # time
       #    times = content[3*n+1].split(' --> ')
       #    start_time = convert_2_sec(times[0])
       #    end_time = convert_2_sec(times[1])
           # trans
           raw = raw + ' ' + content[3*n+2]
           trans = trans + ' ' + process_text(content[3*n+2])
       if trans == '':
           continue
       fout_utt2spk.write(video_id + ' ' + video_id + '\n')
       fout_text.write(video_id + ' ' + trans + '\n')
       fout_raw.write(video_id + ' ' + raw + '\n')

   fout_utt2spk.close()
   fout_text.close()
   fout_seg.close()
   fout_raw.close()
