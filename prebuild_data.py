import math
import glob
import os
import numpy as np
import cv2
import imutils
import shutil
import argparse
from random import randint

# constants
IMG_WIDTH = 178
CROP_HEIGHT_BEG = 42
EXPAND_TIMES = 5
CROPPED_SIZE = 160
# MAX_ROT = 0
# MAX_TRANS = 0
MAX_ROT = 3
MAX_TRANS = 4

y1 = CROP_HEIGHT_BEG
y2 = CROP_HEIGHT_BEG+CROPPED_SIZE
x1 = IMG_WIDTH//2-CROPPED_SIZE//2
x2 = IMG_WIDTH//2+CROPPED_SIZE//2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--base_path', dest="base_path", type=str, help='')
parser.add_argument('--postfix', dest="postfix", nargs='+', type=str, help='')
args = parser.parse_args()

base_path = args.base_path
cnt = 0
for postfix in args.postfix:
    input_dir = base_path + '/img_' + postfix
    output_dir = base_path + '/crop_' + postfix

    print('Solving postfix={}'.format(postfix))

    def solve_dir(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    solve_dir(output_dir)

    for file_path_name in glob.glob(input_dir + '/*.*'):
        filename_without_ext = os.path.splitext(os.path.split(file_path_name)[1])[0]
        img = cv2.imread(file_path_name, 1)
        for i in range(EXPAND_TIMES):
            rotated = imutils.rotate(img, angle=randint(-MAX_ROT,+MAX_ROT))
            translated = imutils.translate(rotated, randint(-MAX_TRANS,+MAX_TRANS), randint(-MAX_TRANS,+MAX_TRANS))
            cropped = translated[y1:y2,x1:x2]
            cv2.imwrite('{}/{}-{}.png'.format(output_dir, filename_without_ext, i), cropped)
            # test output eye crop
            # left_eye = cropped[70-24:70+24,80-48:80]
            # cv2.imwrite('{}/left-{}-{}.png'.format(output_dir, filename_without_ext, i), left_eye)

        cnt += 1
        if cnt % 100 == 0:
            print('count: {}'.format(cnt))