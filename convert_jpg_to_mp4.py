import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, re
import sys
import pdb

def read_cur_imgs(root, files):
    imgs = []
    for f in range(len(files)):
        img = cv2.imread(os.path.join(root, files[f]))
        imgs.append(img)
    return imgs

def read_imgs(imgs_dir, imgs_prefix):
    imgs = []
    flows = []

    concat_imgs = []
    out_prefix = 'outputs/'

    for root, dirs, files in os.walk(imgs_dir):
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        imgs += read_cur_imgs(root, files)

        for img, fil in zip(imgs, files):
            if not os.path.exists(root.replace(imgs_prefix, out_prefix)):
                os.makedirs(root.replace(imgs_prefix, out_prefix))
            concat_imgs.append(img)

    return concat_imgs

def gen_video(imgs, vidfile, fps):

    # Different ways to generate video, need to look up
    # fourcc= cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fourcc= cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    try:
        video = cv2.VideoWriter(vidfile, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
        for i in range(len(imgs)):
            video.write(imgs[i])
        video.release()
    except:
        print(len(imgs))
    

def main():
    imgs = []
    root = '/home/tmisic/Localizing-Visual-Sounds-the-Hard-Way/'
    imgs_dir = root + 'imgs/'
    imgs_prefix = ''
    
    # CHANGE THIS IF FRAMES ARE TOO SLOW/FAST
    fps = 30

    for dir_ in os.listdir(imgs_dir):
        if '.mp4' in dir_:
            continue
        print(dir_)
        imgs = read_imgs(os.path.join(imgs_dir, dir_), imgs_prefix)
        gen_video(imgs, os.path.join(imgs_dir, dir_+'.mp4'), fps)
