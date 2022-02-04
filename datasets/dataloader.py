from cgi import test
import os
import cv2
import json
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
import time
from PIL import Image
import glob, time
import sys 
import scipy.io.wavfile as wav
from scipy import signal
import soundfile as sf
from torchvision.transforms.functional import crop
from torchvideotransforms import video_transforms, volume_transforms

class SubSampledFlickr(Dataset):
    def __init__(self, args, mode='train', transforms=None):
        self.args = args
        self.training_samples = 16
        self.training_samplerate = 16
        data = []
        if args.testset == 'flickr':
            traincsv = 'metadata/flickr_train.csv'
            testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            traincsv = 'metadata/vggss_train.csv'
            testcsv = 'metadata/vggss_test.csv'

        if mode == 'train':
            with open(traincsv) as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    data.append(item[0] + '.mp4')
        elif mode == 'test':
            with open(testcsv) as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    data.append(item[0] + '.mp4')
        self.audio_path = args.data_path + 'audio/'
        self.video_path = args.data_path + 'videos/'
        self.imgSize = args.image_size 
        self.cropSize = 150
        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []
        self.previous_video = None
        for item in data[:]:
            self.video_files.append(item )
        print(len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.video_transform = video_transforms.Compose([
                video_transforms.Resize(int(self.imgSize * 1.1), interpolation='bicubic'),
                video_transforms.RandomCrop(self.imgSize),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.CenterCrop(self.imgSize),
			    volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean, std)
            ])
        else:  
            self.video_transform = video_transforms.Compose([
                video_transforms.Resize(self.imgSize, interpolation='bicubic'),
                video_transforms.CenterCrop(self.imgSize),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean, std)
            ])

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
    
    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indicies = self.sampleframes(frame_count)
        if self.mode == 'train':
            frames = []
            for index in indicies:
                if index >= frame_count:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index % frame_count))
                else:  
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, image = cap.read()
                frames.append(image)
            cap.release()
            return np.asarray(frames)

        if self.mode == 'test':
            counter = 1 # starts at sampling rate index
            frames = []
            while success:
                if counter % self.args.sampling_rate == 0:
                    success, image = cap.read()
                    frames.append(image)
                counter += 1
            cap.release()
            return np.asarray(frames)

    def sampleframes(self, length):
        indicies = []
        overlap = length - (self.training_samples * self.training_samplerate)
        if overlap < 0: # repeat video
            while length + 1 < (self.training_samples * self.training_samplerate):
                length = length * 2
                middle_index = int(length / 2)
            count = 0
            for i in range(middle_index - self.training_samplerate, 0, -self.training_samplerate):
                if count == self.training_samples / 2:
                    indicies.reverse()
                    break
                else:
                    indicies.append(i)
                    count += 1
            count = 0
            for i in range(middle_index, length, self.training_samplerate):
                if count == self.training_samples / 2:
                    break
                else:
                    indicies.append(i)
                    count += 1
            return indicies
        else: # same video
            middle_index = int(length / 2)
            count = 0
            for i in range(middle_index - self.training_samplerate, 0, -self.training_samplerate):
                if count == self.training_samples / 2:
                    indicies.reverse()
                    break
                else:
                    indicies.append(i)
                    count += 1
            count = 0
            for i in range(middle_index, length, self.training_samplerate):
                if count == self.training_samples / 2:
                    break
                else:
                    indicies.append(i)
                    count += 1
            return indicies
    
    def __len__(self):
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]
        # Audio
        samples, samplerate = sf.read(self.audio_path + file[:-3] +'wav')
        # Video
        start_time = time.time()
        frames = self.video_transform(self._load_video(self.video_path + file[:-3] + 'mp4'))
        print("Completed video ID: " + str(idx) + " Time taken: %ss" % (round(time.time() - start_time, 2)))
        # repeat if audio is too short
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*10]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1
        _, _, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512, noverlap=1)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
        return frames, spectrogram, resamples, samplerate, file