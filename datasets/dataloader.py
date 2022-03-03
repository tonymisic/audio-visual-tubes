import cv2, torch, csv, numpy as np, random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from scipy import signal
import soundfile as sf
from torchvision.transforms.functional import crop
from torchvideotransforms import video_transforms, volume_transforms

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data = []
        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test_hardway.csv'
            traincsv = 'metadata/flickr_train.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/vggss_test.csv'
            traincsv = 'metadata/vggss_train.csv'
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
        self.audio_path = args.og_data_path + 'audio/'
        self.frame_path = args.og_data_path + 'frames/'
        self.imgSize = args.image_size 
        self.cropSize = 150
        self.mode = mode
        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()
        #  Retrieve list of audio and video files
        self.video_files = []
   
        for item in data[:]:
            self.video_files.append(item )
        print(len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])            
    def croptopLeft(self, image):
        return crop(image, 0, 0, self.cropSize, self.cropSize)
    def cropbottomRight(self, image):
        return crop(image, image.size[0] - self.cropSize, image.size[1] - self.cropSize, 
                    self.cropSize, self.cropSize)
    def cropmiddleLeft(self, image):
        return crop(image, int(image.size[0] / 4), 0, 
                    self.cropSize, self.cropSize)
    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]
        # Image
        frame = self.img_transform(self._load_frame(self.frame_path + file[:-3] + 'jpg'))
        frame_ori = np.array(self._load_frame(self.frame_path  + file[:-3] + 'jpg'))
        # Audio
        samples, samplerate = sf.read(self.audio_path + file[:-3]+'wav')
        # repeat if audio is too short
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*10]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512, noverlap=1)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
        return frame,spectrogram,resamples,file,torch.tensor(frame_ori)



class SubSampledFlickr(Dataset):
    def __init__(self, args, mode='train', transforms=None):
        self.args = args
        self.middle_sample = False
        self.backup_singleframe = None
        if args.frame_density < 2:
            self.middle_sample = True
            self.training_samples = args.frame_density
            self.training_samplerate = args.frame_density
        else:
            self.training_samples = args.frame_density
            self.training_samplerate = args.frame_density
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
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])   
    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
    
    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indicies, failed, findex = self.sampleframes(frame_count), False, []
        successful_frame = []
        random.shuffle(indicies) # randomize order of frames
        if self.mode == 'train':
            frames = []
            for index in indicies:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index % frame_count))
                success, image = cap.read()
                if success:
                    frames.append(image)
                    successful_frame = image
                else:
                    failed = True
                    findex.append(index)
            if failed:
                for i in findex:
                    print("Frame "+ str(i) +" failed to load for video: " + str(path) + ", loaded backup frame")
                    frames.append(successful_frame)
            cap.release()
            return np.asarray(frames)

        if self.mode == 'test':
            counter = 1 # starts at sampling rate index
            frames = []
            success = True
            while success:
                if counter % self.args.sampling_rate == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
                    success, image = cap.read()
                    if success:
                        frames.append(image)
                counter += 1
            cap.release()
            return frames
    def _load_middleframe(self, path):
        cap = cv2.VideoCapture(path)
        frame_middle = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_middle)
        success, image = cap.read()
        if success:
            self.backup_singleframe = image
            return Image.fromarray(np.asarray(image)).convert('RGB')
        else:
            frame_middle += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_middle)
            success, image = cap.read()
            if success:
                self.backup_singleframe = image
                return Image.fromarray(np.asarray(image)).convert('RGB')
            else:
                return Image.fromarray(np.asarray(self.backup_singleframe)).convert('RGB')

    def sampleframes(self, length):
        indicies = []
        overlap = (length - 1) - (self.training_samples * self.training_samplerate)
        if overlap < 0: # repeat video
            while length - 1 <= (self.training_samples * self.training_samplerate):
                length = length * 2
                middle_index = int(length / 2)
            a = list(range(middle_index - self.training_samplerate, -1, -self.training_samplerate))[0:int(self.training_samples/2)]
            b = list(range(middle_index, length, self.training_samplerate))[0:int(self.training_samples/2)]
            a.reverse()
            a.extend(b)
            if len(a) < 16: # indexing error, will stop dataloader in its tracks
                print("Array: " + str(a) + " Length: " + str(len(a)) + " Middle index: " + middle_index)
            return a
        else: # same video
            middle_index = int(length / 2)
            a = list(range(middle_index - self.training_samplerate, -1, -self.training_samplerate))[0:int(self.training_samples/2)]
            b = list(range(middle_index, length, self.training_samplerate))[0:int(self.training_samples/2)]
            a.reverse()
            a.extend(b)
            if len(a) < 16: # indexing error, will stop dataloader in its tracks
                print("Array: " + str(a) + " Length: " + str(len(a)) + " Middle index: " + middle_index)
            return a
    def __len__(self):
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]
        # Audio
        samples, samplerate = sf.read(self.audio_path + file[:-3] +'wav')
        # Video
        #start_time = time.time()
        if self.middle_sample:
            frames = self.img_transform(self._load_middleframe(self.video_path + file[:-3] + 'mp4'))
        else:
            frames = self.video_transform(self._load_video(self.video_path + file[:-3] + 'mp4'))
        #print("Completed video ID: " + str(idx) + " Time taken: %ss" % (round(time.time() - start_time, 2)))
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

class PerFrameLabels(Dataset):
    def __init__(self, args, mode='test', transforms=None):
        data = []
        if args.testset == 'flickr':
            if mode == 'val':
                testcsv = 'metadata/flickr_val.csv'
            else:
                testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/vggss_test.csv'

        with open(testcsv) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                data.append(item[0] + '.mp4')
        self.audio_path = '/media/datadrive/flickr/FLICKR_5k/audio/'
        self.frame_path = args.data_path + 'frames/'
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
            self.video_files.append(item)
        print(len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            self.video_transform = video_transforms.Compose([
			    volume_transforms.ClipToTensor()
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])        
            self.video_transform = video_transforms.Compose([
                video_transforms.Resize(self.imgSize, interpolation='bicubic'),
                video_transforms.CenterCrop(self.imgSize),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean, std)
            ])

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img
    
    def _load_video(self, path):
        frames = []
        try:
            cap = cv2.VideoCapture(path)
        except:
            print("Error in video loading, sent previous video to loader.")
            return self.previous_video
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
        cap.release()
        if len(frames) <= 1:
            print("Frame data empty, sent previous video to loader.")
            return self.previous_video
        self.previous_video = frames
        return np.asarray(frames)

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def __getitem__(self, idx):
        file = self.video_files[idx]
        # Audio
        samples, samplerate = sf.read(self.audio_path + file[:-3]+'wav')
        # Video
        frames = self.video_transform(self._load_video(self.video_path + file[:-3] + 'mp4'))
        # repeat if audio is too short
        if samples.shape[0] < samplerate * 10:
            n = int(samplerate * 10 / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*10]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512, noverlap=1)
        spectrogram = np.log(spectrogram+ 1e-7)
        spectrogram = self.aid_transform(spectrogram)
        return frames,spectrogram,resamples,samplerate,file