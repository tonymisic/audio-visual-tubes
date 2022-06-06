import glob, cv2, subprocess, numpy as np, csv
from PIL import Image

def sampleframes(length, training_samples, training_samplerate):
    overlap = (length - 1) - (training_samples * training_samplerate)
    if overlap < 0: # repeat video
        while length - 1 <= (training_samples * training_samplerate):
            length = length * 2
            middle_index = int(length / 2)
        a = list(range(middle_index - training_samplerate, -1, -training_samplerate))[0:int(training_samples/2)]
        b = list(range(middle_index, length, training_samplerate))[0:int(training_samples/2)]
        a.reverse()
        a.extend(b)
        if len(a) < 16: # indexing error, will stop dataloader in its tracks
            print("Array: " + str(a) + " Length: " + str(len(a)) + " Middle index: " + middle_index)
        return a
    else: # same video
        middle_index = int(length / 2)
        a = list(range(middle_index - training_samplerate, -1, -training_samplerate))[0:int(training_samples/2)]
        b = list(range(middle_index, length, training_samplerate))[0:int(training_samples/2)]
        a.reverse()
        a.extend(b)
        if len(a) < 16: # indexing error, will stop dataloader in its tracks
            print("Array: " + str(a) + " Length: " + str(len(a)) + " Middle index: " + middle_index)
        return a

def get_frames(path):
    cap = cv2.VideoCapture(path)
    frame_counter, success, frames, length = 0, True, [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length < 2:
        return False
    indicies = sampleframes(length, 16, 16)
    backup_frame = []
    for index in indicies:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index % length))
        success, image = cap.read()
        if not success and frame_counter < length:
            if backup_frame != []:
                frames.append(backup_frame)
            else:
                return False
        else:
            backup_frame = image
            frames.append(image)
        frame_counter += 1
    cap.release()
    return frames

def to_jpgs():
    video_folder = '/media/datadrive/flickr/videos/'
    videos = glob.glob(video_folder + "*.mp4")
    count = 0
    for file in videos[1:]:
        frames = get_frames(file)
        if frames != False:
            subprocess.call(['mkdir', video_folder + file.split('/')[5].split('.')[0]])
            for i, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame)[:,:,::-1]).save(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
                print(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
        else:
            print("Missing frame data for video: " + file)
        count += 1
        print("Progress: " + str(count) + "/" + str(len(videos[1:])))

def file_to_jpgs(file):
    video_folder = '/media/datadrive/flickr/videos/'
    frames = get_frames(file)
    if frames != False:
        subprocess.call(['mkdir', video_folder + file.split('/')[5].split('.')[0]])
        for i, frame in enumerate(frames):
            Image.fromarray(np.asarray(frame)[:,:,::-1]).save(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
            print(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
    else:
        print("Missing frame data for video: " + file)

def hardway_to_jpgs():
    video_folder = '/media/datadrive/flickr/FLICKR_5k/videos/'
    save_folder = '/media/datadrive/flickr/FLICKR_5k/my_frames/'
    videos = glob.glob(video_folder + "*.mp4")
    count, fail = 0, 0
    for file in videos:
        frames = get_frames(file)
        if frames != False:
            frame = frames[7]
            Image.fromarray(np.asarray(frame)[:,:,::-1]).save(save_folder + file.split('/')[6].split('.')[0] + '.jpg')
            print(save_folder + file.split('/')[6].split('.')[0] + '.jpg')
        else:
            print("Missing frame data for video: " + file)
            fail += 1
        count += 1
        print("Progress: " + str(count) + "/" + str(len(videos)) + " Failed: " + str(fail))

def test_to_jpgs():
    video_folder = '/media/datadrive/flickr/videos/'
    testcsv, data = 'metadata/flickr_test.csv', []
    with open(testcsv) as f:
        csv_reader = csv.reader(f)
        for item in csv_reader:
            data.append(item[0] + '.mp4')
    count = 0
    for file in data:
        frames = get_frames(video_folder + file)
        if frames != False:
            subprocess.call(['mkdir', video_folder + file.split('.')[0]])
            for i, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame)[:,:,::-1]).save(video_folder + file.split('.')[0] + '/' +  str(i) + '.jpg')
                print(video_folder + file.split('.')[0] + '/' +  str(i) + '.jpg')
        else:
            print("Missing frame data for video: " + file)
        count += 1
        print("Progress: " + str(count) + "/" + str(len(data)))

test_to_jpgs()