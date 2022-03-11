import glob, cv2, subprocess, numpy as np
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
    indicies = sampleframes(length, 16, 16)
    if length < 2:
        return False
    for index in indicies:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index % length))
        success, image = cap.read()
        if not success and frame_counter < length:
            return False
        else:
            frames.append(image)
        frame_counter += 1
    cap.release()
    return frames

def to_jpgs():
    video_folder = '/media/datadrive/flickr/videos/'
    videos = glob.glob(video_folder + "*.mp4")
    for file in videos[1:]:
        frames = get_frames(file)
        if frames != False:
            subprocess.call(['mkdir', video_folder + file.split('/')[5].split('.')[0]])
            for i, frame in enumerate(frames):
                Image.fromarray(np.asarray(frame)[:,:,::-1]).save(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
                print(video_folder + file.split('/')[5].split('.')[0] + '/' +  str(i) + '.jpg')
                input()
        else:
            print("Missing frame data for video: " + file)
        break

to_jpgs()
