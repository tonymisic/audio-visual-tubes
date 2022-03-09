from turtle import down
import urllib.request, cv2, soundfile as sf, csv, glob, subprocess
from matplotlib.style import available

video_download_folder = '/media/datadrive/flickr/videos/'
audio_folder = '/media/datadrive/flickr/audio/'
download_file = 'metadata/urls_public.txt'
training_file = '/home/tmisic/audio-visual-tubes/metadata/flickr_test.csv'

def good_video(path):
    cap = cv2.VideoCapture(path)
    frame_counter, success = 0, True
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 2:
        return False
    while success:
        success, _ = cap.read()
        if not success and frame_counter < cap.get(cv2.CAP_PROP_FRAME_COUNT):
            return False
        frame_counter += 1
    cap.release()
    return True

def good_audio(path):
    try:
        _, _ = sf.read(path)
        return True
    except:
        return False

def get_finished():
    map = {}
    with open(training_file, 'r') as data:
        for line in csv.reader(data):
            map[line[0]] = line[1]
    return map

def get_possible():
    temp = open(download_file, 'r')
    map = {}
    for line in temp.readlines():
        map[line.split('/')[5]] = line
    temp.close()
    return map

def get_audio():
    temp = glob.glob(audio_folder + "*.wav")
    map = {}
    for line in temp:
        map[line.split('/')[5].rstrip('.wav')] = line
    return map

def download():
    possible_downloads = get_possible()
    already_done = get_finished()
    available_audio = get_audio()
    success_count, corrupted_count, download_error_count = len(already_done), 0, 0
    left_count = len(available_audio) - success_count
    for file in available_audio.keys():
        if file not in already_done and file in possible_downloads:
            try:
                urllib.request.urlretrieve(possible_downloads[file], video_download_folder + file + '.mp4')
                if good_video(video_download_folder + file + '.mp4') and good_audio(audio_folder + file + '.wav'):
                    print("Downloaded " + file + " successfully.")
                    success_count += 1
                else:
                    corrupted_count += 1
                    subprocess.call(['rm', video_download_folder + file + '.mp4'])
                    subprocess.call(['rm', audio_folder + file + '.wav'])
                    print("File " + file + " was corrupted, removed audio and video.")
            except:
                download_error_count += 1
                subprocess.call(['rm', audio_folder + file + '.wav'])
                print("File " + file + " was unreachable, removed linked audio.")
        left_count -= 1
        print("Video count: " + str(success_count) + " Possible Downloads" + str(left_count) + \
            " Download errors: " + str(download_error_count) + " Corrupted: " + str(corrupted_count))

def download_defined():
    possible_downloads = get_possible()
    already_done = get_finished()
    available_audio = get_audio()
    for file in already_done.keys():
        if file not in available_audio and file in possible_downloads:
            try:
                urllib.request.urlretrieve(possible_downloads[file], video_download_folder + file + '.mp4')
                if good_video(video_download_folder + file + '.mp4') and good_audio(audio_folder + file + '.wav'):
                    print("Downloaded " + file + " successfully.")
                    success_count += 1
                else:
                    corrupted_count += 1
                    #subprocess.call(['rm', video_download_folder + file + '.mp4'])
                    #subprocess.call(['rm', audio_folder + file + '.wav'])
                    print("File " + file + " was corrupted, removed audio and video.")
            except:
                download_error_count += 1
                #subprocess.call(['rm', audio_folder + file + '.wav'])
                print("File " + file + " was unreachable, removed linked audio.")

def clean_up():
    completed = get_finished()
    all_videos = glob.glob(video_download_folder + '/*')
    i, count = 0, len(all_videos)
    for vid in all_videos:
        if vid.split('/')[5].split('.')[0] not in completed:
            subprocess.call(['rm', video_download_folder + vid.split('/')[5]])
        i += 1
        print("Progress: " + str(i) + "/" + str(count))

download_defined()