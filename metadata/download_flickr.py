import os
import urllib.request
video_download_folder = '/media/datadrive/flickr/videos/'
download_file = 'metadata/urls_public.txt'
notfound_file = 'metadata/filesnotfound.txt'
temp = open(download_file, 'r')


count, unloaded = 201000, 0
for i, video_location in enumerate(temp.readlines(), 201000):
    videoname = video_location.split('/')[5] + '.mp4'
    print("Saving to: " + video_download_folder + videoname)
    try:
        urllib.request.urlretrieve(video_location, video_download_folder + videoname)
    except:
        temp2 = open(notfound_file, 'a')
        temp2.write(videoname + '\n')
        temp.close()
        unloaded += 1
        print("404 Not found count: " + str(unloaded))
    print("Progress: " + str(count) + " / 2214725")
    count += 1
temp.close()