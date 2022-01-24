import os
import requests

url_file = './metadata/urls_public.txt'
test_set_file = './metadata/flickr_test.csv'
save_location = '/media/datadrive/flickr/FLICKR_5k/videos/'
# get names of test files.
test_files = open(test_set_file).readlines()
links = open(url_file).readlines()
files_to_find = {}
count = 0
for names in test_files:
    name, _ = names.split(',')
    for link in links:
        if name in link: 
            r = requests.get(link, allow_redirects=True)
            open(save_location + name + '.mp4', 'wb').write(r.content)
            print(str(count + 1) + "/ 249")
            print("Video Downloaded: " + name + " Link: " + link)
            break
    count += 1