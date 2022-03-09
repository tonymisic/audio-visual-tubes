import os, argparse, random, glob, csv

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='', type=str, help='Root directory path of videos')
    parser.add_argument('--train_file', default='', type=str, help='File to save video names')
    parser.add_argument('--val_file', default='', type=str, help='File of val set')
    parser.add_argument('--test_file', default='', type=str, help='File of test set')
    parser.add_argument('--subset_size', default=144000, type=int, help='Size of randomized training set')
    return parser.parse_args()

def main():
    args = get_arguments()
    assert args.subset_size in [5000, 10000, 20000, 144000], "Subset invalid"
    files = os.listdir(args.video_path)
    with open(args.val_file, 'r') as file:
        validation = {rows[0]:rows[1] for rows in csv.reader(file)}
    with open(args.test_file, 'r') as file:
        testing = {rows[0]:rows[1] for rows in csv.reader(file)}
    all_videos = glob.glob('/media/datadrive/flickr/videos/*')
    all_audio = glob.glob('/media/datadrive/flickr/audio/*.wav')
    map_audio, map_video = {}, {}

    for file in all_audio:
        map_audio[file.split('/')[5].rstrip('.wav')] = 0
    for file in all_videos:
        map_video[file.split('/')[5].split('.')[0]] = 0
    final_list = list(set(map_video).intersection(set(map_audio)).difference(set(validation)).difference(set(testing)))
    print(len(final_list))
    indicies = random.sample(range(0, len(final_list) - 1), args.subset_size)
    for name in [final_list[i] for i in indicies]:
        savefile = open(args.train_file, "a")
        savefile.write(name + ",0\n") # no classes, would replace 0 with class number if needed
        savefile.close()

if __name__ == "__main__":
    main()