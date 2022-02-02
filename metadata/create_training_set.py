import os, argparse, random

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='', type=str, help='Root directory path of videos')
    parser.add_argument('--train_file', default='', type=str, help='File to save video names')
    parser.add_argument('--subset_size', default=10000, type=int, help='Size of randomized training set')
    return parser.parse_args()

def main():
    args = get_arguments()
    assert args.subset_size in [5000, 10000, 144000], "Subset invalid"
    files = os.listdir(args.video_path)
    indicies = random.sample(range(0, len(files) - 1), args.subset_size)
    selected_files = [files[i] for i in indicies]
    
    for file in selected_files:
        savefile = open(args.train_file, "a")
        savefile.write(file.rstrip(".mp4") + ",0\n") # no classes yet, would replace 0 with class number if needed
        savefile.close()

if __name__ == "__main__":
    main()