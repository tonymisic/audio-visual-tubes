import soundfile as sf, glob, subprocess

all_audio = glob.glob('/media/datadrive/flickr/audio/*.wav')
count, failed = 0, 0
for filename in all_audio:
    try:
        _, _ = sf.read(filename)
    except:
        failed += 1
        subprocess.call("rm " + filename, cwd="/", shell=True)
    count += 1
    print("Progress: " + str(count) + "/" + str(len(all_audio)) + " Failed: " + str(failed))