import os


def extract_frames(path, filename):
    os.system("mkdir frames/{0}".format(filename[:12]))
    os.system("ffmpeg -i {0} -f image2 -vf fps=fps=20 frames/{1}/output%d.png".format(os.path.join(path, filename),
                                                                                      filename[:12]))


paths = ['Happy', 'Neutral', 'Sad']

if __name__ == '__main__':
    os.system("mkdir frames")
    for path in paths:
        for filename in os.listdir(path):
            if (filename.endswith(".avi")):
                extract_frames(path, filename)
