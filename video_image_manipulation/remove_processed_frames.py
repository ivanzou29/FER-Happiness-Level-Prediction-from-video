import os


main_dir = 'frames'

def remove_frames(main_dir):
    for frame in os.listdir(main_dir):
        frame_dir = os.path.join(main_dir, frame)
        for file in os.listdir(frame_dir):
            file_path = os.path.join(frame_dir, file)
            if not os.path.isdir(file_path):
                os.remove(file_path)
                print(file_path + " removed.")

def remove_empty_dir(main_dir):
    for frame in os.listdir(main_dir):
        frame_dir = os.path.join(main_dir, frame)
        if len(os.listdir(frame_dir)) == 0:
            os.system('rm -rf {0}'.format(frame_dir))
            print(frame_dir + " removed.")





if __name__ == '__main__':
    remove_frames(main_dir)
    remove_empty_dir(main_dir)
