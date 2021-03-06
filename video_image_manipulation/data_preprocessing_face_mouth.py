import numpy as np
import os
import cv2
import math

face = 'face.png'
nose = 'nose.png'
mouth = 'mouth.png'
left_eye = 'lefy_eye.png'
right_eye = 'right_eye.png'
region_list = ['face.png', 'mouth.png']
standard_length = 10

def read_data(img_dir):
    img_data = []
    frame_list = []
    for frame_info in os.listdir(img_dir):
        frame_dir = os.path.join(img_dir, frame_info)
        if os.path.isdir(frame_dir):
            frame_list.append(int(frame_info))
    frame_list.sort()

    for frame_num in frame_list:
        try:
            imgs = []
            dir = os.path.join(img_dir, str(frame_num))
            for region in region_list:
                img = cv2.imread(os.path.join(dir, region))
                if img.shape != (56, 56, 3):
                    img = cv2.resize(img, dsize=(56, 56), interpolation=cv2.INTER_NEAREST)
                img = np.transpose(img, (2, 0, 1))
                imgs.append(img)
            img_data.append(imgs)
        except:
            print("skipped one frame in " + img_dir)
    img_data = np.array(img_data)
    return img_data

def unify_sequence_length(data, standard_length):
    cropped_data = []
    original_length = data.shape[0]
    interval = original_length / standard_length
    for i in range(standard_length):
        index = special_round(i * interval)
        if (index == original_length):
            index = index - 1
        index = int(index)
        print("index is: " + str(index))
        cropped_data.append(data[index])
    return np.array(cropped_data)

def special_round(num):
    if num % 1 > 0.5:
        return round(num)
    else:
        return math.floor(num)

def get_label(img_dir):
    start = img_dir.rfind('/') + 1
    end = img_dir.rfind('_')
    label = float(img_dir[start:end])
    return label

def get_all_data(main_dir):
    data = []
    label = []
    for img_frames in os.listdir(main_dir):
        print("loading " + img_frames + "...")
        img_dir = os.path.join(main_dir, img_frames)
        data.append(unify_sequence_length(read_data(img_dir), standard_length))
        label.append(get_label(img_dir))
    data = np.array(data)
    label = np.array(label)
    return data, label

if __name__ == '__main__':
    data, label = get_all_data('frames')
