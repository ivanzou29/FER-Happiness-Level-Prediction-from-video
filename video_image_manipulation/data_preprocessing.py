import numpy as np
import os
import cv2
import math
import torch
face = 'face.png'
nose = 'nose.png'
mouth = 'mouth.png'
left_eye = 'lefy_eye.png'
right_eye = 'right_eye.png'
region_list = ['face.png', 'nose.png', 'mouth.png', 'left_eye.png', 'right_eye.png']
standard_length = 16

img_dir = '/Users/ivanreal/PycharmProjects/Happiness-Level-Prediction-from-video-repo/video_image_manipulation/frames/9.2_00372925'
def read_data(img_dir):
    img_data = []
    frame_list = []
    for frame_info in os.listdir(img_dir):
        frame_dir = os.path.join(img_dir, frame_info)
        if os.path.isdir(frame_dir):
            frame_list.append(int(frame_info))
    frame_list.sort()

    for frame_num in frame_list:
        imgs = []
        dir = os.path.join(img_dir, str(frame_num))
        for region in region_list:
            img = cv2.imread(os.path.join(dir, region))
            if img.shape != (224, 224, 3):
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
            imgs.append(img)
        img_data.append(imgs)
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

if __name__ == '__main__':
    img_data = read_data(img_dir)
    label = get_label(img_dir)
    unified_data = unify_sequence_length(img_data, standard_length)

    print (torch.from_numpy(unified_data[:,0]).shape)