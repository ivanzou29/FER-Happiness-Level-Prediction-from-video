#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 16 -e 500 -o 20 -d 0.1 -t 8 -l 0.000005 -s 1 -r 2
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 16 -e 550 -o 20 -d 0.1 -t 8 -l 0.000005 -s 1 -r 2

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 20 -e 500 -o 20 -d 0.1 -t 8 -l 0.000005 -s 1 -r 2
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 20 -e 550 -o 20 -d 0.1 -t 8 -l 0.000005 -s 1 -r 2

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 16 -e 500 -o 20 -d 0.1 -t 8 -l 0.000005 -s 0 -r 2
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 16 -e 550 -o 20 -d 0.1 -t 8 -l 0.000005 -s 0 -r 2

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 20 -e 500 -o 20 -d 0.1 -t 8 -l 0.000005 -s 0 -r 2
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_with_RNN_face_two_eyes.py -b 20 -e 550 -o 20 -d 0.1 -t 8 -l 0.000005 -s 0 -r 2
