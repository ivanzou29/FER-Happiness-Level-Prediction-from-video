#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python vgg_reduced_face_two_eyes_mouth_model_RMSE.py -b 10 -e 30;

CUDA_VISIBLE_DEVICES=6 python vgg_reduced_face_two_eyes_mouth_model_RMSE.py -b 16 -e 30;

CUDA_VISIBLE_DEVICES=6 python vgg_reduced_face_two_eyes_mouth_model_RMSE.py -b 20 -e 30;
