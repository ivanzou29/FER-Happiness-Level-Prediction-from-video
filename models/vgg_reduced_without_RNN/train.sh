#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 40 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 32 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 20 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 16 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 0

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 40 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 32 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 20 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 0
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 16 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 0

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 40 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 32 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 20 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 16 -e 100 -o 20 -d 0.1 -t 16 -l 0.000001 -s 1

CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 40 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 32 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 20 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 1
CUDA_VISIBLE_DEVICES=3 python vgg_reduced_model_without_RNN.py -b 16 -e 100 -o 10 -d 0.1 -t 16 -l 0.000001 -s 1
