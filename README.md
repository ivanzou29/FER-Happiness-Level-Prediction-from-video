# Happiness Prediction based on Videos of Facial Expressions

This is an undergraduate final year project from the University of Hong Kong. It defines the happiness level from 0 to 10. The video-based happiness prediction problem is treated as a regression task. It focuses on the individual-level happiness prediction from facial expression.

## Contents

### models

The directory contains the training scripts for different models.

### video image manipulation

The directory includes the scripts for video data pre-processing.

### experiments original dataset

The directory contains the training results of the models on the original dataset which has a Gaussian-like distribution of the histogram.

### experiments sampled dataset

The directory contains the training results of the models on the sampled dataset which is more evenly distributed.

For details and queries, please contact ivanzou29@gmail.com


## References:

### Dataset of Videos: 

Dhall, A., Goecke, R., Lucey, S., Gedeon, T. (2012). Collecting Large Richly Annotated Facial-Expression Databases from Movies. In IEEE Multimedia 2012. 

If you are interested in the dataset that is labeled with happiness level, please obtain the license agreement specified in https://cs.anu.edu.au/few/AFEW.html, and then contact ivanzou29@gmail.com.

### Program References:

#### construct network with PyTorch 
https://hellozhaozheng.github.io/z_post/PyTorch-VGG/

#### usage of dlib for face and region detections
https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/


