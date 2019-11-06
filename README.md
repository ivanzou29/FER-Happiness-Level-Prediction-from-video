# Facial Expression Recognition: Happiness Level Prediction from Video

It is an undergraduate final year project about happiness level prediction based on video. It uses computer vision techniques and neural networks to perform happiness level prediction. The following includes the proposal at an early stage.

### 1.Problem Definition

Facial Expression Recognition (FER) has become a topic that attracts a lot of attention from academic researchers. Nowadays, there has already been a certain amount of research with regard to emotion recognition classification, the aim of which is to tell the type of emotion such as “happy”, “sad” or “angry” from an image or a sequence of image (or a video clip). However, few researchers have investigated on the prediction for happiness level from video clips. The prediction for happiness level could be another interesting topic to research on, as sometimes we might only be interested in how happy or how satisfied a person is based on his or her expression, instead of knowing exactly what kind of expression it is. For example, in some mobile applications we may want the users to rate a certain type of service such as a restaurant. When the user is asked to rate the restaurant, it might be difficult for him or her to express the true degree of happiness (or satisfaction) by solely providing the number of “stars” for the rating. The discrete value provided by the number of “stars” could also be restricted. However, it might help analyze the happiness level of the user based on his or her facial expression. If the user is asked to input a video selfie for only 2 to 3 seconds, that will be enough for a model to give the happiness level based on that tiny video clip of facial expression. It could help some rating systems gain a unified-standard (where the standard is defined by the model) analysis from the perspective of happiness level of the users. Simultaneously, it may also help reduce the occurrence of some phenomena where the users are bribed by the service provider when rating the service. In the abovementioned case of happiness level prediction, what we need here is a quantified result rather than an encoded class. The corresponding problem to happiness level prediction is then switched from a classification problem to a regression problem.

### 2.Related Works

Previous researchers have paved way for some basic methodologies on how to deal with facial expression recognition classification based on images [3] or videos [4][5][7][8], which could be illuminating for the problem of happiness level prediction. The convolutional neural network (CNN) methodology used in those implementations is the fundamental component of one solution. Some of the video-based research has also adopted a combination of CNN and recurrent neural network (RNN), which serves as a tool for analysis of progression of frames within a video. Also, there has been one work [6] used 3D CNN to analyze the temporal variation between image frames in a video. Besides, there has also been work related to individual depression level prediction [1], for which the task is similar to what I am going to work on, which is a regression task. It also adopts RNN as a major tool. The performance of that model is measured by root mean squared error (RMSE) and mean absolute error (MAE) instead of a confusion matrix, as it focused on a regression problem. However, there is currently no dataset labelled with “happiness level” or “happiness scale”, which could be a challenge to the project.

### 3.Proposed Methodology

In this section, the design diagram of a proposed model on how to tackle the problem of happiness level prediction would be provided. Besides, some issues about the usage of datasets will be discussed.

#### 3.1. Model

Inspired by those abovementioned previous papers, to tackle the “happiness level problem”, I have proposed a model that combines a multi-region ensemble CNN[3 with an RNN (with Long-Short Term Memory units, which can preserve historical input well), which pairs several key regions of the face (eye, nose, mouth) with the whole face appearance and observes the progression of the image frames in the video. In the meantime, several extensions of this model will be tried during the experiment, such as a combination of 3D CNN (which also convolves along the temporal domain) and RNN. As this project focuses on facial expression and happiness level, the audio part would not be taken into consideration at the very early stage. 

#### 3.2. Dataset

As there has been no available dataset labeled with “happiness level” so far and the development of the application for collecting the data is still on the air, the model will be built and tried with several database labeled with certain type of emotion. For example, the Acted Facial Expressions in The Wild (AFEW) dataset [2] for EmotiW2018 Challenge could be used as experimental dataset. It contains a number of videos of facial expressions labeled with emotion (happy, sad, angry, surprised, neutral, fear, disgust). At this stage, I am going to experiment the data by transforming the labels into several values for happiness level with a range of 1 to 5 (for example: surprised to 5, happy to 4, neutral to 3, sad to 2, angry to 1. video labeled with “fear” and “disgust” will be ignored). Frame extraction will be performed prior to the training of the model, since there might be too many duplicate frames in the whole dataset and that will make the overall algorithm more computationally expensive. The frame extraction could be achieved by calculating the finite-difference between images after applying certain smoothing filters (similar to “corner detection” in 3 dimensions). In the meantime, another dataset called Cohn-Kanade dataset (CK+) [6] of image sequence (with only the changed frames picked) is also obtained and will be another experimental dataset. The label transformation will be performed in a similar manner as mentioned before. As in CK+ dataset the image sequence of video frames is already provided, there is no need for frame extraction.

### 4.Evaluation Plan

The performance of the model will be primarily evaluated based on RMSE. RMSE has a low tolerance against large errors. It preserves a stricter standard for the model. The model and its different extensions of the model proposed above will be evaluated on several different datasets with their label transformed into corresponding figures. Then, the different RMSEs are compared to select the model that performs the relatively better. A baseline for RMSE will be set according to the performance, which will be used to evaluate the model with the dataset labeled with “happiness level” later.

After the dataset labeled with happiness level is prepared, the model and extensions of the model will also be trained with that dataset. The models that performs relatively better with the previous datasets will be trained and evaluated first. The minimum requirement for the performance is to at least pass the baseline.

### References

1.	Chao, L., Tao, J., Yang, M. & Li, Y. (2015). Multi-Task Sequence Learning for Depression Scale Prediction from Video. 2015 International Conference on Affective Computing and Intelligent Interaction (ACII): 527-531.

2.	Dhall, A., Goecke, R., Lucey, S., Gedeon, T. (2012). Collecting Large Richly Annotated Facial-Expression Databases from Movies, IEEE Multimedia 2012.

3.	Fan, Y., Lam, J.C.K., Li, V.O.K. (2018). Multi-Region Ensemble Convolutional Neural Network for Facial Expression Recognition. ICANN 2018.

4.	Fan, Y., Lam, J.C.K., Li, V.O.K. (2018). Video-based Emotion Recognition Using Deeply-Supervised Neural Networks. 20th ACM International Conference on Multimodal Interaction.

5.	Fan, Y., Lu, X., Li, D., Liu, Y. (2016). Video-based Emotion Recognition Using CNN-RNN and C3D Hybrid Networks. Retrieved from: https://www.researchgate.net/publication/308453418_Video-based_emotion_recognition_using_CNN-RNN_and_C3D_hybrid_networks.

6.	Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.

7.	Liu, C., Tang, T., Lv, K., Wang, M. (2018). Multi-Feature Based Emotion Recognition for Video Clips. ICMI’18: 630-634.

8.	Vielzeuf, V., Pateux, S., Jurie, F. (2017). Temporal Multimodal Fusion for Video Emotion Classification in the Wild. 





