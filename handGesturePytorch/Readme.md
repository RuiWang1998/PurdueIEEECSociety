Learning process

![Grid](LearningProcess.png)

# Files:

In `constants.py` you will find the constants I defined for this projects as well as hyperparameters

In `Model.py` I have the models designed for gesture recognition

In 'imageProcessing.py' I have the functions defined for the preprocessing of pictures

## One-shot learning
In `handID.py` I have the training for one shot learning model

In `HandIDFunctions.py` I have the functions of training and testing for the one shot learning model

In `couples.py` you will find the functions for creating couples for the one shot learning process

`input_test.py` creates the new mean for the new gesture

`new_images.py` asks for new images with the desktop/laptop webcam

`predict.py` does the prediction for the new picture and indicates the confidence

`visual.py` performs visualization for one-shot-learning process

## vanilla classification
In `main.py` you will find the training and testing for the vanilla classification models.

In `dataloader.py` I have the functions to load the images for the vanilla classification.

`testTrain.py` is the functions for vanilla training, to be merged with `HandIDFunctions.py`

