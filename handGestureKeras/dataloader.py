from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as kbackend
import platform
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, resolution, input_channel
from coupleMaker import create_couple, create_wrong
from matplotlib import pyplot as plt
import numpy as np

train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=[0.7, 1.3],
        horizontal_flip=True,
        channel_shift_range = 0.4,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

def loadData(batch_size = BATCH_SIZE, down_scaling_factor =  DOWNSCALING_FACTOR, source_dir = SOURCE):

    row = int(resolution[0] * down_scaling_factor)
    col = int(resolution[1] * down_scaling_factor)
    
    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            directory=source_dir + TRAIN_FOLDER,  # this is the target directory
            target_size=(row, col),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='categorical',
            color_mode="rgb",
            shuffle=True)  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            directory=source_dir + TEST_FOLDER,
            target_size=(row, col),
            batch_size=1,
            color_mode="rgb",
            class_mode='categorical')

    all_generator = test_datagen.flow_from_directory(
            directory=source_dir + ALL_FOLDER,
            target_size=(row, col),
            batch_size=1,
            color_mode="rgb",
            class_mode='categorical')

    return train_generator, validation_generator, all_generator

def input_shape(downscaling_factor = DOWNSCALING_FACTOR):

    row = int(resolution[0] * downscaling_factor)
    col = int(resolution[1] * downscaling_factor)

    # Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'
    if kbackend.image_data_format() == 'channels_first':
        input_shape = (input_channel, row, col)
    else:
        input_shape = (row, col, input_channel)

    return input_shape

input_shapes = input_shape()

def generator(batch_size, train_folder):
    '''
    from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb
    '''
  
    while 1:
        X=[]
        y=[]
        switch=True
        for _ in range(batch_size):
        #   switch += 1
            if switch:
                #   print("correct")
                X.append(create_couple(train_folder).reshape((2, input_shapes[0], input_shapes[1], input_shapes[2])))
                y.append(np.array([0.]))
            else:
                #   print("wrong")
                X.append(create_wrong(train_folder).reshape((2, input_shapes[0], input_shapes[1], input_shapes[2])))
                y.append(np.array([1.]))
            switch=not switch
        X = np.asarray(X)
        y = np.asarray(y)
        #XX1=X[0,:]
        #XX2=X[1,:]
        yield [X[:,0],X[:,1]],y

def val_generator(batch_size, test_folder):
    '''
    from https://github.com/normandipalo/faceID_beta/blob/master/faceid_beta.ipynb
    '''
    while 1:
        X=[]
        y=[]
        switch=True
        for _ in range(batch_size):
            if switch:
                X.append(create_couple(test_folder).reshape((2, input_shapes[0], input_shapes[1], input_shapes[2])))
                y.append(np.array([0.]))
            else:
                X.append(create_wrong(test_folder).reshape((2, input_shapes[0], input_shapes[1], input_shapes[2])))
                y.append(np.array([1.]))
            switch=not switch
        X = np.asarray(X)
        y = np.asarray(y)
        #XX1=X[0,:]
        #XX2=X[1,:]
        yield [X[:,0],X[:,1]],y

def create_input(file_path):
  #  print(folder)
    mat=np.zeros((480,640), dtype='float32')
    i=0
    j=0
    depth_file = file_path
    with open(depth_file) as file:
        for line in file:
            vals = line.split('\t')
            for val in vals:
                if val == "\n": continue    
                if int(val) > 1200 or int(val) == -1: val= 1200
                mat[i][j]=float(int(val))
                j+=1
                j=j%640

            i+=1
        mat = np.asarray(mat)
    mat_small=mat[140:340,220:420]
    img = Image.open(depth_file[:-5] + "c.bmp")
    img.thumbnail((640,480))
    img = np.asarray(img)
    img = img[140:340,220:420]
    mat_small=(mat_small-np.mean(mat_small))/np.max(mat_small)
    plt.figure(figsize=(8,8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mat_small)
    plt.show()
    plt.figure(figsize=(8,8))
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
    
    
    
    full1 = np.zeros((200,200,4))
    full1[:,:,:3] = img[:,:,:3]
    full1[:,:,3] = mat_small
    
    return np.array([full1])