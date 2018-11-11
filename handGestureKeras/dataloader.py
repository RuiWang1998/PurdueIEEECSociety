from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as kbackend
import platform
from constants import DOWNSCALING_FACTOR, TRAIN_FOLDER, TEST_FOLDER, ALL_FOLDER, PARENT_FOLDER_NAME, SOURCE, EPOCHS, BATCH_SIZE, learning_rate, NUM_CLASS, DATA_SOURCE, resolution, input_channel

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range = 0.4,
        fill_mode='nearest')

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
            batch_size=batch_size,
            color_mode="rgb",
            class_mode='categorical')

    all_generator = test_datagen.flow_from_directory(
            directory=source_dir + ALL_FOLDER,
            target_size=(row, col),
            batch_size=batch_size,
            color_mode="rgb",
            class_mode='categorical')

    return train_generator, validation_generator, all_generator

def input_shape(train_data, test_data, downscaling_factor = DOWNSCALING_FACTOR):

    row = int(resolution[0] * downscaling_factor)
    col = int(resolution[1] * downscaling_factor)

    # Keras can work with datasets that have their channels as the first dimension ('channels_first') or 'channels_last'
    if kbackend.image_data_format() == 'channels_first':
        input_shape = (input_channel, row, col)
    else:
        input_shape = (row, col, input_channel)

    return input_shape