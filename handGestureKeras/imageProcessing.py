import matplotlib.image as img
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.preprocessing import normalize
from skimage import color
from skimage.transform import rescale, resize
import time
import random
import warnings
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range = 0.4,
        fill_mode='nearest')


def image_augmentation_save(img, ParentDir, Prefix):
    # the input should be a Numpy array
    img = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(img, batch_size=1,
                             save_to_dir=ParentDir, save_prefix=Prefix, save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


# this function takes the file name and read it
def load_images_downscale(folder_name, factor):

    images = []
    for filename in glob.glob(folder_name + '/*.jpg'):
        images.append(process_image(img.imread(filename), factor = factor))

    return images

def load_images(folder_name):

    images = []
    for filename in glob.glob(folder_name + '/*.jpg'):
        images.append(color.rgb2gray(img.imread(filename)))

    return images

def process_image(image, action = "downscale", factor = 0.5):
    # this function provide three options manipulating images
    # this function also performs normalization
    
    # depending on the input this function will be able to make changes to the pictures
    if action == "rescale":
        image = rescale(image, 1.0 / factor, anti_aliasing=False)
    if action == "resize":
        image= resize(image, (image.shape[0] / factors, image.shape[1] / factor), anti_aliasing=True)
    if action == "downscale":
        image = rescale(image, factor)
        
    # this line will be able to shift the image to gray scale since the color is believed to be not useful enough to cause two times more calculations
    #image = color.rgb2gray(image)

    return image

def show_image(img):

    # this will print out the dimension of the image
    print("The dimension of the image being displayed is " + str(img.shape))

    # this will display the image in gray scale
    plt.imshow(img, cmap = 'gray')
    plt.show()

    return

def preprocessing(parent_folder_name, source_dir, train_folder, test_folder, factor, prob = 0.8, augmentation = True):
    images = []
    print("Start loading the images")
    start = time.time()
    for folderName in glob.glob(source_dir + parent_folder_name + '/*'):
        if folderName != 'image_folder/desktop.ini' and folderName != 'image_folder\desktop.ini':
            #print("Start loading the folder: " + folderName)
            images.append(load_images_downscale(folderName, factor = factor))

    print("The time used for loading image was {0:.2f} seconds.".format(time.time() - start))
    start = time.time()

    test_size = 0
    count_train = 20
    l = 0
    for i in images:
        k = 0
        for j in i:
            if random.uniform(0, 1) > prob:
                test_size += 1
                folder = test_folder
            else:
                folder = train_folder
            
            loca = source_dir + folder + "/" + str(l) + "/" + str(k) + ".jpg"
            # some data (512x512)
            data = j
            # a colormap and a normalization instance
            cmap = plt.cm.gray
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = norm(data)
            image_augmentation_save(image, source_dir+folder+"/"+str(l)+"/", 'aug')
            k += 1
        l += 1
    print("The time used for writing image was {0:.2f} seconds.".format(time.time() - start))

    return test_size

def normalize_images(images):
    normalized_images = []
    for i in images:
        new_images = []
        for image in i:
            # here we will normalize the data
            orig_height = image.shape[0]
            image = image.reshape(1, -1)
            image = normalize(image)
            # print("The normalized average is " + str(image.mean()))       # this line is used to make sure the normalization worked
            image = image.reshape(orig_height, -1)
            new_images.append(image)
        normalized_images.append(new_images)

    return normalized_images
