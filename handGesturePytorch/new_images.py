import cv2
import time
import platform

from constants import DATA_SOURCE, SOURCE, VIS_FOLDER

IS_MAC = (platform.system() == 'Darwin')

def get_images(numSamples: int, secsBetweenSamples: float, name: str):

    # On my laptop
    # 0 front camera
    # 1 back camera
    camera = 0
    cap = cv2.VideoCapture(camera)

    # Capture X number of images
    for x in range(numSamples):
        # take image
        ret, frame = cap.read()

        # display image
        cv2.imshow('frame', frame)

        #   Mac default camera is 1280 by 720 (16:9)
        if (IS_MAC):
            frame = frame[:, 160:1120, :]   #   Extract the middle column of the image (go from 16:9 resolution to 4:3 resolution)

        #   Ensure common resolution
        frame = cv2.resize(frame, (640, 480))

        # write image to file
        if (not IS_MAC or x):   #   The first image doesn't save on mac
            out = cv2.imwrite(SOURCE+DATA_SOURCE+VIS_FOLDER+'/' + name + 'capture{}.jpg'.format(x), frame)
        cv2.waitKey(1)

        time.sleep(secsBetweenSamples)

    cap.release()

def getNew(size):
    time_step = time.time()
    k = 3
    while time.time() - time_step <= 3:
        print("get ready in {} for your right hand".format(k))
        k -= 1
        time.sleep(1)  

    get_images(80, 0.1, 'right')

    time.sleep(6) 
    time_step = time.time()
    k = 5
    while time.time() - time_step <= 5:
        print("get ready in {} for your left hand".format(k))
        k -= 1
        time.sleep(1)

    get_images(80, 0.1, 'left')

if __name__ == '__main__':
    # getNew(80)
    get_images(5, 0.1, 'test')

