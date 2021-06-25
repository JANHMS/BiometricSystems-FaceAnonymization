import os
import cv2
import sys
import numpy as np
import detect

from matplotlib import pyplot as plt
# We can choose which type of recognition we want to use.

# Normally we would use the full face uncomment the following line
DETECTION_METHOD_PATH = "/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/haarcascades/haarcascade_frontalface_default.xml"

# This is needed for the mask anonymization, when we just want to black the eyes and not the hole face
# DETECTION_METHOD_PATH = "/Users/janheimes/Main/DTU/02238_Biometric_systems/Face-Anonymization/haarcascades/haarcascade_eye.xml"

# This is needed for the mask anonymization, when we just want to black the eyes and not the hole face
# The only eye had problems detecting eyes for glasses and also detected sometimes the holes of the nose as eyes
# DETECTION_METHOD_PATH = "/Users/janheimes/Main/DTU/02238_Biometric_systems/Face-Anonymization/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

IMAGE_PATH = "/Users/janheimes/Main/DTU/02238_Biometric_systems/Recognition_Anonymization_Face/FERET/reference/"

# Apply to ~whole database which is 785 pictures, 750 choosen because cleaner number
IMG_NUM = 750
faceCascade = cv2.CascadeClassifier(DETECTION_METHOD_PATH)

def anonymise(method:'str', k):

    for i in range(1, IMG_NUM):
        try:
            image = cv2.imread(IMAGE_PATH + str(i) + ".jpg")
            # print(image)
            # test for work with a single image
            # image = cv2.imread('/Users/janheimes/Main/DTU/02238_Biometric_systems/downloaded_files/FERET/probe/300.jpg')
            
            # print(f"This is the image in the for loop: {image}")
            faces = detect_face(image)
            for (x, y, w, h) in faces:
                if method == "blur":
                    image = blur(image,x,y,w,h, k=k)
                    # print(f"Changed blur {image}")

                elif method == "pixelate":
                    image = pixelate(image,x,y,w,h, scale=101-k)
                    # print(f"Changed pixelate {image}")
            
                # image_d = mask(image,x,y,w,h)
                elif method == 'noise':
                    image = noise(image, x, y, w, h, threshold=1 - k/100)
                    # print(f"Changed noise {image}")
                    
                elif method == 'mask':
                    image = mask(image, x, y, w, h)
                    # print(f"Changed mask {image}")

            path = '/Users/janheimes/Main/DTU/02238_Biometric_systems/results/' + method + '/' + str(k)
            try:
                os.makedirs(path)
                print("try")
            except:
                print("except")
            
            path = '/Users/janheimes/Main/DTU/02238_Biometric_systems/results/'+ method + '/' + str(k) + '/' + str(i) + '.jpg'
            print(f"done/{i}")
            cv2.imwrite(path, image)
        except:
            print(f'No image path for{i}')

def blur(img, x, y, w, h, k=350):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    img[startY:endY, startX:endX] = cv2.blur(img[startY:endY, startX:endX], (k, k))
    print("blurring..")
    return img


def pixelate(img, x, y, w, h, scale=16):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    # Resize input to size needed for pixelaxe
    temp = cv2.resize(img[startY:endY, startX:endX], (scale, scale),
                      interpolation=cv2.INTER_LINEAR)
                      
    img[startY:endY, startX:endX] = cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST)
    return img


def mask(img, x, y, w, h):
    startX = x
    endX = x + w
    startY = y
    endY = y + h

    img[startY:endY, startX:endX] = 0
    print("black..")
    return img


def noise(img, x, y, w, h, threshold=0.6):
    startX = x
    endX = x + w
    startY = y
    endY = y + h
    
    random_mask = np.random.randint(low=0, high=255, size=(h, w, 3))
    random_map = np.random.random((h, w, 1)).astype(np.float16)

    temp = np.asarray(img[startY:endY, startX:endX], dtype="int32")
    temp = np.where(random_map < threshold, temp, random_mask)

    img[startY:endY, startX:endX] = temp
    return img


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(f"This is the current img {img}")
    return faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=10,
                                        minSize=(40, 40)
                                        )
    print("succsess")                                
