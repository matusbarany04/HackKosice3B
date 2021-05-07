import cv2
import numpy as np
import tensorflow as tf
import os
import random
import cmapy
import csv
import time

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load model
model = load_model("model/electronDetectAi.h5")

randint = random.randint(100000, 999999)

# make prediction for image to recognize its category
def makePredcition(path):
    p_img = cv2.imread(f"savedPNGS/{randint}.png")
    resized = cv2.resize(p_img, (982, 982), interpolation=cv2.INTER_AREA)
    x = image.img_to_array(resized)
    x = np.expand_dims(x, axis=0)

    stacked = np.vstack([x])

    def predict(image):
        pred_val = model.predict(image)
        if (pred_val[0][np.argmax(pred_val)] <= 0.3):
            print("no match")
        else:
            return np.argmax(pred_val)

    max_val = predict(stacked)

    plt_img = mpimg.imread(path)
    plt.imshow(plt_img)

    return max_val


# simple fitting algorythm
def fitEllipseSimple(path):

    # open png for fitting
    from_png = cv2.imread(f"savedPNGS/{randint}.png")

    start_time = time.time()

    img_name = path
    img_name = img_name.split("/")
    img_name = img_name[-1]

    gray_image = cv2.cvtColor(from_png, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray_image,127,255,0)

    M = cv2.moments(thresh)

    # get center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # gray scale mask

    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(big_contour)
    (xc, yc), (d1, d2), angle = ellipse

    cv2.ellipse(from_png, ellipse, (255, 0, 0), 2)

    # display the image
    cv2.imshow("Image", from_png)

    elapsed_time = (time.time() - start_time) * 1000

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name, xc, yc, d1, d2, angle, elapsed_time])

    cv2.waitKey(0)
    os.remove(os.path.join(f"savedPNGS/{randint}.png"))

# fit more complex ellipsies
def fitEllipseComplex(path):
    png = cv2.imread(f"savedPNGS/{randint}.png")
    start_time = time.time()

    img_name = path
    img_name = img_name.split("/")
    img_name = img_name[-1]

    first_img = cv2.imread(path)
    img = cv2.applyColorMap(first_img, cmapy.cmap('flag_r'))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(big_contour)
    (xc, yc), (d1, d2), angle = ellipse

    # draw ellipse
    result = img.copy()
    cv2.ellipse(png, ellipse, (255, 0, 255), 3)

    xc, yc = ellipse[0]


    cv2.imshow("final", png)
    elapsed_time = (time.time() - start_time) * 1000

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name, xc, yc, d1, d2, angle, elapsed_time])


    os.remove(os.path.join(f"savedPNGS/{randint}.png"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(path):
    # write png for fitting
    plt_img = plt.imread(path, 0)

    if (os.path.exists("savedPNGS") is False):
        os.mkdir("savedPNGS")
        if (os.path.isfile(f"savedPNGS/{randint}.png") is False):
            plt.imsave(f"savedPNGS/{randint}.png", plt_img)
    else:
        if (os.path.isfile(f"savedPNGS/{randint}.png") is False):
            plt.imsave(f"savedPNGS/{randint}.png", plt_img)

    res = makePredcition(path)
    pred_val = res+1

    if (pred_val == 2 or pred_val == 1):
        try:
            fitEllipseSimple(path)
        except:
            try:
                fitEllipseComplex(path)
            except:
                with open('output.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["couldn't fit ellipse in the image"])

    elif (pred_val == 3 or pred_val == 4 or pred_val == 5 or pred_val == 6 or pred_val == 7):
        try:
            fitEllipseComplex(path)
        except:
            try:
                fitEllipseSimple(path)
            except:
                with open('output.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["couldn't fit ellipse in the image"])
    else:
        try:
            fitEllipseComplex(path)
        except:
            with open('output.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["couldn't fit ellipse in the image"])




main("pewdicrion/ilumination_2/g (6).tiff")