import cv2
import numpy as np
import tensorflow as tf
import os
import random
import cmapy
import math

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

path = "prediction/a (64).tiff"

# save predicted image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt_img = plt.imread(path, 0)
cv2.imshow("test", plt_img)
cv2.waitKey(0)

randint = random.randint(100000, 999999)

if (os.path.isfile(f"savedPNGS/{randint}.png") is False):
    plt.imsave(f"savedPNGS/{randint}.png", plt_img)
    plt.imshow(plt_img, cmap='gray', interpolation='bicubic')

# load model
model = load_model("model/electronDetectAi.h5")


def makePredcition():
    p_img = cv2.imread(f"savedPNGS/{randint}.png")
    resized = cv2.resize(p_img, (982, 982), interpolation=cv2.INTER_AREA)
    x = image.img_to_array(resized)
    x = np.expand_dims(x, axis=0)

    stacked = np.vstack([x])

    def predict(image):
        pred_val = model.predict(image)
        if (pred_val[0][np.argmax(pred_val)] <= 0.3):
            print("no match")
            print(pred_val)
        else:
            print(f"{np.argmax(pred_val) + 1} <- cislo suboru s najväcsim % | hodnoty pre vsetky subory -> {pred_val}")

    predict(stacked)

    plt_img = mpimg.imread(path)
    plt.imshow(plt_img)


def fitEllipse():
    from_png = cv2.imread(f"savedPNGS/{randint}.png")

    gray_image = cv2.cvtColor(from_png, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(from_png, cv2.COLOR_BGR2HSV)

    ret,thresh = cv2.threshold(gray_image,127,255,0)

    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    print(cX, cY)

    cv2.circle(gray_image, (cX, cY), 5, (0, 255, 0), -1)
    cv2.putText(gray_image, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # gray scale mask

    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    contours= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(big_contour)
    (xc, yc), (d1, d2), angle = ellipse
    print(xc, yc, d1, d1, angle)

    cv2.ellipse(from_png, ellipse, (255, 0, 0), 2)

    # display the image
    cv2.imshow("Image", from_png)
    cv2.waitKey(0)
    os.remove(os.path.join(f"savedPNGS/{randint}.png"))

# fit grid cut off
def fitE_gridCutOff():
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
    print(xc, yc, d1, d1, angle)

    # draw ellipse
    result = img.copy()
    cv2.ellipse(thresh, ellipse, (0, 255, 0), 3)

    xc, yc = ellipse[0]

    # draw vertical line
    # compute major radius
    rmajor = max(d1, d2) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    print(angle)
    xtop = xc + math.cos(math.radians(angle)) * rmajor
    ytop = yc + math.sin(math.radians(angle)) * rmajor
    xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
    ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
    cv2.line(result, (int(xtop), int(ytop)), (int(xbot), int(ybot)), (0, 0, 255), 3)

    cv2.imshow("labrador_thresh", thresh)
    cv2.waitKey(0)
    os.remove(os.path.join(f"savedPNGS/{randint}.png"))
    cv2.destroyAllWindows()



if __name__ == "__main__":
    makePredcition()
    fitE_gridCutOff()