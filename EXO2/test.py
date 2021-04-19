# import the necessary packages
import numpy as np
import cv2
import imutils
import datetime
import argparse
from matplotlib import pyplot as plt


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# load the image and resize it
image = cv2.imread("./macron.jpg", cv2.IMREAD_COLOR)
image = imutils.resize(image, width=min(400, image.shape[1]))

# detect people in the image
start = datetime.datetime.now()
(rects, weights) = hog.detectMultiScale(image)
print("[INFO] detection took: {}s".format(
	(datetime.datetime.now() - start).total_seconds()))
# draw the original bounding boxes
for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(image,),plt.show()
# show the output image
#cv2.imshow("Detections", image)
#cv2.waitKey(0)
