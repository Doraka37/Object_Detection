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
resized_image = imutils.resize(image, width=min(400, image.shape[1]))

# detect people in the image
start = datetime.datetime.now()
(rects, weights) = hog.detectMultiScale(resized_image)
print("[INFO] detection took: {}s".format(
	(datetime.datetime.now() - start).total_seconds()))
# draw the original bounding boxes
for (x, y, w, h) in rects:
	resized_x = (x / resized_image.shape[1]) * image.shape[1]
	resized_y = (y / resized_image.shape[0]) * image.shape[0]
	resized_w = (w / resized_image.shape[1]) * image.shape[1]
	resized_h = (h / resized_image.shape[0]) * image.shape[0]
	cv2.rectangle(image, (int(resized_x), int(resized_y)), (int(resized_x) + int(resized_w), int(resized_y) + int(resized_h)), (0, 255, 0), 2)

number_of_people = len(rects)

cv2.putText(image, f"{number_of_people} people detected", (5, image.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.show()
# show the output image
cv2.imshow("Detections", image)
cv2.waitKey(0)
