import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image "starry_night.jpg" using the default flag "cv2.IMREAD_COLOR"
# (i.e. discarding any alpha/transparency channel)
img1 = cv.imread("./cluttered.jpeg", cv.IMREAD_COLOR)
# Could also use: img1 = cv.imread("data/starry_night.jpg")

if img1 is None:
    print("Could not read the image.")

img2 = img1[380:860, 500:750]


# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
