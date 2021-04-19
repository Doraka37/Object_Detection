import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image "starry_night.jpg" using the default flag "cv2.IMREAD_COLOR"
# (i.e. discarding any alpha/transparency channel)
img1 = cv.imread("./cluttered.jpeg", cv.IMREAD_COLOR)
# Could also use: img1 = cv.imread("data/starry_night.jpg")

if img1 is None:
    print("Could not read the image.")
else:
    print(f"img1.shape: {img1.shape}")
    print(f"img1.dtype: {img1.dtype}")
    print(f"img1.min(): {img1.min()}")
    print(f"img1.max(): {img1.max()}\n")

img2 = img1[380:860, 500:750]


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()
