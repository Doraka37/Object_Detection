import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("./cluttered.jpeg", cv2.IMREAD_GRAYSCALE)

if img1 is None:
    print("Could not read the image.")
else:
    print(f"img1.shape: {img1.shape}")
    print(f"img1.dtype: {img1.dtype}")
    print(f"img1.min(): {img1.min()}")
    print(f"img1.max(): {img1.max()}\n")

img2 = img1[380:860, 500:750]

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
