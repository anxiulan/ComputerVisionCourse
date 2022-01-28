import cv2
import numpy as np


img = cv2.imread('../../dataset/hua.jpg')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ****************** SIFT ******************************
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp, des = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kp, None)
cv2.imshow("sift_keypoint", img)

# ****************** SURF(需要xfeatures2d) ******************************
# surf = cv2.xfeatures2d.SURF_create(400) # 400: Hessian阈值
# kp, des = surf.detectAndCompute(gray, None) # keypoints, descriptor
# img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
#
# cv2.imshow("sift_keypoint", img)
# cv2.waitKey(0)
# cv2.des
# troyAllWindows()


# ****************** FAST ******************************
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img, None)
img = cv2.drawKeypoints(gray, kp, None)
cv2.imshow("fast", img)



# ****************** ORB ******************************
orb = cv2.ORB_create()
kp = orb.detect(img, None)
img = cv2.drawKeypoints(gray, kp, None)
cv2.imshow("ORB", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

