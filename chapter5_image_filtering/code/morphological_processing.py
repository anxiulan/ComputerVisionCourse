import cv2
import numpy as np

# erode 腐蚀
kernel = np.ones((5, 5), np.uint8)
img = cv2.imread('../../dataset/hua.jpg', 1)
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
cv2.imshow('hua', img)
cv2.waitKey(0)

# erode 腐蚀
erosion = cv2.erode(img, kernel, iterations=3)
cv2.imshow('erosion', erosion)
cv2.waitKey(0)

# dilate 膨胀
dilate = cv2.dilate(erosion, kernel, iterations=3)
cv2.imshow('dilate', dilate)
cv2.waitKey(0)

# 开运算：先腐蚀在膨胀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
cv2.waitKey(0)

# 闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)
cv2.waitKey(0)

# 梯度运算
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)

# 礼帽：原始输入-开运算，得到毛刺
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('tophat', tophat)
cv2.waitKey(0)

# 黑帽：闭运算-原始输入，得到原始轮廓点
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('blackhat', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
