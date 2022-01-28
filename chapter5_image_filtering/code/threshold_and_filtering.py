import cv2
import numpy as np

# 阈值相关操作
img = cv2.imread('../../dataset/hua.jpg', 1)
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1', thresh1)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('2', thresh2)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('3', thresh3)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('4', thresh4)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('5', thresh5)
cv2.waitKey(0)

cv2.destroyAllWindows()

# 滤波操作
# 1.均值滤波
img = cv2.imread('../../dataset/hua.jpg', 1)
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
blur_img = cv2.blur(img, (10, 10))

# 2.高斯滤波
Gaussian_blur_img = cv2.GaussianBlur(img, (15, 15), 10)  # 核的shape一定是（奇数，奇数）

# 3.中值滤波
median_blur_img = cv2.medianBlur(img, 25)

# stack（）：沿着新的轴加入一系列数组。
# vstack（）：堆栈数组垂直顺序（行）
# hstack（）：堆栈数组水平顺序（列）
vstack_img = np.vstack((blur_img, Gaussian_blur_img, median_blur_img))
vstack_img = cv2.resize(vstack_img, (0, 0), None, 0.5, 0.5)
cv2.imshow('img', img)
cv2.imshow('blurred', vstack_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
