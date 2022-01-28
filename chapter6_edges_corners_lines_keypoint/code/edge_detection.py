import cv2
import numpy as np

# 实例1
kernel = np.ones((5, 5), np.uint8)
img = cv2.imread('../../dataset/star.png', 1)
cv2.imshow('star', img)

sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel)
cv2.imshow('sobel_x', sobel_x)

sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel)
cv2.imshow('sobel_y', sobel_y)

# x与y方向的梯度相加
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
cv2.imshow('sobel_xy', sobel_xy)

# 不建议直接令x,y=1
sobel_x1y1 = cv2.Sobel(img, cv2.CV_64F, 1, 1)
cv2.imshow('sobel_x1y1', sobel_x1y1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 实例2
hua = cv2.imread('../../dataset/hua.jpg', 0)
hua = cv2.resize(hua, (0, 0), None, 0.5, 0.5)
cv2.imshow('hua', hua)

hua_x = cv2.Sobel(hua, -1, 0, 1, ksize=3)
hua_y = cv2.Sobel(hua, -1, 1, 0, ksize=3)
hua_x = cv2.convertScaleAbs(hua_x)
hua_y = cv2.convertScaleAbs(hua_y)
hua_xy = cv2.addWeighted(hua_x, 0.5, hua_y, 0.5, 0)
cv2.imshow('hua_xy', hua_xy)

# 用CV_64F做会更明显
hua_x = cv2.Sobel(hua, cv2.CV_64F, 0, 1, ksize=3)
hua_y = cv2.Sobel(hua, cv2.CV_64F, 1, 0, ksize=3)
hua_x = cv2.convertScaleAbs(hua_x)
hua_y = cv2.convertScaleAbs(hua_y)
hua_xy = cv2.addWeighted(hua_x, 0.5, hua_y, 0.5, 0)
cv2.imshow('hua_xy_64F', hua_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sobel:Gx  [ -1  0   1                 Gy  [   -1  -2  -1
#             -2  0   2                         0   0   0
#             -1  0   1   ]                     1   2   1   ]

# scharr：更敏感
# Scharr:  [  -3  0   3                 Gy  [  -3  -10  -3
#            -10  0  10                         0   0    0
#             -3  0   3   ]                     3   10   3   ]

# laplacian     [   0   -1  0
#                   -1  4   -1
#                   0   -1  0   ]

hua_x = cv2.Scharr(hua, cv2.CV_64F, 0, 1)
hua_y = cv2.Scharr(hua, cv2.CV_64F, 1, 0)
hua_x = cv2.convertScaleAbs(hua_x)
hua_y = cv2.convertScaleAbs(hua_y)
hua_xy = cv2.addWeighted(hua_x, 0.5, hua_y, 0.5, 0)
cv2.imshow('hua_xy_Scharr', hua_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()

hua_x = cv2.Laplacian(hua, cv2.CV_64F)
hua_y = cv2.Laplacian(hua, cv2.CV_64F)
hua_x = cv2.convertScaleAbs(hua_x)
hua_y = cv2.convertScaleAbs(hua_y)
hua_xy = cv2.addWeighted(hua_x, 0.5, hua_y, 0.5, 0)
cv2.imshow('hua_xy_Laplacian', hua_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()


# canny 边缘检测
# 1，高斯滤波器
# 2，计算每个像素点的梯度和方向
# 3，非极大值抑制
# 4，双阈值检测，确定真实和潜在边缘
# 5，抑制孤立弱边缘

hua = cv2.imread('../../dataset/hua.jpg', 0)
hua = cv2.resize(hua, (0, 0), None, 0.5, 0.5)
cv2.imshow('hua', hua)

v1 = cv2.Canny(hua, 80, 150)
v2 = cv2.Canny(hua, 50, 100)

res = np.hstack((v1, v2))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
