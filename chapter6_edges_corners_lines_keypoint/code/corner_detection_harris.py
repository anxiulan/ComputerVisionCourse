import cv2
import numpy as np

img = cv2.imread('../../dataset/hua.jpg')
img = cv2.resize(img, (0,0), None, 0.5, 0.5)
# img = cv2.imread('../../dataset/star.png
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("origin", img)

gray = np.float32(gray)

# res = cv2.cornerHarris(gray, 2, 3, 0.04)
# - img： 输入图像，灰度图像，float32
# - blockSize: 用于角点检测的邻域的大小
# - ksize: Sobel导数的孔径参数
# - k : 方程中的k-Harris检测器自由参数
# - res：返回值，灰度图像
res = cv2.cornerHarris(gray, 2, 3, 0.04)

# 扩大标记的内容
res = cv2.dilate(res, None)

# 最佳阈值因图而异
img[res > 0.01 * res.max()] = [0, 0, 255]

cv2.imshow('Harris res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
