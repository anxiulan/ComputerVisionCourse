import cv2


def show_img(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


hua = cv2.imread('hua.jpg', 1)
hua = cv2.resize(hua, (0, 0), None, 0.5, 0.5)
show_img('hua', hua)
print(hua.shape)

# 金字塔上采样
# By default, size of the output image is computed as `Size(src.cols\*2, (src.rows\*2)`
up = cv2.pyrUp(hua)
show_img('up', up)
print(up.shape)
# 金字塔下采样
down = cv2.pyrDown(hua)
show_img('down', down)
print(down.shape)

# 轮廓检测
star = cv2.imread('star.png')
print(star.shape)
gray = cv2.cvtColor(star, cv2.COLOR_BGR2GRAY)
print(gray.shape)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
show_img('star', star)
show_img('thresh', thresh)
# mode:     轮廓检索模式  cv2.RETR_TREE:检索所有轮廓,重构嵌套轮廓的整个层次
# method:   轮廓逼近方法  cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[1]
print(cnt)
# 画轮廓前要拷贝一下，在拷贝的变量上画，否则会覆盖原图
draw_img = star.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 255, 255), thickness=3)  # -1表示所有轮廓都画
show_img('draw_img', draw_img)

# 计算面积
print(cv2.contourArea(cnt))
# 计算周长
print(cv2.arcLength(cnt, True))

# 轮廓近似
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 5)
show_img('res', res)

# 外接矩形
x, y, w, h = cv2.boundingRect(cnt)
img = star.copy()
# show_img(img)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
# show_img(img)

# 外接圆
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
image = cv2.circle(img, center, radius, (0, 255, 0), 2)
show_img('image', image)

