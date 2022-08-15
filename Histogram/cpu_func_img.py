import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

img_name = "1.jpg"
img = cv2.imread("input/" + img_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_new = img_gray.copy()


start = time.time()

hist = [0] * 255
new_hist = [0] * 255

for row in range(0,img_gray.shape[0]) :
	for col in range(0,img_gray.shape[1]) :
		hist[img_gray[row][col]]+= 1

total_pixel = img_gray.shape[0] * img_gray.shape[1]
curr = 0

for i in range(0,255):
	curr += hist[i]
	new_hist[i] = round((curr * 255 )/total_pixel)

for row in range(0,img_gray.shape[0]) :
	for col in range(0,img_gray.shape[1]) :
		img_gray_new[row][col] = new_hist[img_gray[row][col]]


print("time : ", time.time() - start)

# result = img_gray.flatten()
# result1 = img_gray_new.flatten()
# fig, ax = plt.subplots(2,2)
# ax[0,0].hist(result ,bins = range(0,260))
# ax[0,1].hist(result1 ,bins = range(0,260))
# plt.show()

res = np.hstack((img_gray ,img_gray_new))
cv2.imshow("img_gray :: img_gray_new",res)
cv2.waitKey(0)
cv2.imwrite("output/cpu_self_func_img_" + img_name , img_gray_new)

