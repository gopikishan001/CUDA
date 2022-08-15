import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit

img_name = "1.jpg"
img = cv2.imread("input/" + img_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_new = img_gray.copy()

@jit(nopython=True)
def func(img_gray):
	hist = [0] * 255
	new_hist = [0] * 255
	img_gray_new = img_gray.copy()

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

	return img_gray_new

start = time.time()
img_gray_new = func(img_gray)
print("1st time : ", time.time() - start)

start = time.time()
img_gray_new = func(img_gray)
print("2nd time : ", time.time() - start)




res = np.hstack((img_gray ,img_gray_new))
cv2.imshow("img_gray :: img_gray_new",res)
cv2.waitKey(0)
# cv2.imwrite("output/gpu_self_func_img_" + img_name , img_gray_new)

