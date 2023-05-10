import cv2
import numpy as np
import time

img_name = "1.jpg"
img = cv2.imread("input/" + img_name)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_new = img_gray.copy()


start = time.time()

img_gray_new = cv2.equalizeHist(img_gray)

print("time : ", time.time() - start)

res = np.hstack((img_gray ,img_gray_new))
cv2.imshow("img_gray :: img_gray_new",res)
cv2.waitKey(0)
cv2.imwrite("output/cpu_opencv_lib_img_"  + img_name, img_gray_new)