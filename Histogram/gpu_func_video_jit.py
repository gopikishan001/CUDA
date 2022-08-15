import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit
from numba import cuda

new_frame_time = 0
prev_frame_time = 0

vidcap = cv2.VideoCapture(0)
res, frame = vidcap.read()
total_pixel = frame.shape[0] * frame.shape[1]
img_gray_new = frame

# @cuda.jit
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

if vidcap.isOpened():
    
    while(True):
        res, frame = vidcap.read() 

        if res:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # img_gray_new = func[total_pixel , 32](img_gray)
            img_gray_new = func(img_gray)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.putText(img_gray_new, str(fps)[:4], (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (255), 3, cv2.LINE_AA)

            cv2.imshow("img_gray",img_gray)
            cv2.imshow("img_gray_new",img_gray_new)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Capturing frame error")

else:
    print("Camera opening error")