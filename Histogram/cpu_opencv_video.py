import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


new_frame_time = 0
prev_frame_time = 0

vidcap = cv2.VideoCapture(0)

if vidcap.isOpened():
    
    while(True):
        res, frame = vidcap.read() 

        if res: 

            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_gray_new = cv2.equalizeHist(img_gray)

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.putText(img_gray_new, str(fps)[:4], (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (255), 3, cv2.LINE_AA)

            # img_gray = cv2.resize(img_gray, (100, 200))
            # img_gray_new = cv2.resize(img_gray_new, (100, 200))


            cv2.imshow("img_gray" , img_gray)
            cv2.imshow("img_gray_new",img_gray_new)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Capturing frame error")

else:
    print("Camera opening error")