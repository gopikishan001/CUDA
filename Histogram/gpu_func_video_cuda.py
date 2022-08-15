import math
import cv2
import time
from numba import cuda

new_frame_time = 0
prev_frame_time = 0

vidcap = cv2.VideoCapture(0)
res, frame = vidcap.read()
total_pixel = frame.shape[0] * frame.shape[1]
img_gray_new = frame

@cuda.jit
def func1(img_gray, cuda_hist):
    row , col = cuda.grid(2)
    if row < img_gray.shape[0] and col < img_gray.shape[1] :
        cuda.atomic.add(cuda_hist , img_gray[row][col] ,1)

@cuda.jit
def func2(total_pixel,hist , new_hist):
    curr = 0
    for i in range(0,255):
        curr += hist[i]
        new_hist[i] = round((curr * 255 )/total_pixel)

@cuda.jit
def func3(cuda_img_gray , cuda_img_gray_new , cuda_hist_new):
    row , col = cuda.grid(2)

    if row < img_gray.shape[0] and col < img_gray.shape[1] :
        cuda_img_gray_new[row][col] =  cuda_hist_new[cuda_img_gray[row][col]]



if vidcap.isOpened():
    
    while(True):
        res, frame = vidcap.read() 

        if res:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#==================================================================

            hist_type = [0] * 255
            cuda_img_gray = cuda.to_device(img_gray)
            cuda_hist = cuda.to_device(hist_type)
            cuda_hist_new = cuda.to_device(hist_type)

            matrix_row = img_gray.shape[0]  
            matrix_col = img_gray.shape[1]  
                
            threads = (32,32)
            blocks_x = int(math.ceil(matrix_row / threads[0]))
            blocks_y = int(math.ceil(matrix_col / threads[1]))
            # if blocks_x * blocks_y < 6 :
            #     print(blocks_x , " * " , blocks_y )
            # print( "blocks_x " ,blocks_x , "blocks_y " ,blocks_y , " :: col " , matrix_col , " :: thread ", threads)

            func1[(blocks_x,blocks_y),threads](cuda_img_gray , cuda_hist)

            func2[1,1](total_pixel , cuda_hist , cuda_hist_new)

            cuda_img_gray_new = cuda.to_device(img_gray)
            func3[(blocks_x,blocks_y),threads](cuda_img_gray ,cuda_img_gray_new , cuda_hist_new)

            img_gray_new = cuda_img_gray_new.copy_to_host()

#==================================================================

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