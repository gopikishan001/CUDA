import math
import cv2
import time
from numba import jit, cuda , int32
import numpy as np

new_frame_time = 0
prev_frame_time = 0

vidcap = cv2.VideoCapture("SLOW_MOTION_720p_1000FPS.mp4")
res, frame = vidcap.read()
total_pixel = frame.shape[0] * frame.shape[1]
img_gray_new = frame
hist_type = [0] * 256 # np.zeros(256)
cuda_hist = cuda.to_device(hist_type)
cuda_hist_new = cuda.to_device(hist_type)
# img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cuda_img_gray_new = cuda.to_device(img_gray)

@cuda.jit
def func1(img_gray, cuda_hist):

    row , col = cuda.grid(2)
    if row < img_gray.shape[0] and col < img_gray.shape[1] :
        cuda.atomic.add(cuda_hist , img_gray[row][col] ,1)

@cuda.jit
def func1_1(img_gray, cuda_hist):
    row , col = cuda.grid(2)

    shared_arr = cuda.shared.array(shape = 256 ,dtype = int32)

    if cuda.threadIdx.x < 256  and cuda.threadIdx.y == 0:
        shared_arr[cuda.threadIdx.x] = 0
        # cuda_hist[cuda.threadIdx.x] = 0

    cuda.syncthreads()

    if row < img_gray.shape[0] and col < img_gray.shape[1] :
        cuda.atomic.add(shared_arr , img_gray[row][col] ,1)

    cuda.syncthreads()

    if cuda.threadIdx.x < 256 and cuda.threadIdx.y == 0:
        cuda.atomic.add(cuda_hist , cuda.threadIdx.x , shared_arr[cuda.threadIdx.x])
        # cuda.atomic.exch(cuda_hist , cuda.threadIdx.x , shared_arr[cuda.threadIdx.x])

@jit 
def func2(total_pixel , hist) :
    curr = 0
    new_hist = [0] * 256
    for i in range(0,256):
        curr += hist[i]
        new_hist[i] = round((curr * 255 )/total_pixel) 

    return new_hist

@cuda.jit
def func2_1(total_pixel,hist , new_hist):
    curr = 0
    for i in range(0,256):
        curr += hist[i]
        new_hist[i] = round((curr * 255 )/total_pixel)

@cuda.jit
def func3(cuda_img_gray ,  cuda_hist_new):
    row , col = cuda.grid(2)

    if row < img_gray.shape[0] and col < img_gray.shape[1] :
        cuda_img_gray[row][col] =  cuda_hist_new[cuda_img_gray[row][col]]
        # cuda_img_gray_new[row][col] =  cuda_hist_new[cuda_img_gray[row][col]]

if vidcap.isOpened():
    
    while(True):
        res, frame = vidcap.read() 

        if res:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#==================================================================

            cuda_img_gray = cuda.to_device(img_gray)
            cuda_hist = cuda.to_device(hist_type)
            # cuda_hist_new = cuda.to_device(hist_type)

            matrix_row = img_gray.shape[0]  
            matrix_col = img_gray.shape[1]  
                
            threads = (256,4)
            blocks_x = int(math.ceil(matrix_row / threads[0]))
            blocks_y = int(math.ceil(matrix_col / threads[1]))

            # print( "blocks_x " ,blocks_x , "blocks_y " ,blocks_y , " :: col " , matrix_col , " :: thread ", threads)

            # func1[(blocks_x,blocks_y),threads](cuda_img_gray , cuda_hist)

            func1_1[(blocks_x,blocks_y),threads](cuda_img_gray , cuda_hist)
            
            # hist = cuda_hist.copy_to_host()
            # hist_new = func2(total_pixel ,hist )
            # cuda_hist_new = cuda.to_device(hist_new)

            # func2_1[1,1](total_pixel , cuda_hist , cuda_hist_new)

            # cuda_img_gray_new = cuda.to_device(img_gray)
            func3[(blocks_x,blocks_y),threads](cuda_img_gray  , cuda_hist_new)

            img_gray_new = cuda_img_gray.copy_to_host()

#==================================================================

            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            print(fps)

            cv2.putText(img_gray_new, str(fps)[:4], (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (255), 3, cv2.LINE_AA)

            cv2.imshow("img_gray",img_gray)
            cv2.imshow("img_gray_new",img_gray_new)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Capturing frame error")

else:
    print("Camera opening error")