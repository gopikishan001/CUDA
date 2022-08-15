import math
import cv2
import time
import numpy as np
from numba import jit, cuda , int32

img = cv2.imread("input/3.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

k = 25
k_m = np.ones((k,k), dtype=int)
k_data = [k_m.shape[0] , k_m.shape[1] , int(k / 2) ]

img_gray_pad = np.pad(img_gray , (k_data[2]) , "constant" ,constant_values = (255))
#err = max = 255
#dil = min = 0

cuda_img_gray = cuda.to_device(img_gray_pad)
cuda_k = cuda.to_device(k_data)

#===============================================================
@cuda.jit
def reset_0(cuda_img , cuda_k) :
    row , col = cuda.grid(2)

    if ( row > cuda_img.shape[0] - cuda_k[2] -1 or row < cuda_k[2] )  :
        cuda_img[row][col] = 0
    
    if (col > cuda_img.shape[1] - cuda_k[2] -1 or col < cuda_k[2] ) :
        cuda_img[row][col] = 0

@cuda.jit
def dal(cuda_img_new ,cuda_img , cuda_k) :
    row , col = cuda.grid(2)

    max = 0
    if row < cuda_img.shape[0] - cuda_k[2]  and row > cuda_k[2] -1 and col < cuda_img.shape[1] - cuda_k[2] and col > cuda_k[2] -1  :
        for i in range(0,cuda_k[0]) :
            for j in range(0 , cuda_k[1]) :
                val = cuda_img[row-cuda_k[2]+i][col-cuda_k[2]+j] 
                if val > max :
                    max = val 

        cuda_img_new[row][col] = max

@cuda.jit
def ero(cuda_img_new ,cuda_img , cuda_k) :
    row , col = row , col = cuda.grid(2)

    min = 256
    if row < cuda_img.shape[0] - cuda_k[2] and row > cuda_k[2] -1 and col < cuda_img.shape[1] - cuda_k[2] and col > cuda_k[2] -1  :
        for i in range(0,cuda_k[0]) :
            for j in range(0 , cuda_k[1]) :
                val = cuda_img[row-cuda_k[2]+i][col-cuda_k[2]+j] 
                if val < min :
                    min = val 

        cuda_img_new[row][col] = min

@cuda.jit
def w_top_hat(orig_img,opening , cuda_k) :
    row , col = row , col = cuda.grid(2)
    if row < opening.shape[0] - cuda_k[2] and row > cuda_k[2] -1 and col < opening.shape[1] - cuda_k[2] and col > cuda_k[2] -1  :
        orig_img[row][col] = orig_img[row][col] - opening[row][col]

#=========================================================================

matrix_row = img_gray_pad.shape[0]  
matrix_col = img_gray_pad.shape[1]  
                
threads = (32,32)
blocks_x = int(math.ceil(matrix_row / threads[0]))
blocks_y = int(math.ceil(matrix_col / threads[1]))

cuda_ero_img = cuda.to_device(img_gray_pad)
ero[(blocks_x , blocks_y) , threads](cuda_ero_img ,cuda_img_gray , cuda_k)

reset_0[(blocks_x,blocks_y) , threads](cuda_ero_img , cuda_k)

cuda_ero_dal_img = cuda.to_device(img_gray_pad)
dal[(blocks_x , blocks_y) , threads](cuda_ero_dal_img ,cuda_ero_img , cuda_k)

w_top_hat[(blocks_x,blocks_y), threads](cuda_img_gray , cuda_ero_dal_img , cuda_k)

#=========================================================================

w_top_hat = cuda_img_gray.copy_to_host()
w_top_hat = w_top_hat[k_data[2]: img_gray_pad.shape[0] - k_data[2] , k_data[2]: img_gray_pad.shape[1] - k_data[2]]

cv2.imshow("White Top hat" , w_top_hat)

ero = cv2.erode(img_gray, k_m, iterations=1)
dil = cv2.dilate(ero, k_m, iterations=1)
final = img_gray - dil
cv2.imshow("opencv final " , final)

cv2.waitKey(0)


