from concurrent.futures import thread
from traceback import print_tb
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
from numba import jit
from numba import cuda
import math


@cuda.jit
def func(a) :
	row , col = cuda.grid(2)

	if row < a.shape[0] and col < a.shape[1] :
		a[row][col] *=2
		print(row , col)

	if ( row > a.shape[0] - 2 -1 or row < 2 )  :
		a[row][col] = 0
    
	if (col > a.shape[1] - 2 -1 or col < 2 ) :
		a[row][col] = 0

	# print(row , "  " , col)
	# pos = cuda.grid(1)
	# if pos < a.shape[0]:	
	
		# s = 0
		# for k in range(0, a.shape[0]) :
		# 	s += a[k]
		# 	a[k] =s


if __name__ == "__main__" :

	arr = np.ones((30,30) , dtype=int)
	# arr = [1] * 255

	start = time.time()

	# arr_cuda = cuda.device_array((16,16))
	arr_cuda = cuda.to_device(arr)

	# func[10,1024](arr_cuda)
	# func[20,4](arr_cuda)

	# matrix_col = 65535
	matrix_row = 30
	matrix_col = 30
	# threads = 1024
	threads = (5,5)
	blocks_x = int(math.ceil(matrix_row / threads[0]))
	blocks_y = int(math.ceil(matrix_col / threads[1]))
	# if blocks_x * blocks_y < 6 :
	# 	print(blocks_x , " * " , blocks_y )
	print( "blocks_x " ,blocks_x , " :: block_y " , blocks_y , " :: thread ", threads)
	# func[(blocks,matrix_col),threads](arr_cuda)
	func[(blocks_x,blocks_y),threads](arr_cuda)
	# func[(3,3),threads](arr_cuda)
	

	arr = arr_cuda.copy_to_host()

	# print("Time : " , time.time() - start)
	
	print(arr)
	# if 1 in arr:
	# 	print("sss")
