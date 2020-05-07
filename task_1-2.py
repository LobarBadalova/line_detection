import cv2
import numpy as np 


cap=cv2.VideoCapture("Texas 4K - Sunrise Drive - Scenic Drive - USA_1.mp4")

while(cap.isOpened()):

	ret, frame=cap.read()

	if ret is False:
		print("End of the video")
		break

	#task 1: Denoising the video frames using GuassianBlur function
	blurred=cv2.GaussianBlur(frame, (5,5), 0)
	#task 2: converting 3-chanel color image into 1chanel gray image	
	blurred_gray=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	
	#reducing features from image using binarization
	edges = cv2.Canny(blurred_gray,100,200)
	#creating kernel	
	kernel_1=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	#filtering using filter2D
	sharpened_1=cv2.filter2D(edges, -1, kernel_1)

	
	

	
	cv2.imshow('sharpened_1', sharpened_1)
	
	



	q=cv2.waitKey(1)
	if q==ord('q'):
		print("q is pressed:quit")
		break


