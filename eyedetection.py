import numpy as np
import cv2
import time
import serial
cap = cv2.VideoCapture(2) 	#640,480
w = 960
h = 480
blink=0
flag1=0
sums=0
c = 'X'
ard=serial.Serial('COM5',9600,timeout=0.5)
ard.flush()

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		
		
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		faces=cv2.CascadeClassifier('haarcascade_eye.xml')
		#faces = cv2.CascadeClassifier('ojoD.xml')
		detected = faces.detectMultiScale(frame, 1.3, 5)
	
		
		pupilFrame = frame
		pupilO = frame
		windowClose = np.ones((5,5),np.uint8)
		windowOpen = np.ones((2,2),np.uint8)
		windowErode = np.ones((1,1),np.uint8)
		
		if len(detected)==0:
			#blink+=1
		
			 if( c != 'B' ):
				 ard.write('S')
				 fo=open('D:/eye.txt','a')
				 fo.write('B')
				 fo.close()

		else:
			
			for (x,y,w,h) in detected:
				
			#	if blink>1:
			#		print "exit"
			##		fo=open('D:/eye.txt','a')
			#		fo.write('B')
			#		fo.close()
			#		blink=0
##				elif blink>1:
##					print "blink"
##					fo=open('D:/eye.txt','a')
##					fo.write('B')
##					fo.close()
##					blink=0
				blink=0
				img=frame[y+(h*.25):(y+h), x:(x+w)+5]
				#pupilFrame = cv2.equalizeHist(frame[y+(h*.25):(y+h), x:(x+w)+5])
				pupilO = pupilFrame
				template = cv2.imread('E:\eye.jpg',0)
				#cv2.imwrite('center.jpg',img)
				#ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)		#50 ..nothin 70 is better
				#img = cv2.medianBlur(pupilFrame,5)
				img2 = img.copy()
				#cv2.imshow('aaa',img)
				w, h = template.shape[::-1]
				
# All the 6 methods for comparison in a list
				methods = ['cv2.TM_CCOEFF']

				for meth in methods:
					img = img2.copy()
					method = eval(meth)

				    # Apply template Matching
					
					res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
					min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
					
				    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
					if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
					    top_left = min_loc
					else:
					    top_left = max_loc
					bottom_right = (top_left[0] + w, top_left[1] + h)
					

					
					print ((top_left[0]+bottom_right[0])/2)*((top_left[0]+bottom_right[0])/2),((top_left[1]+bottom_right[1])/2)*((top_left[1]+bottom_right[1])/2)
					if ((top_left[0]+bottom_right[0])/2)*((top_left[0]+bottom_right[0])/2)<30000 and (top_left[0]+bottom_right[0])/2*((top_left[0]+bottom_right[0])/2) > 24000:
					    if( c != 'C'):
						    print "center"
						    ard.write('W')
						    c = 'C'
						    fo=open('D:/eye.txt','a')
						    fo.write('C')
						    fo.close()
					#elif abs((top_left[0]+bottom_right[0])/2-(top_left[1]+bottom_right[1])/2)<10:
					elif ((top_left[0]+bottom_right[0])/2)*((top_left[0]+bottom_right[0])/2)<30000 :
					    if( c!= 'R'):    
						    print "right"
						    c = 'R'
						    ard.write('D')
						    fo=open('D:/eye.txt','a')
						    fo.write('R')
						    fo.close()
					elif ((top_left[0]+bottom_right[0])/2)*((top_left[0]+bottom_right[0])/2)>24000:
					    if( c!= 'L') :   
						    print "left"
						    c = 'L'
						    ard.write('A')
						    fo=open('D:/eye.txt','a')
						    fo.write('L')
						    fo.close()
					cv2.imshow('aa',img)
##				
##				cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
##
##				circles = cv2.HoughCircles(pupilFrame,cv2.cv.CV_HOUGH_GRADIENT,1,20,
##			    param1=255,param2=50,minRadius=0,maxRadius=0)
##				cv2.imshow('aa',img)
##				cv2.imwrite('temp.jpg',img)
##				
##				
##				if circles!=None:
##				    print "yes"
##				    circles = np.uint16(np.around(circles))
##				    for i in circles[0,:]:
##	    # draw the outer circle
##										cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
##		# draw the center of the circle
##										cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
##
##										cv2.imshow('detected circles',cimg)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
