import tensorflow as tf
import numpy as np
import os,cv2
import time
c_frame=-1
p_frame=-1
#setting threshold for number og frames to compare
thresholdframes=50
##Let us restore the saved model
sess = tf.Session()
#step-1:Recreate the network graph at this stop only graph is created
saver = tf.train.import_meta_graph('./Hands-Gesture-Recogntion-Using-Neural-Network')
#step-2: Now let's loads the weights saved using the restore method
saver.restore(sess.tf.train.latest_checkpoint('./'))
#accessing the default graph which we have restored
graph = tf.get_default_graph()
#Now, lets's get hold of the op that we can be processed to get the output
#In the Orginal network y_pred is hte tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")
#Let's feed the images to the input paceholders
x = graph.get_tensor_by_name('X:0')
y_true = graph.get_tensor_by_name"y_true:0")
y_test_images = np.zeros((1,10))


#Real Time Prediction
def predict(frame,y_test_images):
	image_size = 50
	image_channels = 3
	images=[]
	image=frame
	cv2.imshow('test',image)
	#Resizing the image to our desired size and preprocessing will be done exactly as done during training
	image= cv2.resize(image,(image_size,image_size),0,0,INTER_LINEAR)
	images.append(image)
	images = np.array(images,dtype=np.uint8)
	images = images.astype('float32')
	images = np.multipy(images,1.0/255.0)
	#This input to the network is of shape [None,image_size,image_size,num_channels] .here we reshape
	#Creating the feed_dict that is required to be required to be fed to calculate y_pred
	feed_dict_testing = {x:x_batch,y_true:y_true_images}
	result = sess.run(y_pred,feed_dict=feed_dict_testing)
	#result is of the this format[probability of gest_0,....,probability_of_gest_9]
	return np.array(result)
	#open camera object
	cap = cv2.VideoCapture(0)
	#Decrease frame size(4=width,5=height)
	cap = cv2.VideoCapture(0)
	#Decrease frame size (4=width,5=heights)
	cap.set(4,700)
	cap.set(5,400)
	h,s,v =150,150,150
	i=0
	while(i<1000000):
	ret,frame =cap.read()
	#makes rectangle 
	cv2.rectangle(frame,(300,300),(100,100),(0,255,0),0)
	crop_frame = frame[100:300,100:300]
	#blur the image
	#blur - cv2.blur(crop_frame,(3,3))
	#In image processing, a Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function
	blur = cv2.GaussianBlur(crop_frame,(3,3),0)
	#convert to HSV color space
	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	#create a binary image with where white will be skin colors and rest in black
	mask2 =cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255])
	#The Median Filter is a non-linear digital filtering technique, often used to remove noise from an image or signal. Such noise 		#reduction is a typical pre-processing step to improve the results of later processing (for example, edge detection on an image)
	med = cv2.medianBlur(mask2,5)
	#Display frames
	cv2.imshow('main',frame)
	cv2.imshow('masked',med)
	#resizing the image
	med = cv2.resize(med,(50,50))
	#making it 3 channel
	med = np.stack((med,)*3)
	#Adjusting rows,columns as per x
	med = np.rollaxis(med,axis=1,start=0)
	med = np.rollaxis(med,axis=2,start=0)
	#converting expo to float
	np.set_printoptions(formatter={'float_kind':'{:f}'.format})
	#printing index of max prob value
	# print(ans)
	#print (np.argmax(max(ans))
	#comparing for 50 continuous frames
	c_frame = np.argmax(max(ans))
	if(c_frame == p_frame):
		counter +=1
		p_frame = c_frame
		if(counter==thresholdframes):
			print(ans)
			print("gesture:"+str(c_frames))
			counter=0
			i=0
		else:
			p_frame = c_frame
			counter =0 
	#close the output video by pressing 'ESC'
	k = cv2.waitKey(2) & 0xFF
	if == 27:
		break
	i+=1
cap.release()
cap.destroyAllWindows()

	

