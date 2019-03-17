import hgr
import numpy as np
import tensorflow as tf
import datetime

#Prepare input data
classes = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5','Gesture_6','Gesture_7','Gesture_8','Gesture_9']
num_classes = len(classes)
print(num_classes)

#parameters of nn
height,width,img_size = 50,50,50
height=width = img_size
num_labels = 10
num_channels = 3
validatation_size =0.2
train_path ='./Traindata'
#reading the data files
data = hgr.read_train_sets(train_path,img_size,classes,validatation_size)

#parameters of each layer 
filter_size_conv1 = 2
num_filters_conv1 = 32

filter_size_conv1 = 2 
num_filters_conv1 = 32

filter_size_conv2 = 2
num_filters_conv2 = 32

filter_size_conv3 = 2
num_filters_conv3 = 64

filter_size_conv4 = 2
num_filters_conv4 = 32

filter_size_conv5 = 2
num_filters_conv5 = 64

fc_layer_size = 1024

#input layer 
#shape=[batch_size,height,width,num_channels]
X = tf.placeholder(tf.float32,shape=[None,height,width,num_channels],name='X')

#labels
# y shape will be like nu_labels will be columns (0-9) and None will be images in batch_size which will be rows
y = tf.placeholder(tf.float32,shape=[None,num_labels],name='y')
y_target_label = tf.argmax(y,dimension=1)


#create weights
def create_weights(shape,nm):
	return tf.Variable(initial_value=tf.truncated_normal(shape),name=nm)

#create biases
def create_baises(size,nm):
	return tf.Variable(tf.constant(0.05,shape=[size]),name=nm)

#create convolution layer
def create_convolution_layer(input,num_input_channels,filter_size,num_filters):
	#weights filters to train the CNN layer, we use weight method to define filter
	weights = create_weights(shape=[filter_size,filter_size,num_input_channels,num_filters],nm="filter")
	#bias 
	bias = create_baises(num_filters,"bias")
	#convolution layer
	layer = tf.nn.conv2d(input,filter=weights,strides=[1,1,1,1],padding='SAME')
	layer+=bias
	return layer

#create  maxpooling layer
def create_max_pool(input):
	layer = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	return layer

#create activation function 
def create_activation_function(input):
	layer = tf.nn.relu(input)
	return layer

#convert to the flat neural layer for fully connected nn
def create_flatten_layer(input):
	#get previous layer shape
	layer_shape = input.get_shape()
	## shape is in this format [batch_size,height,width,num_channels]
	#we need height,width,num_channels, so it will be height*width*num_channels number of features
	num_features =  layer_shape[1:4].num_elements()
	#layer will be
	layer = tf.reshape(input,shape=[-1,num_features])
	return layer

# create fully connected layer
#for 1st fc_layer num_inputs will be height*width*num_channels(2*2*64=256), num_outputs will be 1024 because we are creating the fc layer of 1024
#for 2nd  fc_layer num_inputs will be 1024 , output will be 10 with no relu 
def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	#weights for fully connected neural network
	weights = create_weights(shape=[num_inputs,num_outputs],nm='fc1_weights')
	#bias for fully connected nn
	bais = create_baises(num_outputs,"fc1_baises")
	# shape of layer is input([batch_size,256])*weights([1024,256])+bias(1024)
	layer = tf.matmul(input,weights)+bais 
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

#convolution layer
#1st convolution and max_pooling and then applied activation function
layer_conv1 = create_convolution_layer(input=X,num_input_channels=num_channels,filter_size=filter_size_conv1,num_filters=num_filters_conv1)
layer_max_pool1 = create_max_pool(layer_conv1)
layer_output1 = create_activation_function(layer_max_pool1)

#2nd 
layer_conv2 = create_convolution_layer(input=layer_output1,num_input_channels=num_filters_conv1,filter_size=filter_size_conv2,num_filters=num_filters_conv2)
layer_max_pool2 = create_max_pool(layer_conv2)
layer_output2 = create_activation_function(layer_max_pool2)

#3rd
layer_conv3 = create_convolution_layer(input=layer_output2,num_input_channels=num_filters_conv2,filter_size=filter_size_conv3,num_filters=num_filters_conv3)
layer_max_pool3 = create_max_pool(layer_conv3)
layer_output3 = create_activation_function(layer_max_pool3)

#4th
layer_conv4 = create_convolution_layer(input=layer_output3,num_input_channels=num_filters_conv3,filter_size=filter_size_conv4,num_filters=num_filters_conv4)
layer_max_pool4 = create_max_pool(layer_conv4)
layer_output4 = create_activation_function(layer_max_pool4)

#5th
layer_conv5 = create_convolution_layer(input=layer_output4,num_input_channels=num_filters_conv4,filter_size=filter_size_conv5,num_filters=num_filters_conv5)
layer_max_pool5 = create_max_pool(layer_conv5)
layer_output5 = create_activation_function(layer_max_pool5)

#flattening 
flat_output = create_flatten_layer(layer_output5)

#fully connected layer
#fc1 layer
fc1_layer_output = create_fc_layer(input=flat_output,num_inputs=flat_output.get_shape()[1:4].num_elements(),num_outputs=fc_layer_size,use_relu=True)
#fc2 layer
fc2_layer_output = create_fc_layer(input=fc1_layer_output,num_inputs=fc_layer_size,num_outputs=num_labels,use_relu=False)

#predection
y_pred = tf.nn.softmax(fc2_layer_output)
y_pred_label = tf.argmax(y_pred,dimension=1)

#error calculation
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2_layer_output,labels=y)
cost = tf.reduce_mean(cross_entropy)
#evaluation
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(y_pred_label,y_target_label)
accuracy =  tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
saver =tf.train.Saver()

##Execution phase
n_epoch = 10000
batch_size = 50

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epoch):
		X_batch,y_batch,_,cls_batch = data.train.next_batch(batch_size)
		X_validate_batch,y_validate_batch,_,cls_validate_batch = data.train.next_batch(batch_size)
		sess.run(optimizer,feed_dict ={X:X_batch,y:y_batch})
		if epoch % int(data.train.num_examples/batch_size) ==0:
			validate_loss = sess.run(cost,feed_dict={X:X_validate_batch,y:y_validate_batch})
			acc = sess.run(accuracy, feed_dict={X:X_batch,y:y_batch})
			val_acc = sess.run(accuracy, feed_dict={X:X_validate_batch,y:y_validate_batch})
			msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
			print(msg.format(epoch + 1, acc, val_acc, validate_loss))
			save_path=saver.save(sess,'./model.ckpt')
			print("Nodel saved in path %s" %save_path)
















	
