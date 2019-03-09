import dataset
import tensorflow as tf
import time
from datatime import delta
import math
import random
import numpy as np

#Adding seed so that random initialization is consistent.
from numpy.random import seed
seed()
from tensorflow import set_random_seed
set_random_seed(2)
batch_size=32


#Prepare input data
classes=['Gesture-0','Gesture-1','Gesture-2','Gesture-3','Gesture-4','Gesture-5','Gesture-6','Gesture-7','Gesture-8','Gesture-9']

#20% of the data will be automatically be used for validation
validation_size=0.2
img_size=50
num_channels=3
train_path='./Hands-Gesture-Recognition-Using-Neural-Network'

#we shall load all the training and validation images and labels into memory using openCV and we use that during training
data=dataset.read_train_set(train_path,img_size,classes,validation_size=validation_size)
print("Complete reading input data,will now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in validation-set:\t{}".format(len(data.valid.labels)))
session=tf.Session()

#Input Layer
x=tf.placeholder(tf.float32,shape=[None,img-size,num_channels],name='x')

#labels
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='actual_value')
y_true_cls=tf.argmax(y_true,dimesion=1)

#Network graph parameters
filter_size_conv1=2
num_filters_conv1=32

filter_size_conv2=2
num_filters_conv2=32

filter_size_conv3=2
num_filters_conv3=32

filter_size_conv4=2
num_filters_conv4=32

filter_size_conv5=2
num_filters_conv5=32

fc_layer_size=1024

#Creating weights
def create_weights(shape):
	return tf.variable(tf.truncated_normal(shape,stddev=0.05))

#Create biases
def create_biases(size):
	return tf.variable(tf.constant(0.05,shape=[size]))

#Creating Convolution layer
def create_convolution_layer(input,num_input_channels,conv_filter_size,num_filters):
	#Weights
	weights=create_weights(shape=[conv_filter_size,num_input_channels,num_filters])
	#Bias
	bias=create_biases(num_filters)
	#Convolutin layer
	c=tf.nn.conv2d(input=input,filter=weights,stride=[1,1,1,1],padding='SAME')
	c+=bias
	#Maxpooling layer
	s=tf.nn.maxpool(value=ci,k_size=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	#applying activation function Relu to max pooling output
	layer=tf.nn.relu(s)
	return layer

#Creating flattening layer
def create_flatten_layer(layer):
	#Calculating shape of layer based on previous layer
	layer_shape=layer.get_shape()
	#Number of features calulated instead of hard coding
	num_features=layer_shape[1:4].num_elements()
	#Flattening the layer
	layer=tf.reshape(layer,[-1,num_features])
	return layer

#Creating fully-connected layer
def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):
	#Defining trainable weights
	weights=create_weights(shape=[num_input,num_output])
	#Defining the biases
	bias=create_biases(num_outputs)
	#Fully connected layer takes input x and produces wx+b.WE use matmul to multiply matrices
	layer=tf.matmul(input,weights)+bias
	if use_relu:
		layer=tf.nn.relu(layer)
	return layer

#Convolution Layer and Max Pooling Layer 1
layer_conv1=create_convolution_layer(input=x,num_input_channels=num_channels,conv_filter_size=filter_size_conv1,num_filters=num_filters_conv1)

#Convolution Layer and Max Pooling Layer 2
layer_conv2=create_convolution_layer(input=layer_conv1,num_input_channels=num_filters_conv1,conv_filter_size=filter_size_conv2,num_filters=num_filters_conv2)

#Convolution Layer and Max Pooling Layer 3
layer_conv3=create_convolution_layer(input=layer_conv2,num_input_channels=num_filters_conv2,conv_filter_size=filter_size_conv3,num_filters=num_filters_conv3)

#Convolution Layer and Max Pooling Layer 4
layer_conv4=create_convolution_layer(input=layer_conv3,num_input_channels=num_filters_conv3,conv_filter_size=filter_size_conv4,num_filters=num_filters_conv4)

#Convolution Layer and Max Pooling Layer 5
layer_conv5=create_convolution_layer(input=layer_conv4,num_input_channels=num_filters_conv4,conv_filter_size=filter_size_conv5,num_filters=num_filters_conv5)

#Flattening Layer
layer_flat=create_flatten_layer(layer_conv5)

#Fully connected layer 1
layer_fc1=create_fc_layer(input=layer_flat,num_inputs=layer_flat.get_shape()[1:4].num_elements(),num_outputs=fc_layer_size,use_relu=True)

#Fully connected layer 2
layer_fc2=create_fc_layer(input=layer_fc1,num_inputs=fc_layer_size,num_outputs=num_classes,use_relu=False)

#Output Layers
y_pred=tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls=tf.argmax(y_pred,dimension=1) #Largest y_pred value classes,will be selected

Session.run(tf.global_variables_initializer())

#Error calculation and learning
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
Session.run(tf.global_variables_initializer())














