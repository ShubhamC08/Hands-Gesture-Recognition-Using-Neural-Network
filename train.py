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
	layer=tf.nn.relu(layer)
	return layer


