"""import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

DATADIR = "./Traindata"
CATEGORIES = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5','Gesture_6','Gesture_7','Gesture_8','Gesture_9']
#classes=['Gesture-0','Gesture-1','Gesture-2','Gesture-3','Gesture-4','Gesture-5','Gesture-6','Gesture-7','Gesture-8','Gesture-9']
IMG_SIZE=50
training_data=[]
def creating_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    num_classes = CATEGORIES.index(category)
    print(num_classes)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
      new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
      training_data.append([new_array,num_classes])
            
creating_training_data()
#print(len(training_data))
random.shuffle(training_data)
"""for sample  in training_data[:10]:
  print(sample[1])
"""

X = []
y = []
for features,labels in training_data:
  X.append(features)
  y.append(labels)
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(X.shape)

"""