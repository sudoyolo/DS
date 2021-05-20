#Importing required Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

#Defining Data Directory and Categories 
DATADIR = "C:/EvolvFit/Dataset/Images"
CATEGORIES = ["Bhuvneshwar_Kumar", "Dinesh_Karthik", "Hardik_Pandya", "Jasprit_Bumrah", "K._L._Rahul", "Kedar_Jadhav", "Kuldeep_Yadav", "Mohammed_Shami", "MS_Dhoni", "Ravindra_Jadeja", "Rohit_Sharma", "Shikhar_Dhawan", "Vijay_Shankar", "Virat_Kohli", "Yuzvendra_Chahal"]

#Normalising Data - Setting Images to be 100x100
IMG_SIZE = 200

#Creating Dataset - Mapping Category to respective Image Folder
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)/15
        for img in os.listdir(path):
            try: 
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass           
create_training_data()

#Shuffling Data 
random.shuffle(training_data)

#Packing the data into variables before feeding through Neural Network
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

#Saving Built Dataset
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()




