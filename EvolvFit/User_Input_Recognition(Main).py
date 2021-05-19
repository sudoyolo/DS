#Importing Required Libraries
import cv2
import tensorflow as tf

#Taking Image Input from User
img_path = input("Enter Image Path (Example:'X:/Images/Person1.jpg'): ")

#Categories (to find if image is of the following Cricketers)
CATEGORIES = ["Bhuvneshwar_Kumar", "Dinesh_Karthik", "Hardik_Pandya", "Jasprit_Bumrah", "K._L._Rahul", "Kedar_Jadhav", "Kuldeep_Yadav", "Mohammed_Shami", "MS_Dhoni", "Ravindra_Jadeja", "Rohit_Sharma", "Shikhar_Dhawan", "Vijay_Shankar", "Virat_Kohli", "Yuzvendra_Chahal"]

#Modifying Image to fit in the Model
def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Opening Saved Model
model = tf.keras.models.load_model("CNN.model")

#Running Prediction, Testing Image in the Model
prediction = model.predict([prepare(img_path)])

#Showing Output
print(CATEGORIES[int(prediction[0][0])])

 
    
