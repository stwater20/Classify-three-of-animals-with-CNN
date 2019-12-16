import sys
import numpy as np
import cv2
import tensorflow as tf
IMG_SIZE=150
def prepare(filepath): 
    img_array = cv2.imread(filepath)  
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE,3) 
def main():
    #print(sys.argv[0])
    categories = ["Cat","Dog","Squirrel"]
    model = tf.keras.models.load_model("64x3-CNN.model")
    x = sys.argv[1]
    #print(x)
    testing = x # example: C:\Users\box88\Desktop\ai\1.jpg
    prediction = model.predict([prepare(testing)])
    maxindex  = np.argmax(prediction)
    print(categories[maxindex])

if __name__ == "__main__":
    main()

main()