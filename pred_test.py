import sys
import numpy as np
import cv2
import tensorflow as tf

def main():
    print(sys.argv[0])
    categories = ["Cat","Dog","Squirrel"]
    model = tf.keras.models.load_model("64x3-CNN.model")
    x = sys.argv[1]
    testing = repr(x) # example: C:\Users\box88\Desktop\ai\1.jpg
    prediction = model.predict([prepare(testing)])
    maxindex  = np.argmax(prediction)
    print(categories[maxindex])

if __name__ == "__main__":
    main()

main()