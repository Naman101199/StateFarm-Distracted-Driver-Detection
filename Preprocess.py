import os
import numpy as np
import cv2


path="train"

CATEGORIES = ["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
IMG_SIZE = 224


for category in CATEGORIES : 
    l=os.path.join(path,category)
    for img in tqdm(os.listdir(path)) :
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)


training_data = []

def creating_training_data() :  
    for category in CATEGORIES :
        l=os.path.join(path,category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(l)) :
                img_array = cv2.imread(os.path.join(l,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            
creating_training_data()
            
shuffle(training_data)


X = []    #features
y = []    #labels

for features, labels in training_data :
    X.append(features)
    y.append(labels)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


np.save("X",X)
np.save("y",y)
