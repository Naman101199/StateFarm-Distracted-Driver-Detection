import os
import numpy as np
import cv2
import keras



TEST_DIR = "test"
IMG_SIZE = 224


testing_data = []
def test_data():
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)    
    #np.save('test_data.npy',testing_data)    
        
test_data()

X_test = []    #features
y_test = []    #labels

for features, labels in testing_data :
    X_test.append(features)
    y_test.append(labels)


X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

X_test = X_test/255.0

model_test = keras.models.load_model("my_model.h5")
prediction = model_test.predict([X_test])
print(prediction[2])

print(len(testing_data))


final_prediction = []
for i in range(len(testing_data)):
    final_prediction.append([testing_data[i][1],prediction[i]])

print(final_prediction)
