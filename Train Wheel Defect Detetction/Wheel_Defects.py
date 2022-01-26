import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


dataset=pd.read_csv('wheels.csv')
X=dataset[['sensor_1','sensor_2','sensor_3','sensor_4']]
y=dataset[['defect']]

ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.30)

model = Sequential()
model.add(Dense(6, input_dim = 4))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, epochs = 50, batch_size = 5)
history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 30)

Y_pred = model.predict(X_test)
pred = list()
for i in range(len(Y_pred)):
    pred.append(np.argmax(Y_pred[i]))
test = list()
for i in range(len(Y_test)):
    test.append(np.argmax(Y_test[i]))
a = accuracy_score(pred,test)
print("Accuracy of the model: ", a*100)
model.save('model.h5')


print(model.predict([4,5,3,1]))