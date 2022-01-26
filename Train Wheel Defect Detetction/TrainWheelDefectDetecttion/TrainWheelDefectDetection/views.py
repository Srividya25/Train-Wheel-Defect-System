import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import *

from django.shortcuts import render
from django.contrib import messages


# Create your views here.
def home(request):
    inp = []
    if request.method == 'POST':
        s1 = float(request.POST['s1'])
        s2 = float(request.POST['s2'])
        s3 = float(request.POST['s3'])
        s4 = float(request.POST['s4'])
        print("Sensor Values:",s1,s2,s3,s4)
        print("Types: ", type(s1), type(s2), type(s3), type(s4))
        if(s1 < 0 or s1 > 10 or s2 < 0 or s2 > 10 or s3 < 0 or s3 > 10 or s4 < 0 or s4 > 10):
            messages.error(request, 'Values defying sensor limitations!!!')
            messages.error(request, 'Values might not be recorded from sensor.')
        else:
            inp.append(s1)
            inp.append(s2)
            inp.append(s3)
            inp.append(s4)
            print("Input to the Neural Network: ", inp, type(inp))
            model = load_model('model.h5')
            arg = np.argmax(model.predict([inp]))
            print("Result: ",arg)
            if arg == 0:
                messages.info(request, "Detected Flat Spot!!!")
            elif arg == 1:
                messages.info(request, "Detected Non-Roundness!!!")
            elif arg == 2:
                messages.info(request, "Detected Shelling!!!")
            else:
                messages.error(request, "System unable to detect the defect!!!")
        return render(request, 'home.html')
    else:
        return render(request, 'home.html')

def NeuralNetwork():
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

