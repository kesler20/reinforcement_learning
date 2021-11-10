import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing as pp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.lite as tflite
import time
from data_processing import DataProcessing
import random
import math

gestures = {
    0:'tense',
    1:'flex',
    2:'relaxed'
}

class PatternRecognition(DataProcessing):

    def __init__(self, number_of_gestures, gestures, X, Y):
        self.number_of_gestures = number_of_gestures
        self.gestures = gestures
        self.X = X
        self.Y = Y

    def initialize_model_architecture(self, data, number_of_labels):
        X = self.X
        Y = self.Y
        train_labels, train_samples = shuffle(X,Y)

        model = Sequential([
            Dense(units=len(data.columns), activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=number_of_labels, activation='softmax')
        ])

        X = np.asarray(X).astype('float32')
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

        model.fit(X, Y, epochs=30, shuffle=True, batch_size=30, use_multiprocessing=True, verbose=2)

    def save_best_model(self, repeats=15):
        x_data, y_data = self.X, self.Y
        best = (0, None)
        for n in range(repeats):
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
            model.fit(x_train, y_train, epochs=30, shuffle=True, batch_size=30, use_multiprocessing=True)
            acc = model.evaluate(x_data, y_data, use_multiprocessing=True)[1]
            if acc > best[0]:
                best = (acc, model)
        print(f'Best was {best[0] * 100}%')
        best[1].save('model2.tf')
        lite_model = tflite.TFLiteConverter.from_keras_model(best[1])
        open("model2.tflite", "wb").write(lite_model.convert())

    def discrete_model(self,gestures, model):
        X, Y = self.X, self.Y
        position = random.randint(0,len(X))
        n1 = X[position]
        n = self.series_to_list(n1)
        h = []
        h.append(n)
        n = np.array(h, dtype='float32')
        prediction = model.predict(n)
        predicted_label = gestures[np.argmax(prediction)]
        actual_label = gestures[Y[position]]
        print(f"prediction : {predicted_label}, actual : {actual_label}")
        time.sleep(5)

    def elbow_angle(self,data, model):
        X, Y = self.X, self.Y
        position = random.randint(0,len(X))
        n1 = X[position]
        n = self.series_to_list(n1)
        h = []
        h.append(n)
        n = np.array(h, dtype='float32')
        prediction = model.predict(n)
        angles_in_radians = []
        for i in data['gyro0']:
            if i > 135:
                pass
            elif i < 0:
                pass
            else:
                x = (i/360)*math.pi
                angles_in_radians.append(x)
        try:
            print(prediction, angles_in_radians[position])
        except IndexError:
            time.sleep(1)


number_of_gestures = 7
gestures = {
    0:'Clench-Fist',
    1:'Spider-Man',
    2:'Thumb-to-pinky',
    3:'Wrist-side-to-side-horizontal',
    4:'Wrist-up-and-down',
    5:'Wrist-rotate-inwards',
    6:'Wrist-side-to-side-vertical',
    7:'Pointer-finger'
}

# make sure that the data is save in the save repository of the pyton file and the model as well
mark_data = pd.read_excel(r'bionic_society\clean_data_mark.xlsx')
mark_data = mark_data.drop([' '], axis=1)
print(mark_data)
print(mark_data.columns)
mark_data = mark_data.drop(['time'], axis=1)
X = np.array(mark_data.drop(['Classes'], axis=1))
Y = np.array(mark_data['Classes'])

# load the model architecture 
model = load_model('model2.tf')

# make an instance of the PatternRecogninition object
arm_model = PatternRecognition(number_of_gestures,gestures, X, Y)

if __name__ == '__main__':
    # both methods simulate an indefinite number of random inputs passed to the model
    while True:
        #the ouput of the elbow_angle method is printed on the terminal and it shows 
        # on the left: an arraw of which gesture has been detected, there are 7 outputs for the 7 gestures and the predicted gestures has a value of 1
        # on the right the angle that the gyro sensor detected normalised in radians from 0 - 135 degrees equivalent
        # when the input is out of range i.e. when the gyro detects a movement outside of the specified interval the model waits 1 second
        arm_model.elbow_angle(mark_data, model)

        # the output of the discrete model method is on the left the predicted gesture and on the right the actual gesture being passed onto the model
        arm_model.discrete_model(gestures, model)

