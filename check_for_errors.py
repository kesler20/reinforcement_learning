
from bionic_society import model_emg 
import numpy as np
import pandas as pd

'''
from serial import Serial 

ser = Serial('COM3', 9600)
data = ser.readline(1000)
'''

mark_data = pd.read_excel(r'bionic_society\clean_data_mark.xlsx')
mark_data = mark_data.drop(['time'], axis=1)
X = np.array(mark_data.drop(['Classes'], axis=1))
Y = np.array(mark_data['Classes'])

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

model = model_emg.PatternRecognition(number_of_gestures, gestures, X, Y)
model.initialize_model_architecture(mark_data, number_of_gestures)
