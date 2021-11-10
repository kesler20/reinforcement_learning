import pandas as pd
from numpy.core.numeric import NaN
import pandas as pd
from pandas.core import series
import talib as tl
from matplotlib import pyplot as plt
import numpy as np

class DataProcessing(object):

    def __init__(self, data):
        self.data = data

    def get_datetime_from_twelve(self):
        data = pd.DataFrame(self.data)
        data.to_csv('batch.csv')
        data = pd.read_csv('batch.csv')
        print(data)
        try:
            t = data.datetime
        except KeyError:
            t = data.date
        return t

    def get_ticker_from_twelve(self):
        data = pd.DataFrame(self.data)
        print(data)
        data.to_csv('batch.csv')
        data0 = pd.read_csv('batch.csv')
        data = data.columns
        return data0[data[0]]

    def real_data_to_double(self,real_data):
        float_data = [float(x) for x in real_data]
        np_float_data = np.array(float_data)
        return np_float_data  
    
    def generate_n_lenght_list(self, n):
        t = []
        null = [t.append(i) for i in range(n)]
        return t

    def list_of_zeros(self, value):
        list_of_zeroes = []
        empty_container = [list_of_zeroes.append(0) for x in range(len(value) + 1)]
        return list_of_zeroes, empty_container
    
    def series_to_list(self, series):
        list1 = []
        for i in series:
            list1.append(i)
        return list1

    def remove_repeats(self, list1):
        list2 = []
        for i in list1:
            if i in list2:
                pass
            else:
                list2.append(i)
        return list2

    def find_location(self, value):
        i = self.data.isin([value])
        c = 0
        t = 0
        for p in i:
            c += 1
            if p == True:
                t = c
            else:
                pass
        return t

    def Convert(self, tup, di): 
        for a, b in tup: 
            di.setdefault(a, []).append(b) 
        return di 
    
    def seperate_even_location_odd_location(self):
        x = self.data
        even = []
        odd = []
        for i in range(len(x)):
            if i % 2 == 1:
                even.append(x[i])
            else:
                odd.append(x[i])

        df1 = pd.DataFrame(odd)
        df2 = pd.DataFrame(even)
        return df1, df2
        
if __name__ == '__main__':
    number_of_gestures = 7
    gestures = { '0': 'Clench-Fist', 
    '1': 'Spider-Man',
    '2': 'Thumb-to-pinky',
    '3': 'Wrist-side-to-side-horizontal',
    '4': 'Wrist-up-and-down',
    '5': 'Wrist-rotate-inwards',
    '6': 'Wrist-side-to-side-vertical',
    '7': 'Pointer-finger'}

    #import all the data
    data0 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G0.csv')
    data1 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G1.csv')
    data2 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G2.csv')
    data3 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G3.csv')
    data4 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G4.csv')
    data5 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G5.csv')
    data6 = pd.read_csv(r'C:\Users\Uchek\OneDrive\Documents\Projects\learningpython\bionic_society\data_final-mark_emgData-G6.csv')

    classes_data0 = []
    for i in range(len(data0)):
        classes_data0.append(0)

    classes_data1 = []
    for i in range(len(data1)):
        classes_data1.append(1)

    classes_data2 = []
    for i in range(len(data2)):
        classes_data2.append(2)

    classes_data3 = []
    for i in range(len(data3)):
        classes_data3.append(3)

    classes_data4 = []
    for i in range(len(data4)):
        classes_data4.append(4)

    classes_data5 = []
    for i in range(len(data5)):
        classes_data5.append(5)

    classes_data6 = []
    for i in range(len(data6)):
        classes_data6.append(6)

    data0['Classes'] = pd.DataFrame(classes_data0)
    data1['Classes'] = pd.DataFrame(classes_data1)
    data2['Classes'] = pd.DataFrame(classes_data2)
    data3['Classes'] = pd.DataFrame(classes_data3)
    data4['Classes'] = pd.DataFrame(classes_data4)
    data5['Classes'] = pd.DataFrame(classes_data5)
    data6['Classes'] = pd.DataFrame(classes_data6)

    big = data0.append(data1)
    data = data2.append(data3)
    big_data = big.append(data)
    t = data4.append(data5)
    g = big_data.append(t)
    p = g.append(data6)
    p = p.fillna(0)

    p.replace('Repeat 1', 0, inplace=True)
    p.replace('Repeat 2', 0, inplace=True)
    p.replace('Repeat 3', 0, inplace=True)
    p.replace('Repeat 4', 0, inplace=True)
    p.replace('Repeat 5', 0, inplace=True)
    p.replace('Repeat 6', 0, inplace=True)
    p.replace('Repeat 7', 0, inplace=True)
    p.replace('Repeat 8', 0, inplace=True)
    p.replace('Repeat 9', 0, inplace=True)
    p.replace('Repeat 10', 0, inplace=True)
    p.replace('Repeat 11', 0, inplace=True)
    p.replace('Repeat 12', 0, inplace=True)
    p.replace('Repeat 13', 0, inplace=True)

    print(p)
    #p.to_excel('clean_data_mark.xlsx')
