# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:41:35 2025

@author: Deepika
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('D:/Diabetes_Prediction_Using_Machine_Learning/trained_model.sav', 'rb'))


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
  
