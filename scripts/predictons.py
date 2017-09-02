# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""

import pickle
def predict(model_file_path, x_predictor):
    loaded_model = pickle.load(open(model_file_path, 'rb'))
    print("loaded model is :: ", loaded_model)
    y_pred = loaded_model.predict(x_predictor)
    return y_pred

