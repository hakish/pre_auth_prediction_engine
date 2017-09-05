# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:43:35 2017

@author: kisho
"""

import pickle
import logging
def predict(model_file_path, x_predictor):
    logging.info('model file path is :: '+model_file_path)
    loaded_model = pickle.load(open(model_file_path, 'rb'))
    logging.info("loaded model is :: "+str(loaded_model))
    y_pred = loaded_model.predict(x_predictor)
    logging.info("y_pred is :: "+str(y_pred))
    y_pred_prob = loaded_model.predict_proba(x_predictor)[:,1]
    logging.info("y_pred probabilities is :: "+str(y_pred_prob))
    return (y_pred, y_pred_prob)


