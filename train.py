import os
import sys
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical

def load_data(data_dir):
	filelist = os.listdir(data_dir)
	trainX = read_csv(os.path.join(data_dir, 'xtrain.csv'), header = None).values
	trainy = read_csv(os.path.join(data_dir, 'ytrain.csv'), header = None).values
	trainX = trainX.reshape(trainX.shape[0], 20, 90)
	trainy = to_categorical(trainy)
	return trainX, trainy
 
# fit an model from base_model
def fit_new_model(trainX, trainy, base_model, new_model):
	verbose, epochs, batch_size = 1, 20, 5
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.load_weights(base_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)
	model.save_weights(new_model)

if __name__ == '__main__':
	arg = sys.argv
	data_dir = arg[1]
	new_model_path =  arg[2]
	base_model_path = arg[3]
	trainX, trainy = load_data(data_dir)
	fit_new_model(trainX, trainy, base_model_path, new_model_path)

