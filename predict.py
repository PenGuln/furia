import sys
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

def prepare_model(model_file):
	n_timesteps = 20
	n_features = 90 
	n_outputs = 6
	filepath = model_file
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.load_weights(filepath)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def read_data():
	inputx = np.zeros(1800, dtype = np.float32)
	data = []
	for line in sys.stdin:
		Ld = line.strip().split()
		if (len(Ld) == 0): break
		data = data + Ld
	data = np.array(data, dtype = np.float32)
	for i in range(len(data)): 
		inputx[i] = data[i]
	inputx = inputx.reshape((1, 20, 90))
	return inputx

def predict(model):
	X = read_data()
	predict = model.predict(X)
	classification = np.argmax(predict, axis = 1)
	print(classification[0])

if __name__ == '__main__':
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
	model_file = sys.argv[1]
	model = prepare_model(model_file)
	predict(model)
