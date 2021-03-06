import os
os.environ["CUDA_VISIBLE_DEVICES"]='/cpu:0'
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
	namelist = ['sit', 'stand', 'walk', 'upstairs', 'downstairs', 'run']
	Label = {'sit':0, 'stand':1, 'walk':2, 'upstairs': 3, 'downstairs': 4, 'run': 5}
	trainX = []
	trainy = []
	window = 20
	for name in namelist:
		datafile = os.path.join(data_dir, name + '.csv')
		if (os.path.exists(datafile)):
			tx = read_csv(datafile, header = None).values[:,0:45]
			if (tx.shape[0] < window):
				break
			for i in range(tx.shape[0] - window):
				data1s = []
				for j in range(window):
					data1s = data1s + list(tx[i + j])
				trainX.append(data1s)
				trainy.append(Label[name])

	#trainX = read_csv(os.path.join(data_dir, 'xtrain.csv'), header = None).values
	#trainy = read_csv(os.path.join(data_dir, 'ytrain.csv'), header = None).values
	#trainX = trainX.reshape(trainX.shape[0], 20, 45)
	trainX = np.array(trainX)
	trainX = trainX.reshape(trainX.shape[0], 20, 45)
	trainy = np.array(trainy)
	trainy = trainy.reshape(-1, 1)
	trainy = to_categorical(trainy)
	return trainX, trainy
 
# fit an model from base_model
def fit_new_model(trainX, trainy, base_model, new_model):
	verbose, epochs, batch_size = 1, 15, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.load_weights(base_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(trainX, trainy, epochs = epochs, batch_size = batch_size, verbose = verbose)
	model.save_weights(new_model, save_format = 'h5')

if __name__ == '__main__':
	arg = sys.argv
	data_dir = arg[1]
	new_model_path =  arg[2]
	base_model_path = arg[3]

	link_file = os.path.join(os.path.dirname(base_model_path), 'model.h5')
	if (os.path.exists(link_file)):
		os.remove(link_file)
	os.symlink(base_model_path, link_file)

	trainX, trainy = load_data(data_dir)
	fit_new_model(trainX, trainy, link_file, new_model_path)

