import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='/cpu:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
def predict(model_file):
	n_timesteps = 20
	n_features = 45 
	n_outputs = 6
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	data = []
	line = sys.stdin.readline()
	while line:
		Ld = line.strip().split(',')
		xx = [float(item) for item in Ld]
		data = data + xx
		if (len(data) >= 900):
			data = data[-900:]
			X = np.array(data).reshape((1, n_timesteps, n_features))
			predict = model.predict(X)
			classification = np.argmax(predict, axis = 1)
			print(classification[0])
			sys.stdout.flush()
		line = sys.stdin.readline()

if __name__ == '__main__':
	model_file = sys.argv[1]
	link_file = os.path.join(os.path.dirname(model_file), 'model.h5')
	if (os.path.exists(link_file)):
		os.remove(link_file)
	os.symlink(model_file, link_file)
	predict(link_file)
	#predict(model_file)