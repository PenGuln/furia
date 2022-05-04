import os
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

def load_dataset(prefix=''):
    trainX = read_csv('dataset/xtrain.csv', header = None).values
    trainy = read_csv('dataset/ytrain.csv', header = None).values
    testX = read_csv('dataset/xtest.csv', header = None).values
    testy = read_csv('dataset/ytest.csv', header = None).values
    trainX = trainX.reshape(trainX.shape[0], 20, 90)
    testX = testX.reshape(testX.shape[0], 20, 90)
    #print(trainX.shape, trainy.shape)
    #print(testX.shape, testy.shape)
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    #print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
 
# fit an initial model
def fit_model(trainX, trainy, testX, testy, output_file):
	verbose, epochs, batch_size = 1, 25, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	filepath = output_file
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(
		trainX, trainy, 
		epochs = epochs, 
		batch_size = batch_size, 
		validation_data = (testX, testy), 
		verbose = verbose, 
		callbacks = callbacks_list
	)

def evaluate_model(model_file, testX, testy):
	verbose, epochs, batch_size = 1, 25, 64
	n_timesteps, n_features, n_outputs = 20, 90, 6

	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.load_weights(model_file)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	_, accuracy = model.evaluate(testX, testy, batch_size = batch_size, verbose = 0)
	return accuracy

def work():
	trainX, trainy, testX, testy = load_dataset()
	fit_model(trainX, trainy, testX, testy, "bestmodel.h5")
	score = evaluate_model('bestmodel.h5', testX, testy)
	print("acc: ", score)

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
work()