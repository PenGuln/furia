import os
import sys
import numpy as np
from pandas import read_csv
import joblib
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
	#trainX = trainX.reshape(trainX.shape[0], 20, 90)
	trainX = np.array(trainX)
	trainX = trainX.reshape(trainX.shape[0], 900)
	trainy = np.array(trainy)
	return trainX, trainy

# fit an model from base_model
def fit_new_model(trainX, trainy, base_model, new_model):
    model = joblib.load(base_model)
    model.fit(trainX, trainy)
    joblib.dump(model, new_model)

if __name__ == '__main__':
	arg = sys.argv
	data_dir = arg[1]
	new_model_path =  arg[2]
	base_model_path = arg[3]
	trainX, trainy = load_data(data_dir)
	fit_new_model(trainX, trainy, base_model_path, new_model_path)

