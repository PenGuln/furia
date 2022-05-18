import sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]='/cpu:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import joblib
def predict(model_file):
	model = joblib.load(model_file)
	data = []
	line = sys.stdin.readline()
	while line:
		Ld = line.strip().split(',')
		xx = [float(item) for item in Ld]
		data = data + xx
		if (len(data) >= 1800):
			data = data[-1800:]
			X = np.array(data).reshape((1, 1800))
			predict = model.predict(X)
			print(predict[0])
		line = sys.stdin.readline()

if __name__ == '__main__':
	model_file = sys.argv[1]
	predict(model_file)