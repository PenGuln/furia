import sys
import numpy as np
import os
import joblib
def predict(model_file):
	model = joblib.load(model_file)
	data = []
	line = sys.stdin.readline()
	while line:
		Ld = line.strip().split(',')
		xx = [float(item) for item in Ld]
		data = data + xx
		if (len(data) >= 900):
			data = data[-900:]
			X = np.array(data).reshape((1, 900))
			predict = model.predict(X)
			print(predict[0])
			sys.stdout.flush()
		line = sys.stdin.readline()

if __name__ == '__main__':
	model_file = sys.argv[1]
	predict(model_file)