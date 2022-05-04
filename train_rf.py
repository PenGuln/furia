import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import sklearn.metrics as sm
from nyoka import skl_to_pmml
xtrain = pd.read_csv('dataset/xtrain.csv', header = None).values
ytrain = pd.read_csv('dataset/ytrain.csv', header = None).values
xtest = pd.read_csv('dataset/xtest.csv', header = None).values
ytest = pd.read_csv('dataset/ytest.csv', header = None).values
ytrain = ytrain.ravel()
ytest = ytest.ravel()
pipe = Pipeline([
                 ('clf',RandomForestClassifier(criterion='gini', n_estimators=24, random_state=80))
                 ])
pipe.fit(xtrain, ytrain)
print('Test accuracy is %.3f' % pipe.score(xtest, ytest))
skl_to_pmml(pipe, range(1800), "Activity", "rf.pmml")