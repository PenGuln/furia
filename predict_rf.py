from pypmml import Model
import pandas as pd
import sklearn.metrics as sm
xtest = pd.read_csv(r'E:\Guosiqi\codes\DSD2022\xtest.csv', header = None).values
ytest = pd.read_csv(r'E:\Guosiqi\codes\DSD2022\ytest.csv', header = None).values
ytest = ytest.ravel()

model = Model.load('rf.pmml')
for i in range(100):
    model.predict(xtest[i])
    print(i)
#print('Test accuracy is %.3f' % sm.accuracy_score(ytest, ypred))