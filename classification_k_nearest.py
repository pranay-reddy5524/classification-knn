import numpy as np 
from sklearn import preprocessing,neighbors
from sklearn.model_selection import cross_validate,train_test_split
import pandas as pd 
df=pd.read_csv("D:\\neuralnetwork\\classification\\breast-cancer-wisconsin.data.txt")
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X= np.array(df.drop(['class'],1))
y= np.array(df['class'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy= clf.score(X_test,y_test)
print(accuracy)
sample = np.array([4,2,1,1,1,2,3,2,1])
sample = sample.reshape(len(sample ),-1)

prediction = clf.predict(sample)
print(prediction)