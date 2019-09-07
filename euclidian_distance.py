import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import style
from collections import Counter
import warnings
import random

"""style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [1,7]

plt.scatter(new_features[0],new_features[1])
[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.show()"""
def k_nearest_neighbors(data, predict, k=5):
	if len(data)>=k:
		warnings.warn('k is lessthan that ')
	distances=[]
	for group in data:
		for features in data[group]:
			euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidian_distance,group])
	votes = [i[1] for i in sorted(distances)[:k]]
	#print(votes)
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

"""result = k_nearest_neighbors(dataset,new_features,k=3)
print(result)"""

df = pd.read_csv("D:\\neuralnetwork\\classification\\breast-cancer-wisconsin.data.txt")
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
test_size = 0.2
train_set ={2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data  = full_data[-int(test_size*len(full_data)):]
"""
train data = 0-80% test_data 80-100%
in train_set we fill the dictionary with list of lits corresponding to the
type of cancer
"""

for i in train_data:
	train_set[i[-1]].append(i[:-1])
for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct=0
total=0
for group in test_set:
	for data in test_set[group]:
		vote = k_nearest_neighbors(train_set,data,k=5)
		if group== vote:
			correct+=1
		total+=1
print("accuracy :",correct/total)