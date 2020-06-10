import sklearn #library for ML algorithm
from sklearn.datasets import load_breast_cancer #fetches database related to breast cancer

from sklearn.model_selection import train_test_split #library to divide data into train and test part
from sklearn.naive_bayes import GaussianNB #library for naive bayes
from sklearn.metrics import accuracy_score # library for accuracy between 0 to 1
data=load_breast_cancer() #store it into varibale data 

label_names=data['target_names'] #class names
labels=data['target']
feature_names =data['feature_names']
features=data['data']

print(label_names)
print(labels[0])

print(feature_names[0])
print(features[0])

train,test,train_labels,test_labels=train_test_split(features,labels,test_size =0.40,random_state=42)
gnb=GaussianNB() #import gaussianNB module 

model=gnb.fit(train,train_labels) #train model with data train and its label and store in model variable 

preds=gnb.predict(test) #predection on our test data 
print("predected data in binary=\n",preds)

print("accuracy is =",accuracy_score(test_labels,preds))#gives accuracy with test label and predected test data 