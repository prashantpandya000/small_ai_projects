import numpy as np 
import sklearn.preprocessing as preprocessing
input_data =np.array([[2.1,-1.9,5.5], [-1.5,2.4,3.5], [0.5,-7.9,5.6], [5.9,2.3,-5.8]])
print(input_data)# printing with rect bracets 

data_bin=preprocessing.Binarizer(threshold=0.5).transform(input_data)# convert data into binary format
print("\n binarzied data ",data_bin)

print("\nmean=",input_data.mean(axis=0))# finding mean of array data 
print("\nstd deviation=",input_data.std(axis=0))#finding standard deviation

#removing mean and standard deviation
data_scaled =preprocessing.scale(input_data)
print("\nmean=",data_scaled.mean(axis=0))
print("\nstd devi=",data_scaled.std(axis=0))
# min max  scaling 
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print ("\nMin max scaled data:\n", data_scaled_minmax)
#normalization---------2 types l1,l2
#n1=sum of the absolute values is always up to 1 in each row     least absolute
data_normalization_l1 =preprocessing.normalize(input_data,norm='l1')
print("\n Normalized data ::\n",data_normalization_l1)
#n2=least square    sum of squareis always upto 1 in each row 
data_normalization_l2 =preprocessing.normalize(input_data,norm='l2')
print("\n L2 Normalized data ::\n",data_normalization_l2)