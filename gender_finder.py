
#importing library for naivebayes accuracy and names 
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names 
#defining letter N as extract feature 
def extract_features(word,N=2):
    last_n_letters=word[-N:]
    return{'feature':last_n_letters.lower()}
#Create the training data using labeled names alredy available in male file 
if __name__=='__main__':
    male_list=[(name,'male') for name in names.words('male.txt')]
    female_list=[(name,'female') for name in names.words('female.txt')]
    data=(male_list+female_list)
    random.seed(5)
    random.shuffle(data)
#data to be tested on
namesInput=['rajesh','gaurav','swati','shubha']
#declaring train and test data 
train_sample=int(0.8*len(data))

for i in range(1,6):
    print("\n number of end letters:",i)
    features=[(extract_features(n,i),gender)for (n,gender)in data]#feature exraction for n with gender 
    train_data,test_data=features[:train_sample],features[train_sample:]
    classifier=NaiveBayesClassifier.train(train_data)#defining classifier
    accuracy_classifier=round(100*nltk_accuracy(classifier,test_data),2)#accuracy of classifier called as this 
    print('accuracy='+str(accuracy_classifier)+'%')
    for name in namesInput:#classified with listing name 
        print(name,'==>',classifier.classify(extract_features(name,1)))