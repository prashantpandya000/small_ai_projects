#predict whether a given sentence belongs to the category email, news, sports, computer, etc.



import nltk
from sklearn.datasets import fetch_20newsgroups#library for datasets
from sklearn.naive_bayes import MultinomialNB#multinomialnm naive bayes classifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#categorizing data to tell what means what
category_map = {'talk.religion.misc':'Religion','rec.autos':'Autos','rec.sport.hockey':'Hockey','sci.electronics':'Electronics', 'sci.space': 'Space'}
#training data with category data setting shuffle true and state change every 5th element
training_data =fetch_20newsgroups(subset='train',categories=category_map.keys(),shuffle=True,random_state=5)

vectorizer_count =CountVectorizer()
train_tc=vectorizer_count.fit_transform(training_data.data)#extract term count
print("\n Dimensions of traianing data:",train_tc.shape)
#transformer is created 
tfidf=TfidfTransformer()
train_tfidf=tfidf.fit_transform(train_tc)
#input testing data storing
input_data = [
   'Discovery was a space shuttle',
   'Hindu, Christian, Sikh all are religions',
   'We must have to drive safely',
   'Puck is a disk made of rubber',
   'Television, Microwave, Refrigrated all uses electricity'
]
classifier =MultinomialNB().fit(train_tfidf,training_data.target)#naive bayes calling for train data
input_tc=vectorizer_count.transform(input_data)#transform it into 
input_tfidf=tfidf.transform(input_tc)#transform vectorized data 
predections = classifier.predict(input_tfidf)

for sent, category in zip(input_data,predections):
   print('\nInput Data:', sent, '\n Category:',category_map[training_data.target_names[category]])