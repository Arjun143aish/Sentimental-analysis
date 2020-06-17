import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\Sentimental analysis")

comments = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t',quoting = 3)

import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
lem = WordNetLemmatizer()
Corpus = []

for i in range(len(comments['Review'])):
    review = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    message = ' '.join(review)
    Corpus.append(message)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

cv = CountVectorizer(max_features =500)
X = cv.fit_transform(Corpus).toarray()
X = pd.DataFrame(X)

pickle.dump(cv,open('CV.pkl','wb'))
CV = pickle.load(open('CV.pkl','rb'))

Y = comments['Liked']

FullRaw = pd.concat([X,Y], axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['Liked'], axis =1)
Train_Y = Train['Liked'].copy()
Test_X = Test.drop(['Liked'], axis =1)
Test_Y = Test['Liked'].copy()

Model = MultinomialNB().fit(Train_X,Train_Y)

Test_pred = Model.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier(random_state=123).fit(Train_X,Train_Y)

RF_Pred = RF_Model.predict(Test_X)

RF_Mat = confusion_matrix(RF_Pred,Test_Y)
sum(np.diag(RF_Mat))/Test_Y.shape[0]*100

pickle.dump(Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


