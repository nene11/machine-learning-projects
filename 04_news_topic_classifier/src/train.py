from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score

tr=fetch_20newsgroups(subset='train',remove=('headers','footers','quotes'))
te=fetch_20newsgroups(subset='test',remove=('headers','footers','quotes'))
model=Pipeline([('tfidf',TfidfVectorizer(sublinear_tf=True,min_df=2,max_df=.95,ngram_range=(1,2),max_features=120000)),('clf',LinearSVC(C=2.0))])
model.fit(tr.data,tr.target); pred=model.predict(te.data)
print({'accuracy':accuracy_score(te.target,pred),'macro_f1':f1_score(te.target,pred,average='macro')})