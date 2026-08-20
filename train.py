import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

X,y=make_classification(n_samples=12000,n_features=25,n_informative=12,weights=[.75,.25],random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced'))])
model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]
print({'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p),'f1':f1_score(yte,p>.5)})
