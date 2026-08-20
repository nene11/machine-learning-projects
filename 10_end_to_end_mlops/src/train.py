from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d=load_breast_cancer(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=.2,stratify=d.target,random_state=42)
model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000))]); model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]; print({'roc_auc':roc_auc_score(yte,p)})
