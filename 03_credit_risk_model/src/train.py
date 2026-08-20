import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score,brier_score_loss

d=fetch_openml('credit-g',version=1,as_frame=True)
X=d.data; y=(d.target.astype(str).str.lower()=='good').astype(int)
cat=X.select_dtypes(include=['object','category']).columns; num=X.columns.difference(cat)
prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
base=Pipeline([('prep',prep),('clf',LogisticRegression(max_iter=3000,class_weight='balanced'))])
model=CalibratedClassifierCV(base,method='sigmoid',cv=5)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]
print({'roc_auc':roc_auc_score(yte,p),'brier':brier_score_loss(yte,p)})