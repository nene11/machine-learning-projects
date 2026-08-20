from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
url='https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'; zp=DATA/'bank_marketing.zip'
if not zp.exists(): zp.write_bytes(urlopen(url,timeout=60).read())
with ZipFile(zp) as z:
    with z.open('bank-full.csv') as f: df=pd.read_csv(f,sep=';')
df=df.replace('unknown',pd.NA); y=df.pop('y').eq('yes').astype(int)
# duration is observed after the call; excluding it prevents target leakage for pre-contact targeting.
df=df.drop(columns=['duration'])
Xtr,Xte,ytr,yte=train_test_split(df,y,test_size=.2,stratify=y,random_state=42)
num=Xtr.select_dtypes(include='number').columns; cat=Xtr.columns.difference(num)
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
model=Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=3000,class_weight='balanced'))])
cv=StratifiedKFold(5,shuffle=True,random_state=42); cv_auc=cross_val_score(model,Xtr,ytr,cv=cv,scoring='roc_auc')
model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]; prec,rec,thr=precision_recall_curve(yte,p); f1s=2*prec*rec/(prec+rec+1e-12); i=f1s[:-1].argmax(); t=float(thr[i])
results={'dataset':'UCI Bank Marketing','rows':len(df),'features':df.shape[1],'positive_rate':float(y.mean()),'cv_roc_auc_mean':float(cv_auc.mean()),'cv_roc_auc_std':float(cv_auc.std()),'test_roc_auc':float(roc_auc_score(yte,p)),'test_pr_auc':float(average_precision_score(yte,p)),'test_f1_optimized':float(f1_score(yte,p>=t)),'threshold':t,'leakage_control':'duration excluded'}
(ROOT/'results_real.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
