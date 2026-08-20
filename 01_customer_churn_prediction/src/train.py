import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

URL='https://raw.githubusercontent.com/IBM/employee-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df=pd.read_csv(URL)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['Churn']=(df['Churn'].astype(str).str.strip()=='Yes').astype(int)
df=df.drop(columns=['customerID'])
X,y=df.drop(columns='Churn'),df['Churn']
cat=X.select_dtypes(include='object').columns
num=X.columns.difference(cat)
prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
model=Pipeline([('prep',prep),('clf',HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]
print({'roc_auc':roc_auc_score(yte,p),'pr_auc':average_precision_score(yte,p)})