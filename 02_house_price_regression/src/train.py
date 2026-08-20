import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data=fetch_openml(name='house_prices',as_frame=True)
df=data.frame
X=df.drop(columns=['SalePrice','Id'],errors='ignore')
y=np.log1p(df['SalePrice'].astype(float))
cat=X.select_dtypes(include=['object','category']).columns
num=X.columns.difference(cat)
prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
model=Pipeline([('prep',prep),('reg',ElasticNet(alpha=.001,l1_ratio=.5,max_iter=5000,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr); p=model.predict(Xte)
print({'rmse':mean_squared_error(yte,p)**.5,'mae':mean_absolute_error(yte,p),'r2':r2_score(yte,p)})