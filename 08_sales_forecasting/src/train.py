import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
rng=np.random.default_rng(42); n=1200
dates=pd.date_range('2022-01-01',periods=n,freq='D')
y=200+0.08*np.arange(n)+25*np.sin(2*np.pi*np.arange(n)/7)+rng.normal(0,10,n)
df=pd.DataFrame({'date':dates,'sales':y}).set_index('date')
for lag in [1,7,14,28]: df[f'lag_{lag}']=df.sales.shift(lag)
df['roll7']=df.sales.shift(1).rolling(7).mean(); df['dow']=df.index.dayofweek; df=df.dropna()
cut=int(len(df)*.8); features=[c for c in df if c!='sales']
m=HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,random_state=42)
m.fit(df.iloc[:cut][features],df.iloc[:cut].sales); p=m.predict(df.iloc[cut:][features])
print({'mae':mean_absolute_error(df.iloc[cut:].sales,p),'rmse':mean_squared_error(df.iloc[cut:].sales,p)**.5})
