import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Download UCI household_power_consumption.txt and place beside this script.
df=pd.read_csv('household_power_consumption.txt',sep=';',na_values='?',low_memory=False)
df['timestamp']=pd.to_datetime(df['Date']+' '+df['Time'],dayfirst=True,errors='coerce')
df['power']=pd.to_numeric(df['Global_active_power'],errors='coerce')
df=df[['timestamp','power']].dropna().set_index('timestamp').resample('1h').mean().dropna()
for lag in [1,2,3,24,48,168]: df[f'lag_{lag}']=df['power'].shift(lag)
df['roll_24']=df['power'].shift(1).rolling(24).mean(); df['hour']=df.index.hour; df['dow']=df.index.dayofweek
df=df.dropna(); cut=int(len(df)*.8); tr,te=df.iloc[:cut],df.iloc[cut:]
features=[c for c in df.columns if c!='power']
model=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=42)
model.fit(tr[features],tr['power']); p=model.predict(te[features])
print({'mae':mean_absolute_error(te['power'],p),'rmse':mean_squared_error(te['power'],p)**.5})