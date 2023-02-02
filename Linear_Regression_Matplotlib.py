import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


fn = '/Users/samuelrussell/Documents/GitHub/Data_Visualisation/Stuttgart_station_temp_data.csv'
df = pd.read_csv(fn)
print('The length of the dataframe is:', len(df))
print()
print(df.head(10))

# Clean (replace no data value of 999.9 with np.nan)

df = df.replace(999.9, value=np.nan)
print(df.isna().sum())

# Creating Variables
years = df['YEAR'].to_numpy(copy=True)
met = df['metANN'].to_numpy(copy=True)
jja = df['J-J-A'].to_numpy(copy=True)
djf = df['D-J-F'].to_numpy(copy=True)

# smoothing
df['metANN_MA3'] = df.loc[:, 'metANN'].rolling(window=3).mean()
met_smooth = df['metANN_MA3'].to_numpy(copy=True)

# get the index of non-NaN values (a.k.a. "finite" values)
idx_notNaN = np.isfinite(met_smooth)

# use the index to extract our x and y values for
x = years[idx_notNaN]
y = met_smooth[idx_notNaN]
slope, intercept, r, p, _ = stats.linregress(x, y)

# Line estimate
n= x.size
n= x.size
y_est = intercept + slope * x
resid = y - y_est
r2 = r**2
ssr = np.sum((resid) ** 2)
rmse = np.sqrt(ssr / (n-2))

print('r: %2.3f'%r)
print('Slope:%2.3f'%slope)
print('R^2:%2.3f'%r2)
print('p:',p)
print('RMSE:%2.3f'%rmse)
print('SSR:%2.3f'%ssr)

#Make a plot 
fig, ax = plt.subplots(figsize=(12, 6)) 
ax.plot(x, y, marker='s', color='gray', ls='', markersize=5, label='raw data') 
ax.plot(x, y_est, 'r-', linewidth=2, label='Fit (y = {:0.2f} + {:0.2f} x)'.format(intercept, slope)) 
ax.set_title("Stuttgart Schnarrenberg Station, Linear Regression \n \ r= 0.90, R^2= 0.81, RMSE= 0.28, p-value: 3.51e-44") 
ax.set_xlabel('Year') 
ax.set_ylabel('Temperature °C') 
ax.grid() 
ax.legend() 

# Saving Output
plt.savefig('/Users/samuelrussell/Documents/GitHub/Data_Visualisation/Linear_Regression_Stuttgart_Schnarrenberg.png')
