# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Loading transformed dataset
df = pd.read_csv('./data/cleaned/train_transformed.csv')
print(df.head())

# Splitting data into train and test set
temp_df = df.set_index('date')
train_df = temp_df.loc[ : '2017-09-30'].reset_index(drop= False)
test_df = temp_df.loc['2017-10-01' :].reset_index(drop=False)
print(train_df.tail())
print(test_df.head())

# VISUALIZATION 
# Aggregate data
monthly_agg = df.groupby('month')['sales'].sum().reset_index()
yearly_agg = df.groupby('year')['sales'].sum().reset_index()

# Create subplots
fig,axs = plt.subplots(nrows=2, ncols=3, figsize=(12,10))
fig.suptitle('Sales Analysis', fontsize = 16)

#Weekly sales(boxplot)
sns.boxplot(x= 'weekday', y= 'sales', data = df, ax = axs[0,0])
axs[0,0].set_title('Weekly Sales Distribution')

# Monthly sales(boxplot)
sns.boxplot(x= 'month', y= 'sales', data= df, ax= axs[0,1], order= range(1,13))
axs[0,1].set_title('Monthly Sales Distribution')

# Yearly sales(boxplot)
sns.boxplot(x='year', y= 'sales', data= df, ax= axs[0,2])
axs[0,2].set_title('Yearly Sales Distribution')

# Monthly aggregated sales(lineplot)
sns.lineplot(x= 'month', y= 'sales', data= monthly_agg, ax= axs[1,0])
axs[1,0].set_title('Monthly Aggregated Sales')

# Yearly aggregated sales(lineplot)
sns.lineplot(x='year', y= 'sales', data= yearly_agg, ax= axs[1,1])
axs[1,1].set_title('Yearly Aggregated Sales')

# Daily sales(lineplot)
sns.lineplot(x='date', y= 'sales', data = df, ax= axs[1,2])
axs[1,2].set_title('Daily Sales Distribution')

# Adjust layout
plt.tight_layout(rect=[0,0,1,1])
plt.show()

# REGRESSION ANALYSIS
# Baseline Model : Seasonal Naive  