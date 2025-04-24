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

# 1) Baseline Model : Seasonal Naive

# Convert date column to datetime format
test_df['date'] = pd.to_datetime(test_df['date'])
train_df['date'] = pd.to_datetime(train_df['date'])

# Now subtract one year from test_df using DateOffset
dates = test_df['date'] - pd.DateOffset(years=1)

# Get sales from train_df for these adjusted dates
seasonal_naive_sales = train_df[train_df['date'].isin(dates)]['sales'] 

# Make a copy of the test_df and make naive predictions for the last 3 months of 2017
sn_pred_df = test_df.copy().drop('sales', axis=1)
sn_pred_df['seasonal_naive_sales'] = pd.DataFrame(seasonal_naive_sales).set_index(test_df.index)
print(sn_pred_df.head())

# Plotting the sales forecasting using Seasonal Naive model 

plt.figure(figsize=(20,10))

plt.plot(train_df['date'] , train_df['sales'] , label = 'Train')
plt.plot(test_df['date'] , test_df['sales'] , label = 'Test')

plt.plot(sn_pred_df['date'] , sn_pred_df['seasonal_naive_sales'] , label = 'Forecast-Seasonal Naive')

plt.legend(loc = 'best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Baseline Model: Seasonal Naive')
plt.show()

#Evaluating the Model: metrics like mean absolute error (MAE), root mean squared error (RMSE) and mean absolute percentage error (MAPE) are used.

# Merge predictions and actual sales into one df 
errors_df = pd.merge(test_df , sn_pred_df , on= 'date')
errors_df = errors_df[['date' , 'sales' , 'seasonal_naive_sales']]

# Model Evaluation
errors_df['errors'] = test_df['sales'] - sn_pred_df['seasonal_naive_sales']
errors_df.insert(0 , 'model' , 'Seasonal Naive')

def mae(err):
    return np.mean(np.abs(err))

def rmse(err):
    return np.sqrt(np.mean(err**2))

def mape(err , sales= errors_df['sales']):
    return np.sum(np.abs(err))/np.sum(sales) * 100

result_df = errors_df.groupby('model').agg(total_sales = ('sales' , 'sum'),
                                          total_sn_pred_sales = ('seasonal_naive_sales' , 'sum'),
                                          overall_error = ('errors' , 'sum'),
                                          MAE = ('errors' , mae),
                                          RMSE = ('errors' , rmse),
                                          MAPE = ('errors' , mape))
print(result_df)

# Plotting Seasonal Naive forecasts with actual sales and errors
plt.figure(figsize=(20,10))

plt.plot(errors_df['date'] , np.abs(errors_df['errors']) , label = 'errors')
plt.plot(errors_df['date'] , errors_df['sales'] , label = 'actual sales')
plt.plot(errors_df['date'] , errors_df['seasonal_naive_sales'] , label = 'forecast')

plt.legend(loc= 'best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Seasonal Naive forecasts with actual sales and errors')
plt.show()

# Time Series Decomposition Plot: allows us to observe the seasonality, trend, and error/residual terms of a time series.

# Creating a time series decomposition plot for training data  and settting date as index 
ts_decomp_df = train_df.set_index('date')
ts_decomp_df['sales'] = ts_decomp_df['sales'].astype(float)
print(ts_decomp_df.head())

# Plotting the seasonal decomposition
result = seasonal_decompose(ts_decomp_df['sales'], model='additive')
fig = result.plot()               # Plot the decomposition
fig.set_size_inches(20, 10)       # Set the figure size
plt.tight_layout()
plt.show()

# 2) Holt Winter's Triple Exponential Smoothing Model

hw_train_df = train_df[['date' , 'sales']].set_index('date')
hw_test_df = test_df[['date' , 'sales']].set_index('date')

# Apply Triple Exponential Smoothing(without Damping)
hw_model_1 = ExponentialSmoothing(hw_train_df , trend = 'additive' , seasonal = 'additive' , seasonal_periods = 12, freq ='D' )
hw_fit_1 = hw_model_1.fit( remove_bias = False)
pred_fit_1 = pd.Series(hw_fit_1.predict(start = hw_test_df.index[0] , end = hw_test_df.index[-1]) ,name='pred_sales').reset_index()
print('Forecasts made , ready for evaluation')

# Merge predictions and actual sales into one df (without Damping)

errors_df_hw = pd.merge(test_df , pred_fit_1 , left_on= 'date' , right_on= 'index')
errors_df_hw = errors_df_hw[['date', 'sales' , 'pred_sales']]
errors_df_hw['errors'] = errors_df_hw['sales'] - errors_df_hw['pred_sales']
errors_df_hw.insert(0 , 'model' , 'Holt-Winters')


# Model Evaluation
result_df_hw = errors_df_hw.groupby('model').agg(total_sales=('sales' , 'sum'),
                                                total_pred_sales=('pred_sales' , 'sum'),
                                                holt_winters_overall_error=('errors' , 'sum'),
                                                MAE=('errors' , mae),
                                                RMSE=('errors' , rmse),
                                                MAPE=('errors' , mape))
print(result_df_hw)

# Plotting the sales forecast for Holt-Winters without damping trend component
plt.figure(figsize=(20,10))
plt.plot(train_df['date'] , train_df['sales'] , label= 'Train')
plt.plot(test_df['date'] , test_df['sales'] , label= 'Test')
plt.plot(errors_df_hw['date'] , errors_df_hw['pred_sales'] , label= 'Forecast - HW no damping')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Holt-Winters without damping trend component')
plt.show()

# Plotting Holt-Winters(without Damp) forecasts with actual sales and errors
plt.figure(figsize=(20,10))
plt.plot(errors_df_hw['date'] , np.abs(errors_df_hw['errors']) , label= 'errors')
plt.plot(errors_df_hw['date'] , errors_df_hw['sales'] , label= 'actual sales')
plt.plot(errors_df_hw['date'] , errors_df_hw['pred_sales'] , label= 'forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Holt-Winters forecasts with actual sales and error')
plt.show()

# Apply Triple Exponential Smoothing(with Damping)
hw_model_2 = ExponentialSmoothing(hw_train_df , trend = 'additive' , seasonal = 'additive' , seasonal_periods =12 ,freq='D', damped=True)
hw_fit_2 = hw_model_2.fit(  remove_bias = False)
pred_fit_2 = pd.Series(hw_fit_2.predict(start= hw_test_df.index[0], end=hw_test_df.index[-1]), name='pred_sales').reset_index()

print('Forecasts made , ready for evaluation')

# Merge predictions and actual sales into one df (with Damping)
errors_df_hwd_damp= pd.merge(test_df , pred_fit_2 , left_on='date' , right_on= 'index')
errors_df_hwd_damp= errors_df_hwd_damp[['date' , 'sales' , 'pred_sales']]
errors_df_hwd_damp['errors'] = errors_df_hwd_damp['sales'] - errors_df_hwd_damp['pred_sales']
errors_df_hwd_damp.insert(0, 'model' , 'Holt-Winters-Damped')

# Model Evaluation 
result_df_hwd_damp = errors_df_hwd_damp.groupby('model').agg(total_sales=('sales' , 'sum'),
                                                  total_pred_sales=('pred_sales' , 'sum'),
                                                  holt_winters_overall_error=('errors' , 'sum'),
                                                  MAE=('errors' , mae),
                                                  RMSE=('errors' , rmse),
                                                  MAPE=('errors' , mape))
print(result_df_hwd_damp)

# Plotting the sales forecast for Holt-Winters with damping trend component
plt.figure(figsize=(20,10))
plt.plot(train_df['date'] , train_df['sales'] , label='Train')
plt.plot(test_df['date'] , test_df['sales'] , label='Test')
plt.plot(errors_df_hwd_damp['date'] , errors_df_hwd_damp['pred_sales'] , label='Forecast - HW damping')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Holt-Winters with damping trend component')
plt.show()

# Plotting Holt-Winters(Damped) forecasts with actual sales and errors
plt.figure(figsize=(14,7))
plt.plot(errors_df_hwd_damp['date'], np.abs(errors_df_hwd_damp['errors']), label='errors')
plt.plot(errors_df_hwd_damp['date'], errors_df_hwd_damp['sales'], label='actual sales')
plt.plot(errors_df_hwd_damp['date'], errors_df_hwd_damp['pred_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Holt-Winters (damping) forecasts with actual sales and errors')
plt.show()

# 3) Autoregressive Integrated Moving Average - ARIMA Model

# Step-1: Check Stationarity: Plotting rolling statistics, Augmented Dickey-Fuller Test, ACF and PACF plots
arima_df= train_df[['date' , 'sales']].set_index('date')
arima_test_df = test_df[['date' , 'sales']].set_index('date')

def test_stationarity(timeseries):
    
    
    
    # Plotting rolling statistics
    rollmean = timeseries.rolling(window=365).mean()
    rollstd = timeseries.rolling(window=365).std()
    
    plt.figure(figsize=(20,10))
    plt.plot(timeseries , color='skyblue' , label='Original Series')
    plt.plot(rollmean , color='black' , label='Rolling Mean')
    plt.plot(rollstd , color='red' , label='Rolling Std')
    plt.legend(loc='best')
    plt.xlabel('date')
    plt.ylabel('sales')
    plt.title('Rolling Statistics')
    plt.show()
    
    # Augmented Dickey-Fuller Test
    adfuller_test= adfuller(timeseries , autolag= 'AIC')
    print('Test Statistic = {:.3f}'.format(adfuller_test[0]))
    print('P-value = {:.3f}'.format(adfuller_test[1]))
    print('Critical values:')
    
    for key,value in adfuller_test[4].items():
        print ('\t{}: {} - The data is {} stationary with {}% confidence'
              .format(key, value , '' if adfuller_test[0] < value else 'not' , 100-int(key[:-1])))
        
        
    # Autocorrelation plots
    fig, ax = plt.subplots(2, figsize=(20,10))
    plot_acf(timeseries, ax=ax[0], lags=20)
    plot_pacf(timeseries, ax=ax[1], lags=20)
    ax[0].set_title('Autocorrelation (ACF)')
    ax[1].set_title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.show()
    
test_stationarity(arima_df.sales)

# Step-2: Differencing: Seasonal or cyclical patterns can be removed by substracting periodical values
first_difference = arima_df.sales - arima_df.sales.shift(1)
first_difference = pd.DataFrame(first_difference.dropna(inplace = False))

# Check for stationarity after differencing 
test_stationarity(first_difference.sales)

# Step 3: Model Building: After determining the AR(p), I(d), MA(q) values
# Fit the ARIMA model
arima_model61 = ARIMA(arima_df.sales , order=(6,1,1) ).fit()
print(arima_model61.summary())

# Plotting the residuals using ACF and PACF and checking for seasonality
residuals = arima_model61.resid
fig, ax = plt.subplots(2, figsize=(20,10))
plot_acf(residuals, ax=ax[0], lags=20)
plot_pacf(residuals, ax=ax[1], lags=20)
ax[0].set_title('Autocorrelation (ACF)')
ax[1].set_title('Partial Autocorrelation (PACF)')
plt.tight_layout()
plt.show()

# Fit the SARIMA model

sarima_model = SARIMAX(arima_df.sales, order=(6, 1, 0), seasonal_order=(6, 1, 0, 7), enforce_invertibility=False, enforce_stationarity=False)
sarima_fit = sarima_model.fit()
arima_test_df['pred_sales'] = sarima_fit.predict(start=arima_test_df.index[0],end=arima_test_df.index[-1], dynamic= True)
sarima_fit.plot_diagnostics(figsize=(14,7))
plt.tight_layout()
plt.show()

# Model Evaluation
arima_test_df['errors'] = arima_test_df.sales - arima_test_df.pred_sales
arima_test_df.insert(0, 'model', 'SARIMA')
result_df_sarima = arima_test_df.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          SARIMA_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
print(result_df_sarima)

# Plotting sales forecast using SARIMA Model 
plt.figure(figsize=(14,7))
plt.plot(train_df['date'], train_df['sales'], label='Train')
plt.plot(arima_test_df.index, arima_test_df['sales'], label='Test')
plt.plot(arima_test_df.index, arima_test_df['pred_sales'], label='Forecast - SARIMA')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Seasonal ARIMA (SARIMA) model')
plt.show()

# Plotting the sales forecast with actual sales and error using SARIMA Model 
plt.figure(figsize=(14,7))
plt.plot(arima_test_df.index, np.abs(arima_test_df['errors']), label='errors')
plt.plot(arima_test_df.index, arima_test_df['sales'], label='actual sales')
plt.plot(arima_test_df.index, arima_test_df['pred_sales'], label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Seasonal ARIMA (SARIMA) forecasts with actual sales and errors')
plt.show()

# Linear Regression
reg_df = df
print(reg_df.head())

# Step-1: Feature Engineering: Creating Lag Feature
for i in range(1,8):
    lag_i = 'lag_' + str(i)
    reg_df[lag_i] = reg_df.sales.shift(i)


# Rolling window
reg_df['rolling_mean'] = reg_df.sales.rolling(window=7).mean()
reg_df['rolling_max'] = reg_df.sales.rolling(window=7).max()
reg_df['rolling_min'] = reg_df.sales.rolling(window=7).min()

reg_df = reg_df.dropna(how='any' , inplace= False)
reg_df = reg_df.drop(['store' , 'item'] , axis=1)

# Split the series to predict the last 3 months of 2017
reg_df = reg_df.set_index('date')
reg_train_df = reg_df.loc[:'2017-09-30']
reg_test_df = reg_df.loc['2017-10-1':]

# Step-2: Feature Selection 
# Correlation matrix with heatmap
corr = reg_train_df.corr()
fig = plt.figure(figsize=(8,6))
_ = sns.heatmap(corr, linewidths=.2)

# Defining independent and dependent variables
X_train = reg_train_df.drop(['sales'] , axis=1)
y_train = reg_train_df['sales'].values

X_test = reg_test_df.drop(['sales'] , axis=1)
y_test = reg_test_df['sales'].values

# Univariate SelectKBest class to extract top 5 best features
top_features = SelectKBest(score_func=f_regression, k=5)
fit = top_features.fit(X_train , y_train)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.columns)

# Concat two dataframes for better visualization
feature_scores = pd.concat([df_columns , df_scores], axis=1)

# Naming the dataframe columns
feature_scores.columns = ['Feature' , 'Score']

# Print 5 Best Features
print(feature_scores.nlargest(5,'Score'))

# Checking for a linear relationship of the top features with sales (target variable)

fig, axs = plt.subplots(ncols=2, figsize=(14,7))
sns.scatterplot(x= reg_train_df['rolling_mean'], y= reg_train_df['sales'] , ax=axs[0])
axs[0].set(title='Linear relationship between sales and rolling_mean of sales')
sns.scatterplot(x= reg_train_df['rolling_max'], y= reg_train_df['sales'] , ax=axs[1])
axs[1].set(title='Linear relationship between sales and rolling_max of sales')

fig, axs = plt.subplots(ncols=2, figsize=(14,7))
sns.scatterplot(x= reg_train_df['rolling_min'], y= reg_train_df['sales'] , ax=axs[0])
axs[0].set(title='Linear relationship between sales and rolling_min of sales')
sns.scatterplot(x= reg_train_df['lag_7'], y= reg_train_df['sales'] , ax=axs[1])
_ = axs[1].set(title='Linear relationship between sales and lag_7 of sales')

# Step-3: Model Building

# Update X_train, X_test to include top features
X_train = X_train[['rolling_mean', 'rolling_max', 'rolling_min', 'lag_7', 'lag_1']]
X_test = X_test[['rolling_mean', 'rolling_max', 'rolling_min', 'lag_7', 'lag_1']]

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Step-4: Model Evaluation
errors_df = reg_test_df[['sales']]
errors_df.loc[:, 'pred_sales'] = preds
errors_df.loc[:, 'errors'] = preds - y_test
errors_df.insert(0, 'model', 'LinearRegression')

result_df_lr = errors_df.groupby('model').agg(total_sales=('sales', 'sum'),
                                          total_pred_sales=('pred_sales', 'sum'),
                                          LR_overall_error=('errors', 'sum'),
                                          MAE=('errors', mae),
                                          RMSE=('errors', rmse), 
                                          MAPE=('errors', mape))
print(result_df_lr)

# Step-4:Plotting Sales Forecast using Linear Regression
fig = plt.figure(figsize=(14,7))
plt.plot(reg_train_df.index, reg_train_df['sales'], label='Train')
plt.plot(reg_test_df.index, reg_test_df['sales'], label='Test')
plt.plot(errors_df.index, errors_df['pred_sales'], label='Forecast - Linear Regression')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Forecasts using Linear Regression model')
plt.show()

fig = plt.figure(figsize=(14,7))
plt.plot(errors_df.index, errors_df.errors, label='errors')
plt.plot(errors_df.index, errors_df.sales, label='actual sales')
plt.plot(errors_df.index, errors_df.pred_sales, label='forecast')
plt.legend(loc='best')
plt.xlabel('date')
plt.ylabel('sales')
plt.title('Linear Regression forecasts with actual sales and errors')
plt.show()

