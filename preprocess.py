# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Loading the dataset
df = pd.read_csv('./data/raw/train.csv')
print(df.head())

# Checking for missing values
null = df.isnull().sum()
print(null)

# Filtering data for store = 1 and item = 1
df = df[df['store'] == 1]
df = df[df['item'] == 1]

# Formatting date to datetime 
df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')

# Feature engineering: date related features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday +1 
print(df.head())

# Uploading preprocessed data to cleaned folder 
upload = df.to_csv('./data/cleaned/train_transformed.csv', index= False)
print(upload)
