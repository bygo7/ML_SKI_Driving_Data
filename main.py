import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

pd.options.display.max_rows = 80
pd.options.display.max_columns = 80
# data pre-processing
df = pd.read_csv('./Process_data.csv')
df = df.drop('Unnamed: 66', axis = 1)
df["x62"] = df['x62'].str.strip("%")
df["x62"] = df["x62"].astype('float')

df_date = df['Date']
df = df.set_index("Date")

# splitting train_data and test_data
train_data = df.iloc[0:691,:] #17년 12월 31일
test_data = df.iloc[691:,:] #18년 4월 22일

train_data.isnull().sum()
test_data.isnull().sum()







