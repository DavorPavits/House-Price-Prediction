#TODO 1:Analyze and explore the Boston house Price Data 
#TODO 2:Split your data for training and testing 
#TODO 3:Run a Multivariable Regression 
#TODO 4:Evalute how your model's coefficients and residuals 
#TODO 5:Use Data transformation to improce your model perfomarnce 
#TODO 6:Use your model to estimate a property price 

import pandas as pd 
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Presentation of Data
pd.options.display.float_format = '{:,.2f}'.format

#Load the Data
""" The first column in the .csv file just has the row numbers,
so will be used as the index."""
data = pd.read_csv("boston.csv", index_col=0)



#Preliminary Data Exploration
# print(data.shape)
# print(data.columns)
# print(data.count())

#Data Cleaning -Check for missing Values and Duplicates
# data.info()   
# print(f"Any NaN values?{data.isna().values.any()}")
# print(f"Any duplicates values?{data.duplicated().values.any()}")


#House Prices
sns.displot(data['PRICE'],
            bins = 50,
            aspect=2,
            kde=True,
            color="#2196f3")
plt.title(f"1970 Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}")
plt.xlabel('Price in 000s')
plt.ylabel('Nr. of Homes')

# plt.show()

#Split Training & Test Dataset
"""We can't use all 506 entries in our dataset to train our model.
The reason is that we want to evaluate our model on data that it hasn't yet
That get a better idea of its performance in the real world."""

target = data["PRICE"]
features = data.drop('PRICE', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size = 0.2,
                                                    random_state=10)

#% of training set
train_pct = 100 * len(X_train)/ len(features)
print(f'Training data is {train_pct:.3}% of the total data.')

#% of test data set
test_pct = 100 *X_test.shape[0]/features.shape[0]
print(f"Test data makes up the remaining {test_pct:0.3}%")
