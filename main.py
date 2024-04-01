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

#Multivariable Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressorsquared = regressor.score(X_train, y_train)

print(f'Training data r-squared: {regressorsquared:.2}')


#Evaluate the Coefficients of the Model
reg_coef = pd.DataFrame(data = regressor.coef_, index=X_train.columns, columns =['Coefficient'])
print(reg_coef)


#Analyse the Estimated Values & Regression Residuals
""" How good our regression is depends on both r-squared and residuals"""
predicted_values = regressor.predict(X_train)
residuals = (y_train - predicted_values)


# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predicted_values, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='red')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predicted_values, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()


#Residual Distribution Chart
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color="indigo")
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()