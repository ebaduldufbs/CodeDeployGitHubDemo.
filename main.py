
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
#data set without textual variables
df = pd.read_csv("C:/Users/Besitzer/Downloads/new_housing_cap (1).csv")
housing_final = df

#seperate train and test date
from sklearn.model_selection import train_test_split, cross_val_score
train_set, test_set = train_test_split(housing_final, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

#lr_prelim = LinearRegression()
#model=lr_prelim.fit(x_1, y_1)
#print(model)

import statsmodels.api as sm

#define response variable
y = train_set['price']
#y = test_set['price']
#define explanatory variable
x = train_set.drop ( columns = ['price'])
#x = test_set.drop ( columns = ['price'])
#x = test_set.drop ( columns = ['price','floorplan_images','images','latitude'])
#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

#for prediction
new_x = test_set.drop ( columns = ['price'])
new_x = sm.add_constant(new_x)  # sm2 = statsmodels.api
y_predict = model.predict(new_x)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
y_test= test_set['price']
#MAE
print(metrics.mean_absolute_error(y_test, y_predict))
# MSE
print(metrics.mean_squared_error(y_test, y_predict))
# RMSE
import numpy as np
print(np.sqrt(metrics.mean_squared_error(y_test, y_predict)))




#data set with textual variables
df = pd.read_csv("C:/Users/Besitzer/Downloads/new_housing_cap.csv")
housing_final = df
#print(housing_final.info())

#seperate train and test date
from sklearn.model_selection import train_test_split, cross_val_score
train_set, test_set = train_test_split(housing_final, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

#lr_prelim = LinearRegression()
#model=lr_prelim.fit(x_1, y_1)
#print(model)

import statsmodels.api as sm

#define response variable
y = train_set['price']
#y = test_set['price']
#define explanatory variable
x = train_set.drop ( columns = ['price'])
#x = test_set.drop ( columns = ['price'])
#x = test_set.drop ( columns = ['price','floorplan_images','images','latitude'])
#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

#for prediction
new_x = test_set.drop ( columns = ['price'])
new_x = sm.add_constant(new_x)  # sm2 = statsmodels.api
y_predict = model.predict(new_x)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
y_test= test_set['price']
#MAE
print(metrics.mean_absolute_error(y_test, y_predict))
# MSE
print(metrics.mean_squared_error(y_test, y_predict))
# RMSE
import numpy as np
print(np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

