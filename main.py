
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv("C:/Users/Besitzer/Downloads/new_housing_cap.csv")
housing_final = df
#print(housing_final.info())

#seperate train and test date
from sklearn.model_selection import train_test_split, cross_val_score
train_set, test_set = train_test_split(housing_final, test_size=0.2, random_state=42)


#linear regression
#x_1= housing.drop ( columns = ['price'])
#y_1= housing['price']

from sklearn.linear_model import LinearRegression

#lr_prelim = LinearRegression()
#model=lr_prelim.fit(x_1, y_1)
#print(model)

import statsmodels.api as sm

#define response variable
#y = train_set['price']
y = test_set['price']
#define explanatory variable
#x = train_set.drop ( columns = ['price'])
x = test_set.drop ( columns = ['price'])
#x = test_set.drop ( columns = ['price','floorplan_images','images','latitude'])
#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())

