import bs4
import inline as inline
import matplotlib
import pandas as pd
import sklearn
assert sklearn.__version__>="0.20"
import sklearn.impute
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sys
assert sys.version_info >= (3, 5)
import os
import inline
#to plot pretty figures
#matplotlib inline
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import tarfile
import urllib
import urllib.request
from pandas.plotting import scatter_matrix
from scipy.stats import randint
import joblib

#define data set
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
df= pd.read_pickle("C:/Users/Besitzer/Downloads/df_london.pkl")
housing= df= pd.read_pickle("C:/Users/Besitzer/Downloads/df_london.pkl")
# delecting textual features
housing= housing.drop(columns= ['id','description','url','feature_items','property_type'])

#housing.info()
#print(housing.info())
#print(housing["feature_items"])

# data cleaning
def check_for_nulls(df):
    '''
    Iterates over columns in the data set
    Creates a list of columns with missing values
    '''
    missing_list = []

    for column in df.columns:
        if df[column].isna().sum() > 0:
            missing_list.append(column)

    return missing_list


def show_na_sum(df, column):
    '''
    Shows a count of missing values in a specific column
    '''
    return df[column].isna().sum()


def fix_na(df, column, value):
    '''
    Fill missing data points with a specific function
    '''
    df[column] = df[column].fillna(value)

    def change_dtype(df, column, map_fxn):
        '''
        Convert a column to a new data type
        '''
        df[column] = df[column].map(map_fxn)


#converting object to float
housing["price"]=housing.images.astype(float)
housing["bedrooms"]=housing.images.astype(float)
housing["bathrooms"]=housing.images.astype(float)
housing["area"]=housing.images.astype(float)
housing["images"]=housing.images.astype(float)
housing["floorplan_images"]=housing.floorplan_images.astype(float)
housing["reception_rooms"]=housing.reception_rooms.astype(float)
housing["broadband_speed"]=housing.broadband_speed.astype(float)
housing["page_views_30_days"]=housing.page_views_30_days.astype(float)
housing["page_views_total"]=housing.page_views_total.astype(float)


#cleaning variable price
show_na_sum(housing, 'price')
#print(show_na_sum(housing, 'price'))
housing['price'].value_counts()
#print(housing['price'].value_counts())
# null values in price replaced with mean value
fix_na(housing, 'price', housing['price'].mean())


#cleaning variable bedrooms
show_na_sum(housing, 'bedrooms')
#print(show_na_sum(housing, 'bedrooms'))
housing['bedrooms'].value_counts()
#print(housing['bedrooms'].value_counts())
# Because the houses where bedrooms are listed as 'None' must have at least one bedrooms, I will fill these
# values with mean.
fix_na(housing, 'bedrooms', housing['bedrooms'].mean())


#cleaning variable bathrooms
show_na_sum(housing, 'bathrooms')
#print(show_na_sum(housing, 'bathrooms'))
housing['bathrooms'].value_counts()
#print(housing['bathrooms'].value_counts())
# Because the houses where bathrooms are listed as 'None'  must have at least one bathrooms,
# I will fill these values with mean.
fix_na(housing, 'bathrooms', housing['bathrooms'].mean())

#cleaning variable 'reception_rooms'
show_na_sum(housing, 'reception_rooms')
#print(show_na_sum(housing, 'reception_rooms'))
housing['reception_rooms'].value_counts()
#print(housing['reception_rooms'].value_counts())
# I will fill these values with none.
fix_na(housing, 'reception_rooms', housing['reception_rooms'].mean())

#cleaning variable 'area'
show_na_sum(housing, 'area')
#print(show_na_sum(housing, 'area'))
housing['area'].value_counts()
#print(housing['area'].value_counts())
# Because the houses where area is listed as 'None' , and area is float64 data type, a property must have area
# measurement , I will fill these values with mean.
fix_na(housing, 'area', housing['area'].mean())


#cleaning variable 'broadband_speed'
show_na_sum(housing, 'broadband_speed')
#print(show_na_sum(housing, 'broadband_speed'))
housing['broadband_speed'].value_counts()
#print(housing['broadband_speed'].value_counts())
# I will fill these values with mean.
fix_na(housing, 'broadband_speed', housing['broadband_speed'].mean())

#cleaning variable images
show_na_sum(housing, 'images')
#print(show_na_sum(housing, 'images'))
housing['images'].value_counts()
#print(housing['images'].value_counts())
# I will fill these values with mean.
fix_na(housing, 'images', housing['images'].mean())


#cleaning variable 'page_views_30_days'
show_na_sum(housing, 'page_views_30_days')
#print(show_na_sum(housing, 'page_views_30_days'))
housing['page_views_30_days'].value_counts()
#print(housing['page_views_30_days'].value_counts())
#I will fill these values with mean .
fix_na(housing, 'page_views_30_days', housing['page_views_30_days'].mean())

#cleaning variable 'page_views_total'
show_na_sum(housing, 'page_views_total')
#print(show_na_sum(housing, 'page_views_total'))
housing['page_views_total'].value_counts()
#print(housing['page_views_total'].value_counts())
# I will fill these values with mean.
fix_na(housing, 'page_views_total', housing['page_views_total'].mean())
check_for_nulls(housing)
print(check_for_nulls(housing))
#all nulls values have been cleaned
