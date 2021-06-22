
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
import nltk
import stopwords
#install and download all nltk packages
#nltk.download('stopwords')to download stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
#Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_pickle("C:/Users/Besitzer/Downloads/df_london.pkl")
housing = df
#print(housing.info())
#housing.hist(bins=50,figsize=(10,10))
#plt.show()
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

#housing data cleaning
#check for missing value
#check_for_nulls(housing)
#print(check_for_nulls(housing))
#list of variables with missing value: ['price', 'description', 'bedrooms', 'bathrooms', 'reception_rooms', 'area',
# 'property_type', 'broadband_speed', 'images', 'page_views_30_days', 'page_views_total']

#cleaning variable price
show_na_sum(housing, 'price')
#print(show_na_sum(housing, 'price'))
housing['price'].value_counts()
#print(housing['price'].value_counts())
# null values in price replaced with mean value
fix_na(housing, 'price', housing['price'].mean())

#cleaning variable description
show_na_sum(housing, 'description')
#print(show_na_sum(housing, 'description'))
housing['description'].value_counts()
#print(housing['description'].value_counts())
# # Because the houses where description is listed are without description and description is
# object type data,I will fill with None
fix_na(housing, 'description', 'None')



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
# Because the houses where reception_rooms are listed as 'None' ,may or may not have reception_rooms
# I will fill these values with none.
fix_na(housing,'reception_rooms', 'NaN')
#fix_na(housing, 'reception_rooms', housing['reception_rooms'].mean())

#cleaning variable 'area'
show_na_sum(housing, 'area')
#print(show_na_sum(housing, 'area'))
housing['area'].value_counts()
#print(housing['area'].value_counts())
# Because the houses where area is listed as 'None' , and area is float64 data type, a property must have area
# measurement , I will fill these values with mean.
fix_na(housing, 'area', housing['area'].mean())

#cleaning variable property_type
show_na_sum(housing, 'property_type')
#print(show_na_sum(housing, 'property_type'))
housing['property_type'].value_counts()
#print(housing['property_type'].value_counts())
# # Because the houses where property_type is listed as none are without information and property_type is
# object type data,I will fill with None
fix_na(housing, 'property_type', 'None')

#cleaning variable 'broadband_speed'
show_na_sum(housing, 'broadband_speed')
#print(show_na_sum(housing, 'broadband_speed'))
housing['broadband_speed'].value_counts()
#print(housing['broadband_speed'].value_counts())
# Because the houses where broadband_speed is listed as 'None' ,may or may not have broadband
# I will fill these values with mean.
fix_na(housing,'broadband_speed', 'NaN')
#fix_na(housing, 'area', housing['area'].mean())

#cleaning variable images
show_na_sum(housing, 'images')
#print(show_na_sum(housing, 'images'))
housing['images'].value_counts()
#print(housing['images'].value_counts())
# # Because the houses where images are listed as none may be were without images abd images are
#object type data,I will fill with None
fix_na(housing, 'images', 'NaN')

#cleaning variable 'page_views_30_days'
show_na_sum(housing, 'page_views_30_days')
#print(show_na_sum(housing, 'page_views_30_days'))
housing['page_views_30_days'].value_counts()
#print(housing['page_views_30_days'].value_counts())
# Because the houses where page_views_30_days are listed as 'None' ,may or may not have page_views_30_days
#I will fill these values with none.
fix_na(housing,'page_views_30_days', 'NaN')
#fix_na(housing, 'page_views_30_days', housing['page_views_30_days'].mean())

#cleaning variable 'page_views_total'
show_na_sum(housing, 'page_views_total')
#print(show_na_sum(housing, 'page_views_total'))
housing['page_views_total'].value_counts()
#print(housing['page_views_total'].value_counts())
# Because the houses where page_views_total is listed as 'None' ,may or may not have page_views_total
# I will fill these values with none.
fix_na(housing,'page_views_total', 'NaN')
#fix_na(housing, 'page_views_total', housing['page_views_total'].mean())
#check_for_nulls(housing)
#all nulls values have been cleaned

#converting object to float
housing["images"]=housing.images.astype(float)
housing["floorplan_images"]=housing.floorplan_images.astype(float)
housing["reception_rooms"]=housing.reception_rooms.astype(float)
housing["broadband_speed"]=housing.broadband_speed.astype(float)
housing["page_views_30_days"]=housing.page_views_30_days.astype(float)
housing["page_views_total"]=housing.page_views_total.astype(float)

#Check for Unusual Values
#to get full display
#pd.set_option('display.width', None)
#print(housing.describe())
#seems to be fine

#identifying outliers  Z-score treatment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#variables price
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['price'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['price'].mean() + 3*housing['price'].std())
print("Lowest allowed",housing['price'].mean() - 3*housing['price'].std())
# Finding the Outliers
housing[(housing['price'] > 9672099.59751944) | (housing['price'] < -7494418.528186293)]
print(housing[(housing['price'] > 9672099.59751944) | (housing['price'] < -7494418.528186293)])
# Trimming of Outliers
new_housing = housing[(housing['price'] <9672099.59751944) & (housing['price'] >-7494418.528186293) ]
new_housing
#Capping on Outliers
upper_limit = housing['price'].mean() + 3*housing['price'].std()
lower_limit = housing['price'].mean() - 3*housing['price'].std()

#Now, apply the Capping
housing['price'] = np.where(
    housing['price']>upper_limit,
    upper_limit,
    np.where(
        housing['price']<lower_limit,
        lower_limit,
        housing['price']
    )
)

#now see the statistics using “Describe” Function
#print(housing['price'].describe())

#variables id
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['id'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['id'].mean() + 3*housing['id'].std())
print("Lowest allowed",housing['id'].mean() - 3*housing['id'].std())
# Finding the Outliers
housing[(housing['id'] > 62992921.066320665) | (housing['id'] < 49401833.92240808)]
print(housing[(housing['id'] > 62992921.066320665) | (housing['id'] < 49401833.92240808)])
# Trimming of Outliers
new_housing = housing[(housing['id'] < 62992921.066320665) & (housing['id'] > 49401833.92240808) ]
new_housing
#Capping on Outliers
upper_limit = housing['id'].mean() + 3*housing['id'].std()
lower_limit = housing['id'].mean() - 3*housing['id'].std()

#Now, apply the Capping
housing['id'] = np.where(
    housing['id']>upper_limit,
    upper_limit,
    np.where(
        housing['id']<lower_limit,
        lower_limit,
        housing['id']
    )
)

#now see the statistics using “Describe” Function
#print(housing['id'].describe())

#variables bedrooms
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['bedrooms'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['bedrooms'].mean() + 3*housing['bedrooms'].std())
print("Lowest allowed",housing['bedrooms'].mean() - 3*housing['bedrooms'].std())
# Finding the Outliers
housing[(housing['bedrooms'] > 9.855728467954904) | (housing['bedrooms'] < -2.7750682229347983)]
print(housing[(housing['bedrooms'] > 9.855728467954904) | (housing['bedrooms'] < -2.7750682229347983)])
# Trimming of Outliers
new_housing = housing[(housing['bedrooms'] < 9.855728467954904) & (housing['bedrooms'] > -2.7750682229347983) ]
new_housing
#Capping on Outliers
upper_limit = housing['bedrooms'].mean() + 3*housing['bedrooms'].std()
lower_limit = housing['bedrooms'].mean() - 3*housing['bedrooms'].std()

#Now, apply the Capping
housing['bedrooms'] = np.where(
    housing['bedrooms']>upper_limit,
    upper_limit,
    np.where(
        housing['bedrooms']<lower_limit,
        lower_limit,
        housing['bedrooms']
    )
)

#now see the statistics using “Describe” Function
#print(housing['bedrooms'].describe())

#variables bathrooms
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['bathrooms'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['bathrooms'].mean() + 3*housing['bathrooms'].std())
print("Lowest allowed",housing['bathrooms'].mean() - 3*housing['bathrooms'].std())
# Finding the Outliers
housing[(housing['bathrooms'] > 4.874131989406412) | (housing['bathrooms'] < -1.1898728489317485)]
print(housing[(housing['bathrooms'] > 4.874131989406412) | (housing['bathrooms'] < -1.1898728489317485)])
# Trimming of Outliers
new_housing = housing[(housing['bathrooms'] < 4.874131989406412) & (housing['bathrooms'] > -1.1898728489317485) ]
new_housing
#Capping on Outliers
upper_limit = housing['bathrooms'].mean() + 3*housing['bathrooms'].std()
lower_limit = housing['bathrooms'].mean() - 3*housing['bathrooms'].std()

#Now, apply the Capping
housing['bathrooms'] = np.where(
    housing['bathrooms']>upper_limit,
    upper_limit,
    np.where(
        housing['bathrooms']<lower_limit,
        lower_limit,
        housing['bathrooms']
    )
)

#now see the statistics using “Describe” Function
#print(housing['bathrooms'].describe())


#variables area
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['area'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['area'].mean() + 3*housing['area'].std())
print("Lowest allowed",housing['area'].mean() - 3*housing['area'].std())
# Finding the Outliers
housing[(housing['area'] > 6274.071170197954) | (housing['area'] < -1531.5400363689582)]
print(housing[(housing['area'] > 6274.071170197954) | (housing['area'] < -1531.5400363689582)])
# Trimming of Outliers
new_housing = housing[(housing['area'] < 6274.071170197954) & (housing['area'] > -1531.5400363689582) ]
new_housing
#Capping on Outliers
upper_limit = housing['area'].mean() + 3*housing['area'].std()
lower_limit = housing['area'].mean() - 3*housing['area'].std()

#Now, apply the Capping
housing['area'] = np.where(
    housing['area']>upper_limit,
    upper_limit,
    np.where(
        housing['area']<lower_limit,
        lower_limit,
        housing['area']
    )
)

#now see the statistics using “Describe” Function
#print(housing['area'].describe())



#variables latitude
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['latitude'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['latitude'].mean() + 3*housing['latitude'].std())
print("Lowest allowed",housing['latitude'].mean() - 3*housing['latitude'].std())
# Finding the Outliers
housing[(housing['latitude'] > 53.51051912588337) | (housing['latitude'] < 49.47075574696455)]
print(housing[(housing['latitude'] > 53.51051912588337) | (housing['latitude'] < 49.47075574696455)])
# Trimming of Outliers
new_housing = housing[(housing['latitude'] < 53.51051912588337) & (housing['latitude'] > 49.47075574696455) ]
new_housing
#Capping on Outliers
upper_limit = housing['latitude'].mean() + 3*housing['latitude'].std()
lower_limit = housing['latitude'].mean() - 3*housing['latitude'].std()

#Now, apply the Capping
housing['latitude'] = np.where(
    housing['latitude']>upper_limit,
    upper_limit,
    np.where(
        housing['latitude']<lower_limit,
        lower_limit,
        housing['latitude']
    )
)

#variables longitude
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['longitude'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['longitude'].mean() + 3*housing['longitude'].std())
print("Lowest allowed",housing['longitude'].mean() - 3*housing['longitude'].std())
# Finding the Outliers
housing[(housing['area'] > 3.0888681278217294) | (housing['area'] < -3.370506235514862)]
print(housing[(housing['longitude'] > 3.0888681278217294) | (housing['longitude'] < -3.370506235514862)])
# Trimming of Outliers
new_housing = housing[(housing['longitude'] < 3.0888681278217294) & (housing['longitude'] > -3.370506235514862) ]
new_housing
#Capping on Outliers
upper_limit = housing['longitude'].mean() + 3*housing['longitude'].std()
lower_limit = housing['longitude'].mean() - 3*housing['longitude'].std()

#Now, apply the Capping
housing['longitude'] = np.where(
    housing['longitude']>upper_limit,
    upper_limit,
    np.where(
        housing['longitude']<lower_limit,
        lower_limit,
        housing['longitude']
    )
)

#now see the statistics using “Describe” Function
#print(housing['longitude'].describe())


#variables images
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['images'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['images'].mean() + 3*housing['images'].std())
print("Lowest allowed",housing['images'].mean() - 3*housing['images'].std())
# Finding the Outliers
housing[(housing['images'] > 32.14688039686595) | (housing['images'] < -4.378262260775413)]
print(housing[(housing['images'] > 32.14688039686595) | (housing['images'] < -4.378262260775413)])
# Trimming of Outliers
new_housing = housing[(housing['images'] < 32.14688039686595) & (housing['images'] > -4.378262260775413) ]
new_housing
#Capping on Outliers
upper_limit = housing['images'].mean() + 3*housing['images'].std()
lower_limit = housing['images'].mean() - 3*housing['images'].std()
#Now, apply the Capping
housing['images'] = np.where(
    housing['images']>upper_limit,
    upper_limit,
    np.where(
        housing['images']<lower_limit,
        lower_limit,
        housing['images']
    )
)

#variables images
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['floorplan_images'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['floorplan_images'].mean() + 3*housing['floorplan_images'].std())
print("Lowest allowed",housing['floorplan_images'].mean() - 3*housing['floorplan_images'].std())
# Finding the Outliers
housing[(housing['floorplan_images'] > 2.39838441144179) | (housing['floorplan_images'] < -0.4897554387884273)]
print(housing[(housing['floorplan_images'] > 2.39838441144179) | (housing['floorplan_images'] < -0.4897554387884273)])
# Trimming of Outliers
new_housing = housing[(housing['floorplan_images'] < 2.39838441144179) & (housing['floorplan_images'] > -0.4897554387884273) ]
new_housing
#Capping on Outliers
upper_limit = housing['floorplan_images'].mean() + 3*housing['floorplan_images'].std()
lower_limit = housing['floorplan_images'].mean() - 3*housing['floorplan_images'].std()
#Now, apply the Capping
housing['floorplan_images'] = np.where(
    housing['floorplan_images']>upper_limit,
    upper_limit,
    np.where(
        housing['floorplan_images']<lower_limit,
        lower_limit,
        housing['floorplan_images']
    )
)

#variables reception_rooms
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['reception_rooms'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['reception_rooms'].mean() + 3*housing['reception_rooms'].std())
print("Lowest allowed",housing['reception_rooms'].mean() - 3*housing['reception_rooms'].std())
# Finding the Outliers
housing[(housing['reception_rooms'] > 4.251186257803757) | (housing['reception_rooms'] < -0.7159719911016011)]
print(housing[(housing['reception_rooms'] > 4.251186257803757) | (housing['reception_rooms'] < -0.7159719911016011)])
# Trimming of Outliers
new_housing = housing[(housing['reception_rooms'] < 4.251186257803757) & (housing['reception_rooms'] > -0.7159719911016011) ]
new_housing
#Capping on Outliers
upper_limit = housing['reception_rooms'].mean() + 3*housing['reception_rooms'].std()
lower_limit = housing['reception_rooms'].mean() - 3*housing['reception_rooms'].std()

#Now, apply the Capping
housing['reception_rooms'] = np.where(
    housing['reception_rooms']>upper_limit,
    upper_limit,
    np.where(
        housing['reception_rooms']<lower_limit,
        lower_limit,
        housing['reception_rooms']
    )
)

#now see the statistics using “Describe” Function
#print(housing['reception_rooms'].describe())

#variables broadband_speed
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['broadband_speed'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['broadband_speed'].mean() + 3*housing['broadband_speed'].std())
print("Lowest allowed",housing['broadband_speed'].mean() - 3*housing['broadband_speed'].std())
# Finding the Outliers
housing[(housing['broadband_speed'] > 370.6585485629011) | (housing['broadband_speed'] < -38.05874573042138)]
print(housing[(housing['broadband_speed'] > 370.6585485629011) | (housing['broadband_speed'] < -38.05874573042138)])
# Trimming of Outliers
new_housing = housing[(housing['broadband_speed'] < 370.6585485629011) & (housing['broadband_speed'] > -38.05874573042138) ]
new_housing
#Capping on Outliers
upper_limit = housing['broadband_speed'].mean() + 3*housing['broadband_speed'].std()
lower_limit = housing['broadband_speed'].mean() - 3*housing['broadband_speed'].std()

#Now, apply the Capping
housing['broadband_speed'] = np.where(
    housing['broadband_speed']>upper_limit,
    upper_limit,
    np.where(
        housing['broadband_speed']<lower_limit,
        lower_limit,
        housing['broadband_speed']
    )
)

#now see the statistics using “Describe” Function
#print(housing['broadband_speed'].describe())

#variables page_views_30_days
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['page_views_30_days'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['page_views_30_days'].mean() + 3*housing['page_views_30_days'].std())
print("Lowest allowed",housing['page_views_30_days'].mean() - 3*housing['page_views_30_days'].std())
# Finding the Outliers
housing[(housing['page_views_30_days'] > 1613.3912734811147) | (housing['page_views_30_days'] < -1103.935375500243)]
print(housing[(housing['page_views_30_days'] > 1613.3912734811147) | (housing['page_views_30_days']
                                                                      < -1103.935375500243)])
# Trimming of Outliers
new_housing = housing[(housing['page_views_30_days'] < 1613.3912734811147) & (housing['page_views_30_days'] >
                                                                              -1103.935375500243) ]
new_housing
#Capping on Outliers
upper_limit = housing['page_views_30_days'].mean() + 3*housing['page_views_30_days'].std()
lower_limit = housing['page_views_30_days'].mean() - 3*housing['page_views_30_days'].std()

#Now, apply the Capping
housing['page_views_30_days'] = np.where(
    housing['page_views_30_days']>upper_limit,
    upper_limit,
    np.where(
        housing['page_views_30_days']<lower_limit,
        lower_limit,
        housing['page_views_30_days']
    )
)

#now see the statistics using “Describe” Function
#print(housing['page_views_30_days'].describe())

#variables page_views_total
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(housing['page_views_total'])
#plt.show()
#Finding the Boundary Values
print("Highest allowed",housing['page_views_total'].mean() + 3*housing['page_views_total'].std())
print("Lowest allowed",housing['page_views_total'].mean() - 3*housing['page_views_total'].std())
# Finding the Outliers
housing[(housing['page_views_total'] > 10750.144716176903) | (housing['page_views_total'] < -7323.241902590983)]
print(housing[(housing['page_views_total'] > 10750.144716176903) | (housing['page_views_total'] < -7323.241902590983)])
# Trimming of Outliers
new_housing = housing[(housing['page_views_total'] < 10750.144716176903) & (housing['page_views_total'] > -7323.241902590983) ]
new_housing
#Capping on Outliers
upper_limit = housing['page_views_total'].mean() + 3*housing['page_views_total'].std()
lower_limit = housing['page_views_total'].mean() - 3*housing['page_views_total'].std()
#Now, apply the Capping
housing['page_views_total'] = np.where(
    housing['page_views_total']>upper_limit,
    upper_limit,
    np.where(
        housing['page_views_total']<lower_limit,
        lower_limit,
        housing['page_views_total']
    )
)
#now see the statistics using “Describe” Function
#print(housing['page_views_total'].describe())

print(housing.info())

#export clean data
housing.to_csv(r'C:\Users\Besitzer\Desktop\housing.csv', index = False)







