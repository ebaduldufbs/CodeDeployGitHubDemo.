
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

import pandas as pd # our main data management package
import matplotlib.pyplot as plt # our main display package
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # our model
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


df = pd.read_csv("C:/Users/Besitzer/Downloads/new_housing_cap.csv")
housing = df
#print(housing.info())
#housing.hist(bins=50,figsize=(10,10))
#plt.show()
#print(housing.count())
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
#split train and test data
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#create a pipeline
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])

#define search
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )
#define x and y
y_train = train_set['price']
x_train = train_set.drop ( columns = ['price'])
#conduct search
search.fit(x_train,y_train)
#see best performance
search.best_params_

#search for best coefficients
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)