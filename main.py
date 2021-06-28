
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
df = pd.read_csv("C:/Users/Besitzer/Desktop/Research/housing.csv")
housing = df
#print(housing.info())


import matplotlib.pyplot as plt

#plt.scatter(housing.bathrooms, housing.price)
#plt.title('bathrooms vs. price')
#plt.xlabel('bathrooms')
#plt.ylabel('price')
#plt.show()

#print(housing.boxplot(column=['bathrooms']))


#detecting outliers IQR technique for price
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['price'].quantile(0.25)
percentile75 = housing['price'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['price'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['price'] > upper_limit]
housing[housing['price'] < lower_limit]
#Trimming
new_housing = housing[housing['price'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['price'])
plt.subplot(2,2,2)
sns.boxplot(housing['price'])
plt.subplot(2,2,3)
sns.distplot(new_housing['price'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['price'])
#plt.show()

# Capping
new_housing_cap = housing.copy()
new_housing_cap['price'] = np.where(
    new_housing_cap['price'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['price'] < lower_limit,
        lower_limit,
        new_housing_cap['price']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['price'])
plt.subplot(2,2,2)
sns.boxplot(housing['price'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['price'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['price'])
#plt.show()

#detecting outliers IQR technique for bedrooms
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['bedrooms'].quantile(0.25)
percentile75 = housing['bedrooms'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['bedrooms'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['bedrooms'] > upper_limit]
housing[housing['bedrooms'] < lower_limit]
#Trimming
new_housing = housing[housing['bedrooms'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['bedrooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['bedrooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing['bedrooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['bedrooms'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['bedrooms'] = np.where(
    new_housing_cap['bedrooms'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['bedrooms'] < lower_limit,
        lower_limit,
        new_housing_cap['bedrooms']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['bedrooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['bedrooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['bedrooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['bedrooms'])
plt.show()



#detecting outliers IQR technique for bathrooms
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['bathrooms'].quantile(0.25)
percentile75 = housing['bathrooms'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['bathrooms'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['bathrooms'] > upper_limit]
housing[housing['bathrooms'] < lower_limit]
#Trimming
new_housing = housing[housing['bathrooms'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['bathrooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['bathrooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing['bathrooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['bathrooms'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['bathrooms'] = np.where(
    new_housing_cap['bathrooms'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['bathrooms'] < lower_limit,
        lower_limit,
        new_housing_cap['bathrooms']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['bathrooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['bathrooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['bathrooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['bathrooms'])
plt.show()

#detecting outliers IQR technique for reception_rooms
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['reception_rooms'].quantile(0.25)
percentile75 = housing['reception_rooms'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['reception_rooms'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['reception_rooms'] > upper_limit]
housing[housing['reception_rooms'] < lower_limit]
#Trimming
new_housing = housing[housing['reception_rooms'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['reception_rooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['reception_rooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing['reception_rooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['reception_rooms'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['reception_rooms'] = np.where(
    new_housing_cap['reception_rooms'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['reception_rooms'] < lower_limit,
        lower_limit,
        new_housing_cap['reception_rooms']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['reception_rooms'])
plt.subplot(2,2,2)
sns.boxplot(housing['reception_rooms'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['reception_rooms'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['reception_rooms'])
plt.show()
#detecting outliers IQR technique for broadband_speed
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['broadband_speed'].quantile(0.25)
percentile75 = housing['broadband_speed'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['broadband_speed'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['broadband_speed'] > upper_limit]
housing[housing['broadband_speed'] < lower_limit]
#Trimming
new_housing = housing[housing['broadband_speed'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['broadband_speed'])
plt.subplot(2,2,2)
sns.boxplot(housing['broadband_speed'])
plt.subplot(2,2,3)
sns.distplot(new_housing['broadband_speed'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['broadband_speed'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['broadband_speed'] = np.where(
    new_housing_cap['broadband_speed'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['broadband_speed'] < lower_limit,
        lower_limit,
        new_housing_cap['broadband_speed']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['broadband_speed'])
plt.subplot(2,2,2)
sns.boxplot(housing['broadband_speed'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['broadband_speed'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['broadband_speed'])
plt.show()

#detecting outliers IQR technique for latitude
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['latitude'].quantile(0.25)
percentile75 = housing['latitude'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['latitude'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['latitude'] > upper_limit]
housing[housing['latitude'] < lower_limit]
#Trimming
new_housing = housing[housing['latitude'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['latitude'])
plt.subplot(2,2,2)
sns.boxplot(housing['latitude'])
plt.subplot(2,2,3)
sns.distplot(new_housing['latitude'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['latitude'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['latitude'] = np.where(
    new_housing_cap['latitude'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['latitude'] < lower_limit,
        lower_limit,
        new_housing_cap['latitude']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['latitude'])
plt.subplot(2,2,2)
sns.boxplot(housing['latitude'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['latitude'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['latitude'])
plt.show()

#detecting outliers IQR technique for longitude
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['longitude'].quantile(0.25)
percentile75 = housing['longitude'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['longitude'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['longitude'] > upper_limit]
housing[housing['longitude'] < lower_limit]
#Trimming
new_housing = housing[housing['longitude'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['longitude'])
plt.subplot(2,2,2)
sns.boxplot(housing['longitude'])
plt.subplot(2,2,3)
sns.distplot(new_housing['longitude'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['longitude'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['longitude'] = np.where(
    new_housing_cap['longitude'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['longitude'] < lower_limit,
        lower_limit,
        new_housing_cap['longitude']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['longitude'])
plt.subplot(2,2,2)
sns.boxplot(housing['longitude'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['longitude'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['longitude'])
plt.show()
#detecting outliers IQR technique for images
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['images'].quantile(0.25)
percentile75 = housing['images'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['images'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['images'] > upper_limit]
housing[housing['images'] < lower_limit]
#Trimming
new_housing = housing[housing['images'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['images'])
plt.subplot(2,2,2)
sns.boxplot(housing['images'])
plt.subplot(2,2,3)
sns.distplot(new_housing['images'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['images'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['images'] = np.where(
    new_housing_cap['images'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['images'] < lower_limit,
        lower_limit,
        new_housing_cap['images']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['images'])
plt.subplot(2,2,2)
sns.boxplot(housing['images'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['images'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['images'])
plt.show()

#detecting outliers IQR technique for page_views_30_days
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['page_views_30_days'].quantile(0.25)
percentile75 = housing['page_views_30_days'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['page_views_30_days'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['page_views_30_days'] > upper_limit]
housing[housing['page_views_30_days'] < lower_limit]
#Trimming
new_housing = housing[housing['page_views_30_days'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['page_views_30_days'])
plt.subplot(2,2,2)
sns.boxplot(housing['page_views_30_days'])
plt.subplot(2,2,3)
sns.distplot(new_housing['page_views_30_days'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['page_views_30_days'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['page_views_30_days'] = np.where(
    new_housing_cap['page_views_30_days'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['page_views_30_days'] < lower_limit,
        lower_limit,
        new_housing_cap['page_views_30_days']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['page_views_30_days'])
plt.subplot(2,2,2)
sns.boxplot(housing['page_views_30_days'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['page_views_30_days'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['page_views_30_days'])
plt.show()


#detecting outliers IQR technique for page_views_total
import seaborn as sns
import numpy as np
#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
#sns.boxplot(housing['bathrooms'])
#plt.show()

#Finding the IQR
percentile25 = housing['page_views_total'].quantile(0.25)
percentile75 = housing['page_views_total'].quantile(0.75)
#defining iqr
q75, q25 = np.percentile(housing['page_views_total'], [75 ,25])
iqr = q75 - q25
#Finding upper and lower limit
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding Outliers
housing[housing['page_views_total'] > upper_limit]
housing[housing['page_views_total'] < lower_limit]
#Trimming
new_housing = housing[housing['page_views_total'] < upper_limit]
new_housing.shape
# Compare the plots after trimming
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['page_views_total'])
plt.subplot(2,2,2)
sns.boxplot(housing['page_views_total'])
plt.subplot(2,2,3)
sns.distplot(new_housing['page_views_total'])
plt.subplot(2,2,4)
sns.boxplot(new_housing['page_views_total'])
plt.show()
# Capping
new_housing_cap = housing.copy()
new_housing_cap['page_views_total'] = np.where(
    new_housing_cap['page_views_total'] > upper_limit,
    upper_limit,
    np.where(
        new_housing_cap['page_views_total'] < lower_limit,
        lower_limit,
        new_housing_cap['page_views_total']
    )
)

# Compare the plots after capping

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(housing['page_views_total'])
plt.subplot(2,2,2)
sns.boxplot(housing['page_views_total'])
plt.subplot(2,2,3)
sns.distplot(new_housing_cap['page_views_total'])
plt.subplot(2,2,4)
sns.boxplot(new_housing_cap['page_views_total'])
plt.show()


new_housing_cap.to_csv(r'C:\Users\Besitzer\Desktop\new_housing_cap.csv', index = False)


