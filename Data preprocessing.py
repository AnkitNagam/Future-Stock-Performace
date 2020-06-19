import pandas as pd
from scipy import stats
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 6)


file = pd.read_csv('E:/Documents/Advanced Machine Learning/project aml/2018_Financial_Data.csv', delimiter=',',quotechar='"')


file.head()
file.describe()
file.shape
file.columns

file.rename(columns={'Unnamed: 0': 'company'}, inplace=True)

#checking the data types for each variable
file.dtypes #sector, company are categorical rest are float

#analyzing target feature
round(file.Class.value_counts()/file.shape[0]*100,2) #there are 70% having 1's and 30% having 0's

#inorganic growth - removing companies who's price variation is above 300%
pricevar_300per = file.index[file['2019 PRICE VAR [%]']>300].tolist()
file.drop(pricevar_300per,axis=0,inplace = True)

#checking for null values
nulls = file.isnull().sum().sort_values(ascending=False)
nulls_per = round(file.isnull().sum()/file.shape[0]*100,2).sort_values(ascending=False)

miss_threshold = 35
nan_col_below35 = []
for i in file:
    if file[i].isnull().sum()/file[i].shape[0]*100 > miss_threshold:
        nan_col_below35.append(i)
        
file_below35 = file.drop(nan_col_below35,axis =1)
file_below35.shape

#round(file_below35.isnull().sum()/file_below35.shape[0]*100).sort_values().sort_values(ascending=False)


#filewo_nans = file_below35.dropna()

#filewo_nans.shape


#checking for zero's
round(file_below35.isin([0]).sum()/file_below35.shape[0]*100,2).sort_values(ascending=False)

zero_threshold = 55

zero_col_below55 = []
for column in file_below35:
    if file_below35[column].isin([0]).sum()/file_below35[column].shape[0]*100 > zero_threshold:
        zero_col_below55.append(column)
print(zero_col_below55)
    
    
filewo_nan_zer = file_below35.drop(zero_col_below55,axis =1)
filewo_nan_zer.shape

#round(filewo_nan_zer.isin([0]).sum()/filewo_nan_zer.shape[0]*100,2).sort_values(ascending=False)


sum(filewo_nan_zer.isnull().sum().sort_values())



#outlier detection using Z-scores 

filewo_nan_zer_objs = filewo_nan_zer.drop(['Sector','company'],axis = 1)
z_scores = np.abs(stats.zscore(filewo_nan_zer_objs.notnull()))
#print(z_scores)

filewo_nan_zer_ext = filewo_nan_zer[(z_scores < 3).all(axis=1)]
filewo_nan_zer_ext.shape #0 rows were considered i.e. 

#outlier detection using Inter quartile range

Q1 = filewo_nan_zer.quantile(0.25)
Q3 = filewo_nan_zer.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)
iqr = filewo_nan_zer[~((filewo_nan_zer < (Q1 - 1.5 * IQR)) |(filewo_nan_zer > (Q3 + 1.5 * IQR))).any(axis=1)]
iqr.shape  #only 25 sample were considered out of 4329



#Coerce. Replaces outliers and extreme values with the nearest value that would not be considered extreme. 
#For example if an outlier is defined to be anything above or below three standard deviations, 
#then all outliers would be replaced with the highest or lowest value within this range.
upper_outliers = filewo_nan_zer.quantile(0.97)
outliers_top = (filewo_nan_zer > upper_outliers)

lower_outliers = filewo_nan_zer.quantile(0.03)
outliers_low = (filewo_nan_zer < lower_outliers)

filewo_nan_zer_out = filewo_nan_zer.mask(outliers_top, upper_outliers, axis=1)
filewo_nan_zer_out = filewo_nan_zer_out.mask(outliers_low, lower_outliers, axis=1)


filewo_nz_outliers = pd.concat([filewo_nan_zer.quantile(1), upper_outliers,filewo_nan_zer.quantile(0),lower_outliers], 
                      axis=1, keys=['before_1','After_0.97','before_0','After_0.3'])

filewo_nan_zer.describe()
filewo_nz_outliers.describe()


#finally replacing remaining missing values with their means according to each sector

final_file = filewo_nan_zer_out.groupby(['Sector']).transform(lambda x: x.fillna(x.mean()))
final_file.isnull().sum() #zero nulls

round(final_file.Class.value_counts()/final_file.shape[0]*100,2)

# data exploration 
corr = final_file.corr() > 0.7


sns.heatmap(final_file)


final_file.to_csv("E:\Documents\Advanced Machine Learning\project aml\cleaned.csv",index = False)

final_file.shape
final_file.iloc[:,199]
