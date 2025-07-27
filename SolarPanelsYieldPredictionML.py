import sklearn as sk 
from scipy import stats 
import pandas as pd 
import numpy as np 



from scipy.stats import ttest_1samp #We are going to perform this hypothesis test between each of our x features  (csv columns) and our y label or target variable if our p value exceeds 0.05 

dfGenerationData = pd.read_csv("Plant_1_Generation_Data.csv") #We obtain the pandas' dataframe from our csv file


#We are going to clean the data before performing our hypothesis test and then running the model using XGBoost 

#There will be three steps 1st: data cleaning, 2nd: hypothesis testing and 3rd: model running 





#1st step: data cleaning starts here and it consists of cleaning the following data features: This is the link explaining data cleaning https://medium.com/@aditib259/data-cleaning-in-python-how-to-handle-missing-values-outliers-more-8f8b68b12436

#1st: Missing values, 2nd: Outliers, 3rd: duplicates, 4th: inconsistent formatting, 5th: incorrect data types and more 



#1st: Missing values cleaning starts here 


print(dfGenerationData.isnull().sum()) #We start by counting the total amount of null values in our data set (now converted into a pandas' data frame)

print("\n \n")

print(dfGenerationData.info()) #We show the column types and names and see if there are missing values this way too by checking non null values.

print("\n \n")
for column in ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]:

    print("Column name: " + column + " \n" + "Amount of zero values out of total in this column: \n" + str(len(dfGenerationData[column] == 0)) + "Zero values out of a total of " + str(len(dfGenerationData[column])) + " total values")


dfGenerationData.drop(["DC_POWER", "AC_POWER", "DAILY_YIELD"], axis='columns', inplace=True) 

print("\n \n")

print(dfGenerationData.info())

#After printing these values on the console I've found there are no null values in any of the columns 

print("\n \n")
 

#2nd: Outliers cleaning starts here 


dfGenerationData = dfGenerationData[(stats.zscore(dfGenerationData[["TOTAL_YIELD"]]) < 3)] #We apply zscore to filter out outliers by checking if the standard deviations for our columns' values lie under 3 

print(dfGenerationData.info())

#After printing the the info of our table after filtering for standard deviations less than three in each column¡s values (called the zscore method) to filter out outliers from our dataset we find there are no outliers 

print("\n \n")

#3rd Duplicates cleaning starts here 

print(dfGenerationData.duplicated().sum())

dfGenerationData.drop_duplicates()

#4th: Inconsistent formatting iis not necessary to check for since we have no text data 

#5th Incorrect data types 

#Let's convert Source key to string and date time to date time 

dfGenerationData['DATE_TIME'] =pd.to_datetime(dfGenerationData["DATE_TIME"])