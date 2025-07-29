import sklearn as sk 
from scipy import stats 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.

import numpy as np 
import xgboost as xg

from scipy.stats import ttest_1samp #We are going to perform this hypothesis test between each of our x features  (csv columns) and our y label or target variable if our p value exceeds 0.05 

dfGenerationData = pd.read_csv("Plant_1_Generation_Data.csv") #We obtain the pandas' dataframe from our energy generation csv file
dfWeatherSensorData = pd.read_csv("Plant_1_Weather_Sensor_Data.csv") #We obtain the pandas' dataframe from our weather sensor csv file 


#We are going to clean the data before performing our hypothesis test and then running the model using XGBoost 

#There will be three steps 1st: data cleaning, 2nd: hypothesis testing and 3rd: model running 

print("SHOWING INFO FOR EACH DATA FRAME")

print("\n \n")

print("ENERGY GENERATION DATA FRAME: " + str(dfGenerationData.info()))

print("\n \n")

print("WEATHER SENSOR DATA FRAME" + str(dfWeatherSensorData.info()))

print("\n \n")


#1st step: data cleaning starts here and it consists of cleaning the following data features: This is the link explaining data cleaning https://medium.com/@aditib259/data-cleaning-in-python-how-to-handle-missing-values-outliers-more-8f8b68b12436

#1st: Missing values, 2nd: Outliers, 3rd: duplicates, 4th: inconsistent formatting, 5th: incorrect data types and more 



#1st: Missing values cleaning starts here 


print("CLEANING MISSING VALUES")


#First we are going to check if there are any null values next we will check for zero values.
print("\n \n")

print(dfGenerationData.isnull().sum()) #We start by counting the total amount of null values in our data set (now converted into a pandas' data frame)

print("\n \n")

dfGenerationData = dfGenerationData.dropna()

print(dfGenerationData.info()) #We show the column types and names and see if there are missing values this way too by checking non null values.

print("\n \n")

print(dfWeatherSensorData.isnull().sum())

dfWeatherSensorData = dfWeatherSensorData.dropna()
print("\n \n")

print(dfWeatherSensorData.info())

print("\n \n")

#Now we check for zero values 

for column in ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]:

    print("Column name: " + column + " \n" + "Amount of zero values out of total in this column: \n" + str(len(dfGenerationData[column] == 0)) + " Zero values out of a total of " + str(len(dfGenerationData[column])) + " total values")

print("\n \n")

dfGenerationData.drop(["DC_POWER", "AC_POWER", "DAILY_YIELD"], axis="columns", inplace=True) 

print(dfGenerationData.info())

for column in ["DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]:
    print("Column name: " + column + " \n" + "Amount of zero values out of total in this column: \n" + str(len(dfWeatherSensorData[column] == 0)) + " Zero values out of a total of " + str(len(dfWeatherSensorData[column])) + " total values")

dfWeatherSensorData.drop(["IRRADIATION"], axis="columns", inplace=True)

print(dfWeatherSensorData.info())

print("\n \n")

#After printing these values on the console I've found there are no null values in any of the columns but there are zeros

print("\n \n")
 
#2nd: Outliers cleaning starts here 

print("CLEANING OUTLIERS")

print("\n \n")

dfGenerationData = dfGenerationData[(stats.zscore(dfGenerationData[["TOTAL_YIELD"]]) < 3)] #We apply zscore to filter out outliers by checking if the standard deviations for our columns' values lie under 3 

print(dfGenerationData)

print(dfGenerationData.info())

#After printing the the info of our table after filtering for standard deviations less than three in each column¡s values (called the zscore method) to filter out outliers from our dataset we find there are no outliers 

print("\n \n")


dfWeatherSensorData = dfWeatherSensorData[(stats.zscore(dfWeatherSensorData[["AMBIENT_TEMPERATURE"]]) < 3)] #We apply zscore to filter out outliers by checking if the standard deviations for our columns' values lie under 3 

dfWeatherSensorData = dfWeatherSensorData[(stats.zscore(dfWeatherSensorData["MODULE_TEMPERATURE"]) < 3)]


print(dfWeatherSensorData.info())

print("\n \n")

#3rd Duplicates cleaning starts here 

print("REMOVING DUPLICATES")

print("\n \n")

print(dfGenerationData.duplicated().sum())

dfGenerationData.drop_duplicates()

print("\n \n") 

print(dfGenerationData.info())

print("\n \n")

print(dfWeatherSensorData.duplicated().sum())

dfWeatherSensorData.drop_duplicates()

print("\n \n") 

print(dfWeatherSensorData.info())

#4th: Inconsistent formatting iis not necessary to check for since we have no text data 

#5th Incorrect data types 

#Let's convert Source key to string and date time to date time 

dfGenerationData['DATE_TIME'] = pd.to_datetime(dfGenerationData["DATE_TIME"])

#Now we are going to merge the two data frames into a single data frame and for that we need to drop repeated columns between data frames. The ones repeating are the date time and plant id columns, the name of the Source key column must be changed in both dataframes and called SOLAR_PANEL_INVERTER_ID
#in the generation data data frame and SENSOR_INVERTER_ID in the weather sensor data data frame.


dfGenerationData.drop(["DATE_TIME", "PLANT_ID"], axis="columns", inplace=True)

print(dfGenerationData.info())

print("\n \n") 

dfGenerationData.rename(columns={"SOURCE_KEY": "SOLAR_PANEL_INVERTER_ID"}, inplace=True)

print(dfGenerationData.info())

print("\n \n") 

dfWeatherSensorData.rename(columns={"SOURCE_KEY": "SENSOR_INVERTER_ID"}, inplace=True)

print("\n \n")

dfWeatherSensorAndGenerationDataFrameMerge = pd.concat([dfWeatherSensorData, dfGenerationData])

print(dfWeatherSensorAndGenerationDataFrameMerge.info())

dfWeatherSensorAndGenerationDataFrameMerge["DATE_TIME"] = pd.to_datetime(dfWeatherSensorAndGenerationDataFrameMerge["DATE_TIME"])


print(dfWeatherSensorAndGenerationDataFrameMerge.isnull().sum())

#There is an issue. Because our merged tables differ in row count with one having 3182 values and the other having 68779 values what happens is pandas adds extra null values to each table contained in extra rows from the second merged table to the first merged table thus increasing the row count of the table resulting from our merging.
#First we are going to normalize the nummerical data of our columns next we will pair up matching records without nulls in our table.
#Basically we are going to reduce our entries down to 3182 but BEFORE THAT we will place the first 3182 values after the index 3182 from our 70000 entry columns (TOTAL_YIELD AND SOLAR_PANEL_INVERTED_ID which have 70000 entries because they come from the second table having 68.0000 and we are adding the extra rows with null values) 
#Then we impute null values to the mean or zero depending on whether our values are numeric or not. How are we going to do this? 
#That we will show once we have normalized our numeric columns' data 

#Now we are going to normalize the numerical data of each column

#We are going to use the maximum absolute scaling normalization technique 

maxScaled = dfWeatherSensorAndGenerationDataFrameMerge.copy()

for column in maxScaled[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TOTAL_YIELD"]].columns:
    maxScaled[column] = maxScaled[column] / maxScaled[column].abs().max()

print(maxScaled)

print(maxScaled.info())


#Now we are going to create two isolated data frames one for each of the two columns containing 70000 values in our maxScaled data frame which are as we have said the ones belonging to the 680000 rows table after adding extra null value 
 
dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap = maxScaled[["SOLAR_PANEL_INVERTER_ID"]].reset_index(drop=True)

dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap = maxScaled[["TOTAL_YIELD"]].reset_index(drop=True) #We add reset index to avoid the duplicate row label bug from the pandas library

print(dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap)

print("\n \n")

print(dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap)

#The following algorithm which has been commented out is not necessary for this program but it took me so much effort for it to work 

#for rowIndex in range(len(dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap["SOLAR_PANEL_INVERTER_ID"])): #We are going to make duplicates unique since we need to preserve dimensions for when we add these modified columns to our merged table 
    #if ((dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[[rowIndex]].any() == 0).any()): #We do this by assigning duplicate values to their row's index value 
        #dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap.loc[rowIndex, "SOLAR_PANEL_INVERTER_ID"] = rowIndex

#for rowIndex in range(len(dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap["TOTAL_YIELD"])): #We are going to make duplicates unique since we need to preserve dimensions for when we add these modified columns to our merged table 
    #if ((dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[[rowIndex]].any() == 0).any()): #We do this by assigning duplicate values to their row's index value 
        #dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap.loc[rowIndex, "TOTAL_YIELD"] = rowIndex

print("\n \n")

#Now we set the first 3182 rows to the first 3182 rows after index 3182 

for i in range(3182):
    dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[i] = dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[3182 + i]

for i in range(1382):
    dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[i] = dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap.iloc[3182 + i]

#Now we impute the null values from our isolated data frames 
dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap = dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap.fillna(0)
dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap = dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap.fillna(dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap["TOTAL_YIELD"].mean())
print(dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap)
print("\n \n")
print(dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap)


#Now we replace the columns with 70000 values from the maxScaledTable with the values we have assigned to our isolated dataframes (dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap and dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap) that we have modified in our previous loops and whose null values we have imputed with column's mean values or zero values
maxScaled["SOLAR_PANEL_INVERTER_ID"] = dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap["SOLAR_PANEL_INVERTER_ID"]
maxScaled["TOTAL_YIELD"] = dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap["TOTAL_YIELD"]


print("\n \n")

print(maxScaled)

print("\n \n")

print(maxScaled.info())

maxScaled.drop(maxScaled.tail(65595).index, inplace=True)
maxScaled["DATE_TIME"] = maxScaled["DATE_TIME"].fillna(0)
maxScaled["PLANT_ID"] = maxScaled["PLANT_ID"].fillna(0)
maxScaled["SENSOR_INVERTER_ID"] = maxScaled["SENSOR_INVERTER_ID"].fillna(0)
maxScaled["AMBIENT_TEMPERATURE"] = maxScaled["AMBIENT_TEMPERATURE"].fillna(maxScaled["AMBIENT_TEMPERATURE"].mean())
maxScaled["MODULE_TEMPERATURE"] = maxScaled["MODULE_TEMPERATURE"].fillna(maxScaled["MODULE_TEMPERATURE"].mean())
print("\n \n")

print(maxScaled)

print("\n \n")

print(maxScaled.info())
#Next three lines are necessary to fix this error "ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:DATE_TIME: object, SENSOR_INVERTER_ID: object, SOLAR_PANEL_INVERTER_ID: object"
maxScaled["DATE_TIME"] = pd.to_datetime(pd.Series(maxScaled["DATE_TIME"])).astype(int) / 10 * 9 #We turn date into seconds integer 
maxScaled["SENSOR_INVERTER_ID"] = maxScaled["SENSOR_INVERTER_ID"].astype("category")
maxScaled["SOLAR_PANEL_INVERTER_ID"] = maxScaled["SOLAR_PANEL_INVERTER_ID"].astype("category")
label_encoder = preprocessing.LabelEncoder()
maxScaled = maxScaled.astype(str).apply(label_encoder.fit_transform)
X, y = maxScaled.drop('TOTAL_YIELD', axis=1), maxScaled[['TOTAL_YIELD']]


# Splitting
train_X, test_X, train_y, test_y = train_test_split(X, y,
                      test_size = 0.3, random_state = 123)


# Instantiation
xgb = xg.XGBRegressor(objective ='reg:squarederror',
                  n_estimators = 10, seed = 123)

# Fitting the model
xgb.fit(train_X, train_y)

# Predict the model
pred = xgb.predict(test_X)

# RMSE Computation
rmse = np.sqrt(MSE(test_y, pred))
print("RMSE{0}".format(rmse))