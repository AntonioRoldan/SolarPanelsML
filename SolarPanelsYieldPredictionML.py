import sklearn as sk 
from scipy import stats 
import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

dfWeatherSensorAndGenerationDataFrameMerge["DATE_TIME"] = pd.to_datetime(dfWeatherSensorAndGenerationDataFrameMerge["DATE_TIME"]) #We have to cast the date time to the date time type again because we have merged the date time value from the second table not the first which we casted before 


print(dfWeatherSensorAndGenerationDataFrameMerge.isnull().sum())

#There is an issue. Because our merged tables differ in row count with one having 3182 values and the other having 68779 values what happens is pandas adds extra null values to each table contained in extra rows from the second merged table to the first merged table and vice versa thus increasing the row count of the table resulting from our merging.
#and making sure all columns have the same row number. This increases the row count to 71000. So how do we arrange the data now? We need to solve this problem 
#First we are going to normalize the numerical data of our columns next we will pair up matching records without nulls in our table. (We need to pair them up because the first table's columns in our merged table, which by the way are the last two columns, add an equal row amount to the amount of second table's rows filling them with null values and the columns belonging to the second table do the same but adding the first table's row amount)
#The issue lies with the fact that we have null values in the same rows that we need to train which are 3182 (they are 3182 because our two tables don't have the same number of rows and we must take the smaller one) because the extra null valued rows of equal amount to the second merged table's rows within our merged table have been added at the BEGINNING of the first table columns in our merged table. AND the extra null valued rows added to the first merged table's rows of equal amount to the second merged table's rows have been added AT THE END
#So the 3182 rows that we need for our model contain two entire null columns one of which is our target variable.
#Basically we are going to reduce our entries down to 3182 but BEFORE THAT we will place the first 3182 row values after row index 3182 from our 68000 entry columns to the FIRST 3182 rows to replace their current null values (these columns are TOTAL_YIELD AND SOLAR_PANEL_INVERTED_ID which have 70000 entries because they come from the second table having 68.0000 and we are adding the extra rows with null values) 
#Then we impute remaining null values to the mean or zero depending on whether our values are numeric or not. How are we going to do this? 
#That we will show once we have normalized our numeric columns' data 
#So first we normalize data, next we  we place the first 3182 row values after row index 3182 from our 68000 entry columns to the FIRST 3182 rows to replace their current null values and then we reduce our entries down to 3182 in the merged table.
#To perform the last two steps we are going to need to create copies of the first merged table's columns in our merged table and store them in variables. One variable for each column which will be its own dataframe. We make the modifications to these data frames or isolated columns and store them back in the merged table overwriting the previous columns that we have copied with their copies we have modified. 

#Now we are going to normalize the numerical data of each column

#We are going to use the maximum absolute scaling normalization technique 

dfMaxScaled = dfWeatherSensorAndGenerationDataFrameMerge.copy()

for column in dfMaxScaled[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TOTAL_YIELD"]].columns:
    dfMaxScaled[column] = dfMaxScaled[column] / dfMaxScaled[column].abs().max()

print(dfMaxScaled)

print(dfMaxScaled.info())


#Now we are going to create two isolated data frames one for each of the two columns from the table second table, which is the one containing 68000 values instead of the one contaiining 3182 in our maxScaled data frame which are as we have said the ones belonging to the 680000 rows table after adding extra null value 
 
dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap = dfMaxScaled[["SOLAR_PANEL_INVERTER_ID"]].reset_index(drop=True)

dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap = dfMaxScaled[["TOTAL_YIELD"]].reset_index(drop=True) #We add reset index to avoid the duplicate row label bug from the pandas library

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

#Now we set the first 3182 rows of these columns to the first 3182 rows' values after index 3182 thus replacing their null values

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
dfMaxScaled["SOLAR_PANEL_INVERTER_ID"] = dfTemporaryDataFrameStoringPanelInverterIdColumnToMakeTheLastNRowsWithFirstRowsSwap["SOLAR_PANEL_INVERTER_ID"]
dfMaxScaled["TOTAL_YIELD"] = dfTemmporaryDataFrameStoringTotalYieldColumnToMakeTheLastNRowsWithFirstRowsSwap["TOTAL_YIELD"]


print("\n \n")

print(dfMaxScaled)

print("\n \n")

print(dfMaxScaled.info())
#We are going to get rid of 65595 rows so we will have our remaining 3182 rows. Remember we do this becuase of our row count mismatch between the two original tables that we have merged and whose merge we have just normalized and stored in the dfmaxScaled pandas data frame.
dfMaxScaled.drop(dfMaxScaled.tail(65595).index, inplace=True) 
#Next we impute the remaining null values there are some null values still because we cannot trim the table precisely and get rid of all null values.
dfMaxScaled["DATE_TIME"] = dfMaxScaled["DATE_TIME"].fillna(0)
dfMaxScaled["PLANT_ID"] = dfMaxScaled["PLANT_ID"].fillna(0)
dfMaxScaled["SENSOR_INVERTER_ID"] = dfMaxScaled["SENSOR_INVERTER_ID"].fillna(0)
dfMaxScaled["AMBIENT_TEMPERATURE"] = dfMaxScaled["AMBIENT_TEMPERATURE"].fillna(dfMaxScaled["AMBIENT_TEMPERATURE"].mean())
dfMaxScaled["MODULE_TEMPERATURE"] = dfMaxScaled["MODULE_TEMPERATURE"].fillna(dfMaxScaled["MODULE_TEMPERATURE"].mean())
print("\n \n")

print(dfMaxScaled)

print("\n \n")

print(dfMaxScaled.info())
#Next three lines are necessary to fix this error "ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, the experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:DATE_TIME: object, SENSOR_INVERTER_ID: object, SOLAR_PANEL_INVERTER_ID: object"
dfMaxScaled["DATE_TIME"] = pd.to_datetime(pd.Series(dfMaxScaled["DATE_TIME"])).astype(int) / 10 * 9 #We turn date into seconds integer 
dfMaxScaled["SENSOR_INVERTER_ID"] = dfMaxScaled["SENSOR_INVERTER_ID"].astype("category")
dfMaxScaled["SOLAR_PANEL_INVERTER_ID"] = dfMaxScaled["SOLAR_PANEL_INVERTER_ID"].astype("category")
#Now we are going to use the label encoder function to make sure there are no more than one data type per column which would give us an error 
label_encoder = preprocessing.LabelEncoder()
dfMaxScaled = dfMaxScaled.astype(str).apply(label_encoder.fit_transform)
#Finally we are going to prepare the data for our model without cross validation and next with cross validation 

X, y = dfMaxScaled.drop('TOTAL_YIELD', axis=1), dfMaxScaled[['TOTAL_YIELD']] #We set X and y

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

print("XGBOOST WITHOUT CROSS VALIDATION")
print("RMSE{0}".format(rmse))
print("We are going to check how many distinct values there are in our y column and compare it to our RMSE value the larger the difference between both with the RMSE being smaller the better the model's accuracy")
print("\n \n")
print("Distinct values in y column: {0}     RMSE value: {1}".format(dfMaxScaled["TOTAL_YIELD"].nunique(), rmse))

#Now we run xgboost with cross validation

print("\n \n")

print("XGBOOST WITH CROSS VALIDATION")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(xgb, X, y, cv=kf, scoring='r2') #Remember to set r2 for the scoring parameter when building a regression model which is obviously our case.

print("Cross-validation scores:", cv_scores)
print(f"Mean cross-validation score: {np.mean(cv_scores):.2f} +/- {np.std(cv_scores):.2f}")



