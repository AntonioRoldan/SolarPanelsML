import sklearn as sk 
from scipy import stats 
import pandas as pd 
import matplotlib.pyplot as plt 



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

print(dfGenerationData.info()) #We show the column types and names and see if there are missing values this way too by checking non null values.

print("\n \n")

print(dfWeatherSensorData.isnull().sum())

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

#Now we are going to normalize the numerical data of each column

#We are going to use the maximum absolute scaling normalization technique 

max_scaled = dfWeatherSensorAndGenerationDataFrameMerge.copy()

for column in max_scaled[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TOTAL_YIELD"]]:
    max_scaled[column] = max_scaled[column] / max_scaled[column].abs().max()

print(max_scaled.info())

max_scaled.plot(kind='bar')
plt.show()