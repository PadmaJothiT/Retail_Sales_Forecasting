# Retail_Sales_Forecasting
Retail Sales Forecasting project aims to predict the sales price of a Departmental store.

# Retail Sales Forecasting dataset the project aims to analyze forecast department-wise weekly sales for each store for the following year using historical sales data, which will help make informed business decisions, especially during holiday periods and promotional markdown events. The streamlined Streamlit application integrates Exploratory Data Analysis (EDA) to find trends, patterns, and data insights. It offers users interactive tools to explore top-performing stores and departments, conduct insightful feature comparisons, and obtain personalized sales forecasts. With a commitment to delivering actionable insights, the project aims to optimize decision-making processes within the dynamic retail landscape.

#  Table of Contents:

## Key technologies used in this project
# Pandas
# Numpy
# Matplotlib.Plotly
# Seaborn
# Scikit
# Power BI
# Streamlit

## Steps involved in this Project:

# In this project three excel data contais about Sales data set, Feature data set, and store data set. Merging the data's to convert into a single DataFrame.

# Data Understanding: The dataset contains about Store detail,Date,Temperature- average weather in a region,Fuel_Price-cost of fuel in the region,Markdown1-5 related to promotional markdowns,CPI- the consumer price index,Unemployment the unemployment rate,IsHoliday whether the week is a special holiday week these are in the Feature data sets.In Sales data set it contains about the same Store,Dept there is 45 different departments,Date,Weekly_Sales sales for the give department different store,same IsHoliday.In Store data set details 45 different store details.
# Combining the dataset: Merging the 3 datasets with removing the unique columns in all datasets.
# Data Preprocessing: In Markdown columns some null values were presented replacing with them zero "0" will make the dataset more effective splitting date in month,year and day wise.
# Exploratory Data Analysis: As the data is no more noisy it need not need any statistical tools to the dataset, visualizing the dataset with the charts can make in eda part.
# Model Training and Evaluation : Three models have been used for model training and evaluation Linear Regression, Random Forest Regression and Gradient Boosting(XGBoost).Evaluation metics like mean_absolute_error,root_mean_squared_error, and model score
# Predicting the sale price: Using three regression models Random Forest Regression models have predicting the best accurate sale price. So, it has been used for predicting the sale price.
# Data Visualisation : Visualizing the data in Power BI with extracted data.
