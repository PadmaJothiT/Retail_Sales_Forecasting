import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error


icon = Image.open(r'C:\Users\Padma Jothi\Desktop\Capstone\Retail Sales Prediction\sales.png')
st.set_page_config(page_title= "Retail Sales Forecating",
                   page_icon= icon,
                   layout= "wide",)

st.markdown("<h1 style='text-align: center; color: gold;'>WELCOME TO RETAIL SALES FORECASTING</h1>",unsafe_allow_html=True)

selected = option_menu(None, ["SALES PREDICTION"],
            icons=['cash-coin','trophy',"check-circle"],orientation='horizontal',default_index=0)

if selected == "SALES PREDICTION":
        
    if "SALES PREDICTION":
        col1,col2,col3 = st.columns(3,gap='large')
        with col1:
            #store
            store_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30',
                        '31','32','33','34','35','36','37','38','39','40','41','42','43','44','45']            
            store = st.selectbox('**Store**',store_list)

            #dept
            dept_list = ['1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '16', '17', '18',
                        '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
                        '36', '37', '38', '40', '41', '42', '44', '45', '46', '47', '48', '49', '51', '52', '54', '55', '56',
                        '58', '59', '60', '67', '71', '72', '74', '77', '78', '79', '80', '81', '82', '83', '85', '87', '90',
                        '91', '92', '93', '94', '95', '96', '97', '98', '99', '39', '50', '43', '65']
            dept = st.selectbox('**Dept**',dept_list)

            #type
            type_list = {'A':0,'B':1,'C':2}
            type = st.selectbox('**Type**',type_list)
            
            #size
            size_list = ['151315', '202307', '37392', '205863', '34875','202505','70713','155078',
                        '25833', '126512', '207499', '112238', '219622', '200898','123737', '57197',
                        '93188', '120653', '203819', '203742', '140167', '119557','114533', '128107',
                        '152513', '204184', '206302', '93638', '42988', '203750','203007','39690',
                        '158114', '103681','39910', '184109', '155083', '196321','41062', '118221']
            size = st.selectbox('**Size**',size_list)
            
            #isholiday
            isholiday_list = {"YES":0,"NO":1}
            isholiday = st.selectbox('**IsHoliday**',isholiday_list)

            #Date
            month_list = ['1','2','3','4','5','6','7','8','9','10','11','12']
            month = st.selectbox('**Month**',month_list)

        with col2:

            year_list = ['2010','2011','2012']
            year = st.selectbox('**Year**',year_list)
            

            week_list = [ '1',  '2',  '3',  '4', '5',  '6',  '7',  '8',  '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                        '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42',
                        '43', '44', '45', '46', '47', '48', '49', '50', '51', '52']
            week =st.selectbox('*Month**',month_list)

            #Temperature
            temperature = st.number_input('Enter the **:red[Temperature]** in Fahreneit**---> **:green[(min=5.54 & max=100.14)]**', value=90.0,min_value=5.54,max_value=100.14,)

            #Fuel_Price
            fuel_Price = st.number_input('Enter the **:red[Fuel Price]** ---> **:green[(min=2.472 & max=4.468)]**',value=3.67,min_value=2.472,max_value=4.468)

            #CPI
            cpi = st.number_input('Enter the **:red[CPI]** ----------> **:green[(min=126.0 & max=227.47)]**',value=211.77,min_value=126.0,max_value=227.47)

            #Unemployment
            unemployment = st.number_input('Enter the **:red[Unemployment Rate]** in percentage **:green[(min=3.879 & max=14.313)]**',value=8.106,min_value=3.879,max_value=14.313)

        with col3:

            #markdown
            markdown1 = st.number_input('Enter the **:orange[Markdown1]** in dollars -------- **:green[(min=0.27,max=88646.76)]**',value=2000.00,min_value=0.27,max_value=88646.76)
            markdown1= markdown1

            markdown2 = st.number_input('Enter the **:orange[Markdown2]** in dollars -------- **:green[(min=0.02,max=104519.54)]**',value=65000.00,min_value=0.02,max_value=104519.54)
            markdown2= markdown2

            markdown3=st.number_input('Enter the **:orange[Markdown3]** in dollars -------- **:green[(min=0.01,max=141630.61)]**',value=27000.00,min_value=0.01,max_value=141630.61)
            markdown3= markdown3

            markdown4=st.number_input('Enter the **:orange[Markdown4]** in dollars -------- **:green[(min=0.22,max=67474.85))]**',value=11200.00,min_value=0.22,max_value=67474.85)
            markdown4= markdown4

            markdown5=st.number_input('Enter the **:orange[Markdown5]** in dollars -------- **:green[(min=135.06,max=108519.28)]**',value=89000.00,min_value=135.06,max_value=108519.28)
            markdown5= markdown5
        
        button = st.button("Predict")
        
        if button:
            #Predict
            import pickle
            model=pickle.load(open(r'C:\Users\Padma Jothi\Desktop\Capstone\Retail Sales Prediction\XGBRegressor.pkl', 'rb'))

            def safe_convert(value):
                try:
                    return float(value)
                except ValueError:
                    return np.nan
                
            input_data = [store, dept, isholiday, temperature, fuel_Price, markdown1, markdown2, markdown3, markdown4, markdown5, cpi, unemployment, type, size, month, year, week]
            cleaned_data = [safe_convert(store),
                            safe_convert(dept),
                            safe_convert(isholiday),
                            safe_convert(temperature),
                            safe_convert(fuel_Price),
                            safe_convert(markdown1),
                            safe_convert(markdown2),
                            safe_convert(markdown3),
                            safe_convert(markdown4),
                            safe_convert(markdown5),
                            safe_convert(cpi),
                            safe_convert(unemployment),
                            safe_convert(type),
                            safe_convert(size),
                            safe_convert(month),
                            safe_convert(year),
                            safe_convert(week)
                            ]
            
            input_data = np.array(cleaned_data).reshape(1,-1)
            # Make a prediction
            y_pred = model.predict(input_data)
            st.success(f"**Predicted Sales price: {y_pred}**")
        else: 
            st.error("Please enter valid values")
