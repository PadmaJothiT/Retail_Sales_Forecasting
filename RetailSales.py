import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error


st.set_page_config(page_title= "Retail Sales Forecating",
                   page_icon= 'random',
                   layout= "wide",)

st.markdown("<h1 style='text-align: center; color: gold;'>WELCOME TO RETAIL SALES FORECASTING</h1>",unsafe_allow_html=True)

selected = option_menu(None, ["SALES PREDICTION"],
            icons=['cash-coin','trophy',"check-circle"],orientation='horizontal',default_index=0)

if selected == "SALES PREDICTION":
    try:
        if "SALES PREDICTION":
            store_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30',
                          '31','32','33','34','35','36','37','38','39','40','41','42','43','44','45']
            type_list = ['A','B','C']
            size_list = ['151315', '202307', '37392', '205863', '34875','202505','70713','155078',
                        '25833', '126512', '207499', '112238', '219622', '200898','123737', '57197',
                        '93188', '120653', '203819', '203742', '140167', '119557','114533', '128107',
                        '152513', '204184', '206302', '93638', '42988', '203750','203007','39690',
                        '158114', '103681','39910', '184109', '155083', '196321','41062', '118221']
            month_list = ['1','2',  '3',  '4',  '5',  '6',  '7',  '8',  '9', '10', '11', '12']
            year_list = ['2010', '2011', '2012']
            isholiday_list = ['0','1']
            temperature_list = ['']
    except:
        print(f"Please enter a valid value")