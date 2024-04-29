import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

st.set_page_config(
    page_title='Grocery Store Forecasting Challenge For Azubian',
    layout='wide',
    page_icon='📊'
)

# Project Title
st.title('Grocery Store Forecasting Challenge For Azubian')

#Read the Image
img = Image.open(".\image.png")
st.image(img, width=None, use_column_width=True)

# Read the CSV file and assign it to the 'data' variable
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    st.error("Dataset file not found.")
    data = None  # Assign None if file is not found to avoid NameError

# Button to view the dataset
if st.button('View Dataset'):
    if data is not None:
        st.write(data)


if data is not None:
             st.header('* The sales data is spanning from December 30, 1900, to August 10, 1902, the data is quite old,')



 # Sidebar inputs
st.sidebar.title('Filters')
Store = st.sidebar.selectbox('Select Store ID', data['store_id'].unique())
Holiday = st.sidebar.selectbox('Select Holiday', ['All', 'Holiday', 'RegularDay'])
    

feature = 'target'
feature2 = 'nbr_of_transactions'

# Filter the data
filtered_data = data[data['store_id'] == Store]
if Holiday == 'Holiday':
    filtered_data = filtered_data[filtered_data['is_holiday'] == 1]
elif Holiday == 'RegularDay':
    filtered_data = filtered_data[filtered_data['is_holiday'] == 0]

# Display features in separate columns
col1, col2 = st.columns(2)

with col1:

# Display feature 1
    st.subheader(f"Feature: Sales")

    # Display Total Sales
    total_sales = filtered_data['target'].sum()
    st.markdown(f'<span style="color: green;">Total Sales: {total_sales}</span>', unsafe_allow_html=True)
    fig = px.line(filtered_data, x='date', y=feature, title=f'Sales for: {Store}')
    st.plotly_chart(fig, use_container_width=True)

with col2:

# Display feature 2
    st.subheader(f"Feature: Transactions")

# Display Total Transaction 
    total_transactions = filtered_data['nbr_of_transactions'].sum()
    st.markdown(f'<span style="color: green;">Number of Transactions: {total_transactions}</span>', unsafe_allow_html=True)
    fig2 = px.line(filtered_data, x='date', y=feature2, title=f'Transaction for : {Store}')
    st.plotly_chart(fig2, use_container_width=True)


features, City, store_type = st.tabs(["FEATURES", "CITY", "STORE TYPE"]) 


with features:
    st.header('These are Features that are in the dataset and those that I had to engineree')

    st.markdown('*  date: The date of the transaction.')
    st.markdown('*  store_id: The ID of the store where the transaction occurred.')
    st.markdown('*  category_id: The category ID of the product.')
    st.markdown('*  target: The target variable, possibly the number of products purchased or sales amount.')
    st.markdown('*  onpromotion: Indicates whether the product was on promotion (1 for yes, 0 for no).')
    st.markdown('*  nbr_of_transactions: The number of transactions.')
    st.markdown('*  store_id: The ID of the store.')
    st.markdown('*  city: The city where the store is located.')
    st.markdown('*  type: The type of store.')
    st.markdown('*  cluster: The cluster to which the store belongs.')

with City:

 # Calculate total sales  for each City
    City_groups = data.groupby('city')['target'].sum().reset_index()

# Create a bar chart
    fig = px.bar(City_groups, x='city', y='target', labels={'city': 'City', 'target': 'Total Sale'},
             title='Total Sales by Cities', text='target')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig)


with store_type:

 # Calculate total sales  for each store type
    Store_sales = data.groupby('store_type')['target'].sum().reset_index()

# Create a bar chart
    fig = px.bar(Store_sales, x='store_type', y='target', labels={'store_type': 'Store Type', 'target': 'Total Sales'},
                  title='Total Sales by Store Type')
    fig.update_xaxes(categoryorder='total ascending')  # Sort stores by total sales 
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    st.plotly_chart(fig)


Trends, Resampling = st.tabs(["TREND & SEASONALITY", "RESAMPLING"]) 

with Trends:
       
     # Display unique cities in a dropdown
     selected_city = st.selectbox('Select City', data['city'].unique())

# Filter the data based on the selected city
     filtered_data = data[data['city'] == selected_city]

     # Convert selected_city to string
     selected_city_str = str(selected_city)

# Plot filtered data
     fig = px.line(filtered_data, x='date', y='target', title='Target vs. Date by Category for ' + selected_city)
     st.plotly_chart(fig)

with Resampling: 
     st.markdown('hello') 



