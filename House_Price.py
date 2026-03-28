# Import Libraries

import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

# Import Load Data

@st.cache_data
def Load_Data(Csv):
    return pd.read_csv(Csv)

# Model Train 

@st.cache_resource
def Train_Model(Data):

    X = Data.drop("price",axis = 1)
    Y = Data["price"]

    num_cols = [num for num in ["area_sqft","bedrooms","bathrooms","floors","age","garage","garden","near_school","location_score"] if num in Data.columns]

    Numerical_Transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    
    Preprocessor = ColumnTransformer(
        transformers=[
            ("num", Numerical_Transformer, num_cols),
        ]
    )
    
    Model = Pipeline(
        steps=[
            ("Preprocessor", Preprocessor),
            ("Regression", LinearRegression())
        ]
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    
    Model.fit(X_train, Y_train)

    Y_Predict = Model.predict(X_test)

    Metrics = {
        "mean_absolute_error": int(mean_absolute_error(Y_test, Y_Predict)),
        "mean_squared_error": int(mean_squared_error(Y_test, Y_Predict)),
        "r2_score": float(r2_score(Y_test, Y_Predict))
    }

    return Model, Metrics

# Data Load 

st.sidebar.header("(1) Load Data")
Csv = st.sidebar.text_input(
    "Csv Path",
    value = "House_Price.csv",
    help = "Put The Csv File Name Into The Folder"
)

try:
    Data = Load_Data(Csv)
    st.sidebar.success("Data Loaded")
 
except:
    st.sidebar.error("Data Not Loaded")
    st.stop()

# Model Train

st.sidebar.header("(2) Model Train")
Train = st.sidebar.button("Train Re Train")

if Train:
    st.cache_resource.clear()
    st.sidebar.success("Model Train Sucessfull")

Model, Metrics = Train_Model(Data)

# Streamlit Page Setup

st.set_page_config(page_title = "House Price Preditions")
st.title("House Price Predictions")
st.caption("Data Science with M.L Project")

st.subheader("(1) Data Preview")
st.write(Data.head())


st.subheader("(2) Model-Metrics")
st.write({
        "mean_absolute_error" : f"{Metrics['mean_absolute_error'] :,}",
        "mean_squared_error" :  f"{Metrics['mean_squared_error'] :,}",
        "r2_score" : f"{round(Metrics['r2_score'],2)}"
})

# Try Prediction

st.subheader("(3) Try Prediction")

col1,col2,col3 = st.columns(3)

with col1:

    area_sqft = st.number_input("area_sqft",max_value = 6000, min_value = 400, value = 600, step = 50)
    bedrooms = st.number_input("bedrooms",max_value = 6, min_value = 0, value = 4 , step = 1)
    bathrooms = st.number_input("bathrooms",max_value = 5, min_value = 0, value = 4 , step = 1)
    
with col2:

    floors = st.number_input("floor",max_value = 3, min_value = 0, value = 2 , step = 1)
    garage = st.number_input("garage",max_value = 1, min_value = 0, value = 1 , step = 1)
    garden = st.number_input("garden",max_value = 1, min_value = 0, value = 1 , step = 1)

with col3:

    near_school = st.number_input("near_school",max_value = 1, min_value = 0, value = 1 , step = 1)
    location_score = st.number_input("location_score",max_value = 10, min_value = 0, value = 1 , step = 1)
    age = st.number_input("age",max_value = 40, min_value = 0, value = 1 , step = 1)

# Streamlit Ui Inputs

User_Input = {
    "area_sqft" : [area_sqft],
    "bedrooms" : [bedrooms],
    "bathrooms" : [bathrooms],
    "floors" :  [floors],
    "garage" : [garage],
    "garden" : [garden],
    "near_school" : [near_school],
    "location_score" : [location_score],
    "age" : [age],
}

User_Input = pd.DataFrame(User_Input)

# Prediction Button

if st.button("Predict Price"):

    Prediction = Model.predict(User_Input) 
    
    formatted_price = f"{Prediction[0]:,.0f}"
    
    st.balloons()
    st.success(f"Predicted House Price is : {formatted_price}")
    
    
        
